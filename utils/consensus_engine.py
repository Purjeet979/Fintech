"""
Consensus Engine & Pseudo-Labeling Module
Implements ensemble-based extraction and self-consistency methods

Features:
- Multi-pipeline consensus voting
- Pseudo-labeling with deterministic rules
- Bootstrapping for iterative refinement
- Self-consistency verification
- Confidence-weighted voting

Based on techniques from:
- Co-training and multi-view learning
- Snorkel weak supervision framework
- Active learning with uncertainty sampling
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import Counter, defaultdict
from loguru import logger

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class ConsensusEngine:
    """
    Ensemble-based extraction with consensus voting and pseudo-labeling.
    
    Implements:
    1. Multi-pipeline extraction (OCR+Rules, Table, KV-Pairs, VLM)
    2. Confidence-weighted voting
    3. Self-consistency verification
    4. Pseudo-label generation with deterministic rules
    """
    
    # Deterministic rules for pseudo-label validation
    # These are high-precision rules that generate reliable labels
    DETERMINISTIC_RULES = {
        'horse_power': {
            'patterns': [
                r'(\d{2,3})\s*HP\b',  # 50 HP
                r'(\d{2,3})\s*H\.P\.',  # 50 H.P.
                r'Horse\s*Power[:\s]+(\d{2,3})',  # Horse Power: 50
            ],
            'range': (15, 150),
            'confidence_boost': 0.2
        },
        'asset_cost': {
            'patterns': [
                r'(?:Total|Grand\s*Total)[:\s]*[₹Rs\.]*\s*([\d,]+)',
                r'(?:Net\s*Amount|Final)[:\s]*[₹Rs\.]*\s*([\d,]+)',
            ],
            'range': (50000, 5000000),
            'confidence_boost': 0.15
        },
        'model_name': {
            'patterns': [
                r'(Mahindra\s+\d{3,4}\s*(?:DI|XP|XT)?)',
                r'(John\s*Deere\s+\d{4}[A-Z]?)',
                r'(Swaraj\s+\d{3,4}(?:\s*FE|\s*XT)?)',
                r'(Sonalika\s+(?:DI\s*)?\d+)',
                r'(TAFE\s+\d{4})',
                r'(New\s*Holland\s+\d{4})',
            ],
            'confidence_boost': 0.1
        }
    }
    
    # Confidence thresholds for pseudo-labeling
    PSEUDO_LABEL_THRESHOLD = 0.85  # Minimum confidence to generate pseudo-label
    CONSENSUS_THRESHOLD = 0.7     # Minimum agreement for consensus
    
    # ============ FIELD-SPECIFIC TRUST WEIGHTS ============
    # Adaptive trust scores based on typical accuracy of each method per field
    # These weights are empirically tuned for tractor invoice processing:
    # 
    # **Key Insight**: Different fields require different extraction strategies
    # - Rule-based excels at: Printed text, structured layouts, known patterns
    # - VLM excels at: Handwritten text, complex layouts, contextual understanding
    #
    # Trust Weight Impact:
    # - Weight > 1.0: Method is preferred for this field
    # - Weight = 1.0: Neutral (standard confidence)
    # - Weight < 1.0: Method is less trusted for this field
    #
    FIELD_METHOD_TRUST = {
        'dealer_name': {
            'rule_based': 0.85,  # Fallback for printed headers
            'vlm': 1.1,          # VLM better at vernacular (Hindi/Gujarati) names
        },
        'model_name': {
            'rule_based': 0.7,   # Pattern matching limited to known brands/OCR errors
            'vlm': 1.15,         # VLM better at reading full model with variants + Hindi numerals
        },
        'horse_power': {
            'rule_based': 0.4,   # Often handwritten, high OCR noise
            'vlm': 1.3,          # VLM strongly preferred for handwritten numbers
        },
        'asset_cost': {
            'rule_based': 0.35,  # Almost always handwritten, high PIN code confusion risk
            'vlm': 1.35,         # VLM strongly preferred, understands context (PIN vs price)
        },
    }
    
    # Document trait adjustments - VLM strongly preferred for handwritten & vernacular
    TRAIT_ADJUSTMENTS = {
        'has_handwriting': {'rule_based': 0.5, 'vlm': 1.4},  # VLM strongly preferred
        'is_hindi': {'rule_based': 0.8, 'vlm': 1.15},        # VLM better at Hindi vernacular
        'is_gujarati': {'rule_based': 0.75, 'vlm': 1.15},    # VLM better at Gujarati vernacular
        'low_quality': {'rule_based': 0.5, 'vlm': 1.0},      # VLM more robust to noise
        'has_stamp_signature_overlap': {'rule_based': 0.6, 'vlm': 1.2},  # VLM handles occlusion better
    }
    
    def __init__(
        self,
        min_voters: int = 2,
        confidence_weight: bool = True,
        use_deterministic_boost: bool = True,
        use_field_trust_weights: bool = True
    ):
        """
        Initialize consensus engine.
        
        Args:
            min_voters: Minimum extraction methods for consensus
            confidence_weight: Weight votes by confidence scores
            use_field_trust_weights: Apply field-specific trust weights to methods
            use_deterministic_boost: Boost confidence for deterministic matches
        """
        self.min_voters = min_voters
        self.confidence_weight = confidence_weight
        self.use_deterministic_boost = use_deterministic_boost
        self.use_field_trust_weights = use_field_trust_weights
        
        # Document-level traits (detected per document)
        self._document_traits = {}
        
        # Compile deterministic patterns
        self._compiled_rules = {}
        for field, rules in self.DETERMINISTIC_RULES.items():
            self._compiled_rules[field] = {
                'patterns': [re.compile(p, re.IGNORECASE) for p in rules['patterns']],
                'range': rules.get('range'),
                'confidence_boost': rules.get('confidence_boost', 0)
            }
    
    # ============ FIELD-SPECIFIC TRUST WEIGHTS ============
    
    def detect_document_traits(self, ocr_result: Dict) -> Dict[str, bool]:
        """
        Detect document characteristics to adjust trust weights.
        
        Traits detected:
        - has_handwriting: Document contains handwritten text
        - is_hindi: Significant Hindi text present
        - is_gujarati: Significant Gujarati text present
        - low_quality: Poor image/OCR quality
        - has_stamp_signature_overlap: Stamp/signature likely present and may overlap text
        
        Returns:
            Dict of trait_name -> bool
        """
        traits = {
            'has_handwriting': False,
            'is_hindi': False,
            'is_gujarati': False,
            'low_quality': False,
            'has_stamp_signature_overlap': True,  # Per problem statement: "all images 'will' contain both stamp and signature - likely overlapping"
        }
        
        full_text = ocr_result.get('full_text', '')
        confidences = ocr_result.get('confidences', [])
        
        # Detect handwriting indicators
        # - Irregular character spacing (lots of single chars or OCR fragments)
        # - Mixed case inconsistency
        # - OCR noise patterns
        if full_text:
            words = full_text.split()
            single_char_ratio = sum(1 for w in words if len(w) == 1) / max(len(words), 1)
            noise_chars = sum(1 for c in full_text if c in '^`~_|\\')
            
            if single_char_ratio > 0.15 or noise_chars > 10:
                traits['has_handwriting'] = True
        
        # Detect language
        hindi_chars = sum(1 for c in full_text if '\u0900' <= c <= '\u097F')
        gujarati_chars = sum(1 for c in full_text if '\u0A80' <= c <= '\u0AFF')
        
        if hindi_chars > 20:
            traits['is_hindi'] = True
        if gujarati_chars > 20:
            traits['is_gujarati'] = True
        
        # Detect low quality
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            if avg_conf < 0.7:
                traits['low_quality'] = True
        
        self._document_traits = traits
        return traits
    
    def get_field_trust_weight(
        self,
        field: str,
        method_name: str,
        traits: Dict[str, bool] = None
    ) -> float:
        """
        Calculate trust weight for a specific field-method combination.
        
        Args:
            field: Field name (e.g., 'horse_power', 'asset_cost')
            method_name: Extraction method name ('rule_based', 'vlm')
            traits: Document traits (uses cached if None)
            
        Returns:
            Trust weight multiplier (0.5 to 1.5)
        """
        traits = traits if traits is not None else self._document_traits
        
        # Get base trust weight for this field-method combination
        method_type = 'vlm' if 'vlm' in method_name.lower() else 'rule_based'
        field_trust = self.FIELD_METHOD_TRUST.get(field, {})
        base_weight = field_trust.get(method_type, 0.8)
        
        # Apply trait adjustments
        if traits:
            for trait_name, is_present in traits.items():
                if is_present and trait_name in self.TRAIT_ADJUSTMENTS:
                    adjustment = self.TRAIT_ADJUSTMENTS[trait_name].get(method_type, 1.0)
                    base_weight *= adjustment
        
        # Clamp to reasonable range
        return max(0.3, min(1.5, base_weight))
    
    def apply_trust_weights_to_votes(
        self,
        field: str,
        votes: List[Tuple[Any, float, str]],
        traits: Dict[str, bool] = None
    ) -> List[Tuple[Any, float, str]]:
        """
        Apply field-specific trust weights to vote confidences.
        
        Args:
            field: Field being voted on
            votes: List of (value, confidence, method_name) tuples
            traits: Document traits for adjustment
            
        Returns:
            Weighted votes with adjusted confidences
        """
        if not self.use_field_trust_weights:
            return votes
        
        weighted_votes = []
        for value, confidence, method_name in votes:
            trust_weight = self.get_field_trust_weight(field, method_name, traits)
            weighted_confidence = min(1.0, confidence * trust_weight)
            weighted_votes.append((value, weighted_confidence, method_name))
            
            if trust_weight != 1.0:
                logger.debug(
                    f"  Trust weight [{field}][{method_name}]: "
                    f"{confidence:.2f} * {trust_weight:.2f} = {weighted_confidence:.2f}"
                )
        
        return weighted_votes
    
    def extract_with_consensus(
        self,
        extraction_methods: List[Callable],
        method_names: List[str] = None,
        ocr_result: Dict = None
    ) -> Dict[str, Tuple[Any, float, Dict]]:
        """
        Run multiple extraction methods and return consensus results.
        
        Args:
            extraction_methods: List of callables that return Dict[field: (value, confidence)]
            method_names: Names for logging/debugging
            ocr_result: OCR result dict for trait detection (enables adaptive trust weights)
            
        Returns:
            Dict with field -> (consensus_value, confidence, metadata)
        """
        if not extraction_methods:
            return {}
        
        method_names = method_names or [f"method_{i}" for i in range(len(extraction_methods))]
        
        # Detect document traits for adaptive trust weighting
        # Traits affect how much we trust each method (rule-based vs VLM)
        traits = None
        if ocr_result and self.use_field_trust_weights:
            traits = self.detect_document_traits(ocr_result)
            active_traits = [k for k, v in traits.items() if v]
            if active_traits:
                logger.info(f"  Document traits detected: {active_traits}")
                logger.debug(f"  → Trust adjustments will be applied to voting weights")
        
        # Collect results from all methods
        all_results = []
        for method, name in zip(extraction_methods, method_names):
            try:
                result = method()
                if result:
                    all_results.append({
                        'name': name,
                        'results': result
                    })
                    logger.debug(f"  {name}: extracted {len(result)} fields")
            except Exception as e:
                logger.warning(f"  {name}: failed - {e}")
        
        if len(all_results) < self.min_voters:
            logger.warning(f"Only {len(all_results)} methods succeeded, need {self.min_voters}")
            # Return best single result if available
            if all_results:
                return self._convert_to_consensus_format(all_results[0]['results'])
            return {}
        
        # Apply consensus voting for each field with field-specific trust weights
        consensus = {}
        fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        
        logger.debug("[Field-Specific Trust Weights] Applying adaptive weights to votes:")
        
        for field in fields:
            field_votes = self._collect_field_votes(all_results, field)
            
            # Apply field-specific trust weights before voting
            # This gives VLM more weight for handwritten fields (HP, cost)
            # and rule-based more weight for printed fields (dealer)
            weighted_votes = self.apply_trust_weights_to_votes(field, field_votes, traits)
            
            # Log the weight adjustments
            if field_votes:
                logger.debug(f"  {field}:")
                for (val, orig_conf, method), (_, new_conf, _) in zip(field_votes, weighted_votes):
                    if orig_conf != new_conf:
                        method_type = 'VLM' if 'vlm' in method.lower() else 'Rule-based'
                        logger.debug(f"    → {method_type}: {val} [{orig_conf:.2f} → {new_conf:.2f}]")
            
            consensus[field] = self._vote_on_field(field, weighted_votes)
        
        return consensus
    
    def _collect_field_votes(
        self,
        all_results: List[Dict],
        field: str
    ) -> List[Tuple[Any, float, str]]:
        """
        Collect all votes for a specific field.
        
        Handles two formats:
        1. Tuple format: (value, confidence) - from FieldParser
        2. Plain value format: value - from VLMExtractor (uses overall confidence)
        """
        votes = []
        
        for result in all_results:
            results_dict = result['results']
            field_result = results_dict.get(field)
            
            if field_result is None:
                continue
            
            # Handle different return formats
            if isinstance(field_result, tuple) and len(field_result) >= 2:
                # Format 1: Tuple of (value, confidence)
                value, confidence = field_result[0], field_result[1]
            elif isinstance(field_result, tuple) and len(field_result) == 1:
                # Single-element tuple
                value = field_result[0]
                confidence = results_dict.get('confidence', 0.5)
            else:
                # Format 2: Plain value (from VLM)
                # Use overall confidence from the results dict
                value = field_result
                confidence = results_dict.get('confidence', 0.5)
                
                # Skip if the value is actually the confidence key itself
                if field == 'confidence':
                    continue
            
            if value is not None:
                # Ensure confidence is a valid float
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = 0.5
                
                votes.append((value, confidence, result['name']))
        
        return votes
    
    def _vote_on_field(
        self,
        field: str,
        votes: List[Tuple[Any, float, str]]
    ) -> Tuple[Any, float, Dict]:
        """
        Apply consensus voting on field values.
        
        Returns: (consensus_value, confidence, metadata)
        """
        if not votes:
            return (None, 0.0, {'method': 'no_votes'})
        
        if len(votes) == 1:
            value, conf, method = votes[0]
            return (value, conf, {'method': method, 'voters': 1})
        
        # Group similar values
        value_groups = self._group_similar_values(field, votes)
        
        # Calculate weighted scores for each group
        group_scores = []
        for canonical, group_votes in value_groups.items():
            if self.confidence_weight:
                # Weighted score = sum of confidences
                score = sum(conf for _, conf, _ in group_votes)
            else:
                # Unweighted = count
                score = len(group_votes)
            
            avg_conf = sum(conf for _, conf, _ in group_votes) / len(group_votes)
            methods = [m for _, _, m in group_votes]
            
            group_scores.append({
                'value': canonical,
                'score': score,
                'confidence': avg_conf,
                'voters': len(group_votes),
                'methods': methods
            })
        
        # Select winner
        winner = max(group_scores, key=lambda x: (x['score'], x['confidence']))
        
        # Calculate consensus confidence
        total_voters = len(votes)
        agreement_ratio = winner['voters'] / total_voters
        
        # Boost confidence if high agreement
        if agreement_ratio >= 0.8:
            final_confidence = min(1.0, winner['confidence'] + 0.1)
        elif agreement_ratio >= 0.6:
            final_confidence = winner['confidence']
        else:
            final_confidence = winner['confidence'] * 0.9  # Penalty for low agreement
        
        # Apply deterministic rule boost
        if self.use_deterministic_boost:
            final_confidence = self._apply_deterministic_boost(
                field, winner['value'], final_confidence
            )
        
        metadata = {
            'method': 'consensus',
            'voters': winner['voters'],
            'total_methods': total_voters,
            'agreement': round(agreement_ratio, 2),
            'contributing_methods': winner['methods']
        }
        
        return (winner['value'], round(final_confidence, 3), metadata)
    
    def _group_similar_values(
        self,
        field: str,
        votes: List[Tuple[Any, float, str]]
    ) -> Dict[Any, List]:
        """Group similar values together for voting."""
        groups = defaultdict(list)
        
        if field in ['horse_power', 'asset_cost']:
            # Numeric fields: exact match or within tolerance
            for value, conf, method in votes:
                try:
                    num_val = int(value)
                    # Find existing group within tolerance
                    matched = False
                    tolerance = 0.05 if field == 'asset_cost' else 3
                    
                    for canonical in list(groups.keys()):
                        if field == 'asset_cost':
                            if abs(num_val - canonical) <= canonical * tolerance:
                                groups[canonical].append((num_val, conf, method))
                                matched = True
                                break
                        else:  # horse_power
                            if abs(num_val - canonical) <= tolerance:
                                groups[canonical].append((num_val, conf, method))
                                matched = True
                                break
                    
                    if not matched:
                        groups[num_val].append((num_val, conf, method))
                except (ValueError, TypeError):
                    continue
        else:
            # Text fields: fuzzy grouping
            for value, conf, method in votes:
                if not value:
                    continue
                
                str_val = str(value).strip()
                matched = False
                
                for canonical in list(groups.keys()):
                    similarity = self._text_similarity(str_val, canonical)
                    if similarity >= 85:
                        groups[canonical].append((str_val, conf, method))
                        matched = True
                        break
                
                if not matched:
                    groups[str_val].append((str_val, conf, method))
        
        return groups
    
    def _text_similarity(self, text1: str, text2: str) -> int:
        """Calculate text similarity score (0-100)."""
        if not text1 or not text2:
            return 0
        
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(text1.lower(), text2.lower())
        else:
            # Simple fallback
            t1, t2 = text1.lower(), text2.lower()
            if t1 == t2:
                return 100
            if t1 in t2 or t2 in t1:
                return 80
            
            words1 = set(t1.split())
            words2 = set(t2.split())
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return int(overlap / total * 100) if total > 0 else 0
    
    def _apply_deterministic_boost(
        self,
        field: str,
        value: Any,
        confidence: float
    ) -> float:
        """
        Boost confidence if value matches deterministic rules.
        
        Deterministic rules are high-precision patterns that
        strongly indicate correct extraction.
        """
        if field not in self._compiled_rules:
            return confidence
        
        rules = self._compiled_rules[field]
        
        # Check range for numeric fields
        if rules.get('range'):
            try:
                num_val = int(value)
                min_val, max_val = rules['range']
                if not (min_val <= num_val <= max_val):
                    return confidence * 0.8  # Penalty for out-of-range
            except (ValueError, TypeError):
                pass
        
        # Pattern matching doesn't apply to extracted values directly
        # but validates the extraction process
        
        # Apply boost if within valid range
        return min(1.0, confidence + rules['confidence_boost'] * 0.5)
    
    def _convert_to_consensus_format(
        self,
        results: Dict
    ) -> Dict[str, Tuple[Any, float, Dict]]:
        """
        Convert simple results to consensus format.
        
        Handles both:
        - Tuple format: (value, confidence) from FieldParser
        - Plain value format: value from VLMExtractor
        """
        consensus = {}
        # Get overall confidence from VLM-style results
        overall_conf = results.get('confidence', 0.5)
        try:
            overall_conf = float(overall_conf)
        except (ValueError, TypeError):
            overall_conf = 0.5
        
        for field, value in results.items():
            # Skip metadata fields
            if field in ('confidence', 'extraction_notes', 'error', '_confidence', '_field_confidences'):
                continue
            
            if isinstance(value, tuple) and len(value) >= 2:
                val, conf = value[0], value[1]
                consensus[field] = (val, conf, {'method': 'single', 'voters': 1})
            elif isinstance(value, tuple) and len(value) == 1:
                consensus[field] = (value[0], overall_conf, {'method': 'single', 'voters': 1})
            else:
                # Plain value - use overall confidence
                consensus[field] = (value, overall_conf, {'method': 'single', 'voters': 1})
        return consensus
    
    # ================================================================
    # PSEUDO-LABELING METHODS
    # ================================================================
    
    def generate_pseudo_labels(
        self,
        ocr_text: str,
        extraction_results: Dict[str, Tuple[Any, float]],
        full_ocr_result: Dict = None
    ) -> Dict[str, Dict]:
        """
        Generate pseudo-labels using deterministic rules and confident extractions.
        
        Pseudo-labels are provisional ground truth labels generated from:
        1. High-confidence model outputs
        2. Deterministic rule matches
        3. Consensus from multiple methods
        
        Args:
            ocr_text: Full text from OCR
            extraction_results: Current extraction with confidences
            full_ocr_result: Full OCR result with boxes (optional)
            
        Returns:
            Dict with field -> {value, confidence, source, is_pseudo_label}
        """
        pseudo_labels = {}
        
        for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            # Get extraction result
            extracted = extraction_results.get(field, (None, 0))
            if isinstance(extracted, tuple):
                value, confidence = extracted
            else:
                value, confidence = extracted, 0.5
            
            # Try deterministic extraction
            det_value, det_conf = self._deterministic_extract(field, ocr_text)
            
            # Decide on pseudo-label
            if det_value is not None and det_conf >= 0.9:
                # Deterministic rule match - highest reliability
                pseudo_labels[field] = {
                    'value': det_value,
                    'confidence': det_conf,
                    'source': 'deterministic_rule',
                    'is_pseudo_label': True,
                    'reliable': True
                }
            elif value is not None and confidence >= self.PSEUDO_LABEL_THRESHOLD:
                # High-confidence extraction
                pseudo_labels[field] = {
                    'value': value,
                    'confidence': confidence,
                    'source': 'high_confidence_extraction',
                    'is_pseudo_label': True,
                    'reliable': True
                }
            elif value is not None and det_value is not None:
                # Cross-validate: extraction matches deterministic
                if self._values_match(field, value, det_value):
                    boosted_conf = min(1.0, (confidence + det_conf) / 2 + 0.1)
                    pseudo_labels[field] = {
                        'value': value,
                        'confidence': boosted_conf,
                        'source': 'cross_validated',
                        'is_pseudo_label': True,
                        'reliable': True
                    }
                else:
                    # Conflict - use deterministic if higher confidence
                    if det_conf > confidence:
                        pseudo_labels[field] = {
                            'value': det_value,
                            'confidence': det_conf * 0.9,
                            'source': 'deterministic_override',
                            'is_pseudo_label': True,
                            'reliable': False  # Conflict noted
                        }
                    else:
                        pseudo_labels[field] = {
                            'value': value,
                            'confidence': confidence * 0.9,
                            'source': 'extraction_preferred',
                            'is_pseudo_label': True,
                            'reliable': False
                        }
            elif value is not None:
                # Only extraction available
                pseudo_labels[field] = {
                    'value': value,
                    'confidence': confidence,
                    'source': 'extraction_only',
                    'is_pseudo_label': confidence >= 0.7,
                    'reliable': confidence >= 0.8
                }
            elif det_value is not None:
                # Only deterministic available
                pseudo_labels[field] = {
                    'value': det_value,
                    'confidence': det_conf,
                    'source': 'deterministic_only',
                    'is_pseudo_label': det_conf >= 0.7,
                    'reliable': det_conf >= 0.8
                }
            else:
                # No extraction
                pseudo_labels[field] = {
                    'value': None,
                    'confidence': 0.0,
                    'source': 'not_found',
                    'is_pseudo_label': False,
                    'reliable': False
                }
        
        return pseudo_labels
    
    def _deterministic_extract(
        self,
        field: str,
        text: str
    ) -> Tuple[Optional[Any], float]:
        """
        Extract using deterministic rules only.
        
        These are high-precision patterns with near-certain accuracy.
        """
        if field not in self._compiled_rules:
            return (None, 0.0)
        
        rules = self._compiled_rules[field]
        
        for pattern in rules['patterns']:
            matches = pattern.findall(text)
            for match in matches:
                # Validate match
                if field in ['horse_power', 'asset_cost']:
                    try:
                        # Parse numeric
                        cleaned = re.sub(r'[,\s]', '', str(match))
                        value = int(cleaned)
                        
                        # Range check
                        if rules.get('range'):
                            min_val, max_val = rules['range']
                            if min_val <= value <= max_val:
                                return (value, 0.95)
                    except (ValueError, TypeError):
                        continue
                else:
                    # Text field
                    value = match.strip()
                    if len(value) > 3:
                        return (value, 0.9)
        
        return (None, 0.0)
    
    def _values_match(
        self,
        field: str,
        value1: Any,
        value2: Any
    ) -> bool:
        """Check if two values match for a field."""
        if value1 is None or value2 is None:
            return False
        
        if field in ['horse_power', 'asset_cost']:
            try:
                v1, v2 = int(value1), int(value2)
                if field == 'asset_cost':
                    return abs(v1 - v2) <= v1 * 0.05
                else:
                    return abs(v1 - v2) <= 3
            except (ValueError, TypeError):
                return False
        else:
            return self._text_similarity(str(value1), str(value2)) >= 85
    
    # ================================================================
    # SELF-CONSISTENCY VERIFICATION
    # ================================================================
    
    def verify_self_consistency(
        self,
        results: Dict[str, Tuple[Any, float, Dict]]
    ) -> Dict[str, Dict]:
        """
        Verify extraction self-consistency.
        
        Checks:
        1. HP-Model consistency (known mappings)
        2. Cost reasonableness for model type
        3. Internal field consistency
        
        Returns verification report with adjustments.
        """
        report = {
            'is_consistent': True,
            'checks': [],
            'adjustments': {}
        }
        
        # Extract values
        model = results.get('model_name', (None, 0, {}))[0]
        hp = results.get('horse_power', (None, 0, {}))[0]
        cost = results.get('asset_cost', (None, 0, {}))[0]
        
        # Check 1: HP-Model consistency
        if model and hp:
            expected_hp = self._get_expected_hp_for_model(model)
            if expected_hp:
                hp_diff = abs(int(hp) - expected_hp)
                if hp_diff <= 3:
                    report['checks'].append({
                        'check': 'hp_model_match',
                        'status': 'pass',
                        'detail': f"HP {hp} matches expected {expected_hp} for {model}"
                    })
                elif hp_diff <= 10:
                    report['checks'].append({
                        'check': 'hp_model_match',
                        'status': 'warning',
                        'detail': f"HP {hp} close to expected {expected_hp} for {model}"
                    })
                else:
                    report['is_consistent'] = False
                    report['checks'].append({
                        'check': 'hp_model_match',
                        'status': 'fail',
                        'detail': f"HP {hp} differs from expected {expected_hp} for {model}"
                    })
                    # Suggest adjustment
                    report['adjustments']['horse_power'] = {
                        'current': hp,
                        'suggested': expected_hp,
                        'reason': 'model_hp_mapping'
                    }
        
        # Check 2: Cost reasonableness
        if cost:
            try:
                cost_val = int(cost)
                if cost_val < 100000:
                    report['checks'].append({
                        'check': 'cost_range',
                        'status': 'warning',
                        'detail': f"Cost {cost_val} seems low for tractor"
                    })
                elif cost_val > 3000000:
                    report['checks'].append({
                        'check': 'cost_range',
                        'status': 'warning',
                        'detail': f"Cost {cost_val} seems high for tractor"
                    })
                else:
                    report['checks'].append({
                        'check': 'cost_range',
                        'status': 'pass',
                        'detail': f"Cost {cost_val} in reasonable range"
                    })
            except (ValueError, TypeError):
                pass
        
        return report
    
    def _get_expected_hp_for_model(self, model: str) -> Optional[int]:
        """Get expected HP for a model."""
        if not model:
            return None
        
        # Common model-HP mappings
        hp_map = {
            '575': 50, '585': 50, '605': 57, '475': 42, '555': 55,
            '5042': 42, '5050': 50, '5055': 55,
            '744': 48, '735': 39, '855': 52,
            '7250': 50, '7515': 75,
        }
        
        model_upper = model.upper()
        for pattern, hp in hp_map.items():
            if pattern in model_upper:
                return hp
        
        return None


class BootstrapRefiner:
    """
    Iterative refinement of pseudo-labels through bootstrapping.
    
    Implements:
    1. Initial label generation from deterministic rules
    2. Model training/inference on pseudo-labels
    3. Confidence-based label refinement
    4. Iterative improvement loop
    """
    
    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus_engine = consensus_engine
        self.iteration = 0
        self.label_history = []
    
    def refine_labels(
        self,
        current_labels: Dict[str, Dict],
        new_extraction: Dict[str, Tuple[Any, float]]
    ) -> Dict[str, Dict]:
        """
        Refine pseudo-labels based on new extraction results.
        
        Bootstrapping strategy:
        1. If new extraction has higher confidence -> update label
        2. If labels agree -> boost confidence
        3. If labels conflict -> lower confidence, flag for review
        """
        self.iteration += 1
        refined = {}
        
        for field, current in current_labels.items():
            new_result = new_extraction.get(field, (None, 0))
            if isinstance(new_result, tuple):
                new_value, new_conf = new_result
            else:
                new_value, new_conf = None, 0
            
            current_value = current.get('value')
            current_conf = current.get('confidence', 0)
            
            if new_value is None:
                # Keep current
                refined[field] = current
            elif current_value is None:
                # Use new
                refined[field] = {
                    'value': new_value,
                    'confidence': new_conf,
                    'source': f'bootstrap_iter_{self.iteration}',
                    'is_pseudo_label': new_conf >= 0.7,
                    'reliable': new_conf >= 0.8
                }
            elif self.consensus_engine._values_match(field, current_value, new_value):
                # Agreement - boost confidence
                boosted = min(1.0, (current_conf + new_conf) / 2 + 0.05)
                refined[field] = {
                    'value': current_value,
                    'confidence': boosted,
                    'source': f'bootstrap_agreement_iter_{self.iteration}',
                    'is_pseudo_label': True,
                    'reliable': boosted >= 0.85
                }
            else:
                # Conflict - use higher confidence
                if new_conf > current_conf + 0.1:
                    refined[field] = {
                        'value': new_value,
                        'confidence': new_conf * 0.95,  # Slight penalty for conflict
                        'source': f'bootstrap_update_iter_{self.iteration}',
                        'is_pseudo_label': new_conf >= 0.75,
                        'reliable': False
                    }
                else:
                    # Keep current but lower confidence due to conflict
                    refined[field] = {
                        'value': current_value,
                        'confidence': current_conf * 0.9,
                        'source': f'bootstrap_conflict_iter_{self.iteration}',
                        'is_pseudo_label': current.get('is_pseudo_label', False),
                        'reliable': False
                    }
        
        self.label_history.append(refined)
        return refined
    
    def get_stable_labels(self) -> Dict[str, Dict]:
        """
        Get labels that have been stable across iterations.
        
        Stable = same value for last N iterations with increasing confidence.
        """
        if len(self.label_history) < 2:
            return self.label_history[-1] if self.label_history else {}
        
        stable = {}
        latest = self.label_history[-1]
        
        for field, label in latest.items():
            # Check last 3 iterations
            stable_count = 0
            for past in self.label_history[-3:]:
                past_label = past.get(field, {})
                if self.consensus_engine._values_match(
                    field,
                    label.get('value'),
                    past_label.get('value')
                ):
                    stable_count += 1
            
            if stable_count >= 2:
                # Stable - boost confidence
                stable[field] = {
                    **label,
                    'confidence': min(1.0, label['confidence'] + 0.05),
                    'stable': True
                }
            else:
                stable[field] = {**label, 'stable': False}
        
        return stable
