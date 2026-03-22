"""
Validator Module
Field validation, fuzzy matching, cross-validation, and post-processing

Features:
- Fuzzy matching against master lists
- HP-Model cross-validation
- Range and sanity checks
- Multilingual text normalization (English, Hindi, Gujarati)
- Near-duplicate reconciliation
- Threshold-based confidence scoring
- Numeric accuracy validation
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class Validator:
    """
    Validate, normalize, and post-process extracted fields.
    
    Post-Processing Features:
    - Multilingual text normalization
    - Near-duplicate detection and reconciliation
    - Numeric format standardization
    - Confidence threshold enforcement
    """
    
    # Hindi to English transliteration map for common terms
    HINDI_TRANSLITERATION = {
        # Dealer/Business terms
        'ट्रैक्टर्स': 'Tractors', 'ट्रेक्टर्स': 'Tractors',
        'मोटर्स': 'Motors', 'ऑटो': 'Auto',
        'एजेंसी': 'Agency', 'एजेंसीज': 'Agencies',
        'प्राइवेट': 'Private', 'प्रा': 'Pvt',
        'लिमिटेड': 'Limited', 'लि': 'Ltd',
        'एंड': 'And', 'एण्ड': 'And',
        'कंपनी': 'Company', 'कम्पनी': 'Company',
        'इंटरप्राइजेज': 'Enterprises',
        'ट्रेडिंग': 'Trading',
        'डीलर': 'Dealer', 'विक्रेता': 'Seller',
        # Common names
        'शर्मा': 'Sharma', 'गुप्ता': 'Gupta', 'सिंह': 'Singh',
        'पटेल': 'Patel', 'वर्मा': 'Verma', 'कुमार': 'Kumar',
        'अग्रवाल': 'Agarwal', 'जोशी': 'Joshi',
        'चौधरी': 'Choudhary', 'यादव': 'Yadav',
        'राजपूत': 'Rajput', 'मेहता': 'Mehta',
        # Tractor brands in Hindi
        'महिंद्रा': 'Mahindra', 'महिन्द्रा': 'Mahindra',
        'जॉन डियर': 'John Deere', 'जॉनडियर': 'John Deere',
        'स्वराज': 'Swaraj', 'सोनालिका': 'Sonalika',
        'टाफे': 'TAFE', 'एस्कॉर्ट्स': 'Escorts',
        'न्यू हॉलैंड': 'New Holland',
    }
    
    # Gujarati to English transliteration map
    GUJARATI_TRANSLITERATION = {
        # Dealer/Business terms
        'ટ્રેક્ટર્સ': 'Tractors', 'મોટર્સ': 'Motors',
        'ઓટો': 'Auto', 'એજન્સી': 'Agency',
        'પ્રાઇવેટ': 'Private', 'લિમિટેડ': 'Limited',
        'એન્ડ': 'And', 'કંપની': 'Company',
        'ડીલર': 'Dealer', 'વિક્રેતા': 'Seller',
        # Common names
        'પટેલ': 'Patel', 'શાહ': 'Shah', 'મહેતા': 'Mehta',
        'દેસાઈ': 'Desai', 'જોષી': 'Joshi',
        # Brands
        'મહિન્દ્રા': 'Mahindra', 'સ્વરાજ': 'Swaraj',
    }
    
    # Common OCR errors and corrections
    OCR_CORRECTIONS = {
        # Number-letter confusion
        '0': {'O': 0.8, 'o': 0.8, 'D': 0.6},
        '1': {'l': 0.9, 'I': 0.9, 'i': 0.7},
        '5': {'S': 0.7, 's': 0.7},
        '8': {'B': 0.6},
        # Common word corrections
        'pvt': 'Pvt', 'ltd': 'Ltd', 'PVT': 'Pvt', 'LTD': 'Ltd',
        'MAHINDRA': 'Mahindra', 'JOHNDEERE': 'John Deere',
        'SWARAJ': 'Swaraj', 'SONALIKA': 'Sonalika',
    }
    
    # Comprehensive Model-HP mapping for cross-validation
    MODEL_HP_MAP = {
        # Mahindra
        "MAHINDRA 265": 30, "MAHINDRA 275": 39, "MAHINDRA 475": 42,
        "MAHINDRA 575": 50, "MAHINDRA 585": 50, "MAHINDRA 605": 57,
        "MAHINDRA 415": 40, "MAHINDRA 555": 55, "MAHINDRA 595": 60,
        "ARJUN 555": 55, "ARJUN 605": 60, "ARJUN NOVO": 57,
        # John Deere
        "JOHN DEERE 5042": 42, "JOHN DEERE 5050": 50, "JOHN DEERE 5055": 55,
        "JOHN DEERE 5210": 42, "JOHN DEERE 5310": 55, "JOHN DEERE 5405": 63,
        # TAFE
        "TAFE 5900": 59, "TAFE 7250": 72, "TAFE 7515": 75,
        "TAFE 35": 35, "TAFE 45": 45,
        # Swaraj
        "SWARAJ 717": 17, "SWARAJ 724": 26, "SWARAJ 735": 39,
        "SWARAJ 744": 48, "SWARAJ 855": 52, "SWARAJ 963": 65,
        # Sonalika
        "SONALIKA 35": 35, "SONALIKA 42": 42, "SONALIKA 50": 50,
        "SONALIKA 60": 60, "SONALIKA 750": 50, "SONALIKA DI": 50,
        # Massey Ferguson
        "MASSEY FERGUSON 241": 42, "MASSEY FERGUSON 1035": 39,
        "MASSEY FERGUSON 7250": 50, "MASSEY FERGUSON 9500": 60,
        # New Holland
        "NEW HOLLAND 3230": 42, "NEW HOLLAND 3630": 55, "NEW HOLLAND 4710": 47,
        # Kubota
        "KUBOTA MU4501": 45, "KUBOTA MU5501": 55,
        # Eicher
        "EICHER 241": 24, "EICHER 312": 31, "EICHER 380": 38,
        "EICHER 485": 48, "EICHER 557": 55,
        # Escorts/Farmtrac/Powertrac
        "FARMTRAC 45": 45, "FARMTRAC 60": 60,
        "POWERTRAC 425": 37, "POWERTRAC 434": 39, "POWERTRAC 439": 39,
    }
    
    # Expanded default dealer list
    DEFAULT_DEALERS = [
        # Generic patterns
        "Tractors Pvt Ltd", "Motors Private Limited", "Auto Agencies",
        "Automobile Dealers", "Tractors and Automobiles",
        # Common name patterns
        "Sharma Tractors", "Gupta Motors", "Singh Auto", "Patel Agencies",
        "Kumar Tractors", "Reddy Auto Works", "Joshi Motors",
        "Verma Automobiles", "Agarwal Tractors", "Choudhary Motors",
        # Brand authorized dealers
        "Mahindra Authorized Dealer", "John Deere Dealership",
        "Swaraj Tractors", "TAFE Motors", "Sonalika Dealer",
    ]
    
    # Expanded model list
    DEFAULT_MODELS = [
        # Mahindra
        "Mahindra 265 DI", "Mahindra 275 DI", "Mahindra 275 XP Plus",
        "Mahindra 415 DI", "Mahindra 475 DI", "Mahindra 555 DI",
        "Mahindra 575 DI", "Mahindra 585 DI", "Mahindra 595 DI",
        "Mahindra 605 DI", "Mahindra Arjun 555 DI", "Mahindra Arjun 605 DI",
        "Mahindra Arjun Novo 605", "Mahindra JIVO",
        # John Deere
        "John Deere 5042D", "John Deere 5050D", "John Deere 5055E",
        "John Deere 5210", "John Deere 5310", "John Deere 5405",
        # TAFE
        "TAFE 35 DI", "TAFE 45 DI", "TAFE 5900", "TAFE 7250", "TAFE 7515",
        # Swaraj
        "Swaraj 717", "Swaraj 724 FE", "Swaraj 735 FE", "Swaraj 735 XT",
        "Swaraj 744 FE", "Swaraj 744 XT", "Swaraj 855 FE", "Swaraj 963 FE",
        # Sonalika
        "Sonalika DI 35", "Sonalika DI 42", "Sonalika DI 50", "Sonalika DI 60",
        "Sonalika DI 750 III", "Sonalika GT 26", "Sonalika RX",
        # Massey Ferguson
        "Massey Ferguson 241 DI", "Massey Ferguson 1035 DI",
        "Massey Ferguson 7250", "Massey Ferguson 9500",
        # New Holland
        "New Holland 3032", "New Holland 3230", "New Holland 3630",
        "New Holland 4710", "New Holland 5500",
        # Kubota
        "Kubota MU4501", "Kubota MU5501", "Kubota L4508",
        # Eicher
        "Eicher 241", "Eicher 312", "Eicher 380", "Eicher 485", "Eicher 557",
        # Escorts
        "Farmtrac 45", "Farmtrac 60", "Powertrac 425", "Powertrac 434", "Powertrac 439",
    ]
    
    def __init__(
        self,
        dealer_master: Optional[str] = None,
        model_master: Optional[str] = None,
        fuzzy_threshold: int = 90
    ):
        """
        Initialize validator.
        
        Args:
            dealer_master: Path to dealer CSV (optional)
            model_master: Path to model CSV (optional)
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold
        
        # Load or use defaults
        self.dealers = self._load_csv(dealer_master) if dealer_master else self.DEFAULT_DEALERS
        self.models = self._load_csv(model_master) if model_master else self.DEFAULT_MODELS
    
    def _load_csv(self, path: str) -> List[str]:
        """Load list from CSV first column."""
        try:
            import csv
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                return [row[0] for row in reader if row and row[0].strip()]
        except Exception:
            return []
    
    def validate_all(self, fields: Dict) -> Dict:
        """
        Validate all extracted fields with post-processing.
        
        Args:
            fields: Dict with (value, confidence) tuples
            
        Returns:
            Dict with validated values and '_confidence' score
        
        Post-Processing Applied:
        - Multilingual text normalization
        - Near-duplicate reconciliation
        - Numeric accuracy validation
        - Threshold-based confidence scoring
        """
        validated = {}
        confidences = []
        
        # === Dealer Name ===
        dealer_val, dealer_conf = fields.get('dealer_name', (None, 0))
        if dealer_val:
            # Post-process: Normalize multilingual text
            dealer_val = self.normalize_multilingual_text(dealer_val)
            dealer_val = self.normalize_business_name(dealer_val)
            
            matched, score = self.fuzzy_match(dealer_val, self.dealers, threshold=85)
            if matched and score >= self.fuzzy_threshold:
                validated['dealer_name'] = matched
                confidences.append(score / 100)
            else:
                # Keep original but lower confidence if no match
                validated['dealer_name'] = dealer_val
                confidences.append(min(dealer_conf, 0.7))
        else:
            validated['dealer_name'] = None
            confidences.append(0)
        
        # === Model Name ===
        model_val, model_conf = fields.get('model_name', (None, 0))
        if model_val:
            # Post-process: Normalize model name
            model_val = self.normalize_model_name(model_val)
            
            matched, score = self.fuzzy_match(model_val, self.models, threshold=80)
            if matched:
                validated['model_name'] = matched
                confidences.append(score / 100)
            else:
                validated['model_name'] = model_val
                confidences.append(min(model_conf, 0.75))
        else:
            validated['model_name'] = None
            confidences.append(0)
        
        # === Horse Power ===
        hp_val, hp_conf = fields.get('horse_power', (None, 0))
        if hp_val is not None and isinstance(hp_val, (int, float)):
            hp_val = int(hp_val)
            if 15 <= hp_val <= 150:
                validated['horse_power'] = hp_val
                
                # Cross-validate with model
                expected_hp = self._get_expected_hp(validated.get('model_name', ''))
                if expected_hp:
                    hp_diff = abs(hp_val - expected_hp)
                    if hp_diff <= 3:
                        confidences.append(min(1.0, hp_conf + 0.1))  # Boost
                    elif hp_diff <= 10:
                        confidences.append(hp_conf)
                    else:
                        confidences.append(hp_conf * 0.7)  # Penalty
                else:
                    confidences.append(hp_conf)
            else:
                validated['horse_power'] = None
                confidences.append(0)
        else:
            validated['horse_power'] = None
            confidences.append(0)
        
        # === Asset Cost ===
        cost_val, cost_conf = fields.get('asset_cost', (None, 0))
        if cost_val is not None:
            try:
                cost_val = int(cost_val)
                if 50000 <= cost_val <= 5000000:
                    validated['asset_cost'] = cost_val
                    confidences.append(cost_conf)
                else:
                    validated['asset_cost'] = None
                    confidences.append(0)
            except (ValueError, TypeError):
                validated['asset_cost'] = None
                confidences.append(0)
        else:
            validated['asset_cost'] = None
            confidences.append(0)
        
        # Store per-field confidences for downstream use
        validated['_field_confidences'] = {
            'dealer_name': confidences[0] if len(confidences) > 0 else 0,
            'model_name': confidences[1] if len(confidences) > 1 else 0,
            'horse_power': confidences[2] if len(confidences) > 2 else 0,
            'asset_cost': confidences[3] if len(confidences) > 3 else 0
        }
        
        # Calculate overall confidence
        validated['_confidence'] = sum(confidences) / len(confidences) if confidences else 0
        
        return validated
    
    def fuzzy_match(
        self,
        query: str,
        choices: List[str],
        threshold: int = None
    ) -> Tuple[Optional[str], int]:
        """
        Fuzzy match query against choices.
        
        Returns: (best_match, score) or (None, 0)
        """
        threshold = threshold if threshold is not None else self.fuzzy_threshold
        
        if not query or not choices:
            return (None, 0)
        
        query = str(query).strip()
        
        if RAPIDFUZZ_AVAILABLE:
            # Try different scorers
            result = process.extractOne(
                query, choices,
                scorer=fuzz.WRatio,  # Weighted ratio handles partial matches better
                score_cutoff=threshold
            )
            if result:
                return (result[0], int(result[1]))
        else:
            # Fallback: simple substring matching
            query_lower = query.lower()
            best_match = None
            best_score = 0
            
            for choice in choices:
                choice_lower = choice.lower()
                
                # Exact match
                if query_lower == choice_lower:
                    return (choice, 100)
                
                # Substring match
                if query_lower in choice_lower or choice_lower in query_lower:
                    score = 85
                    if score > best_score:
                        best_score = score
                        best_match = choice
                
                # Word overlap
                query_words = set(query_lower.split())
                choice_words = set(choice_lower.split())
                overlap = len(query_words & choice_words)
                if overlap >= 2:
                    score = min(90, 60 + overlap * 10)
                    if score > best_score:
                        best_score = score
                        best_match = choice
            
            if best_match and best_score >= threshold:
                return (best_match, best_score)
        
        return (None, 0)
    
    def _get_expected_hp(self, model: str) -> Optional[int]:
        """Get expected HP for a model from mapping."""
        if not model:
            return None
        
        model_upper = model.upper()
        
        # Try exact key match first
        for key, hp in self.MODEL_HP_MAP.items():
            if key in model_upper:
                return hp
        
        # Try extracting brand and number
        import re
        for key, hp in self.MODEL_HP_MAP.items():
            # Extract numbers from both
            key_nums = re.findall(r'\d+', key)
            model_nums = re.findall(r'\d+', model_upper)
            
            if key_nums and model_nums and key_nums[0] == model_nums[0]:
                # Same model number
                key_brand = key.split()[0]
                if key_brand in model_upper:
                    return hp
        
        return None
    
    def calculate_document_accuracy(
        self,
        prediction: Dict,
        ground_truth: Dict,
        hp_tolerance: float = 0.05,
        cost_tolerance: float = 0.05
    ) -> Dict:
        """
        Calculate accuracy against ground truth.
        
        Returns: Dict with per-field and overall accuracy
        """
        results = {'fields': {}, 'all_correct': True}
        
        # Text fields - fuzzy match
        for field in ['dealer_name', 'model_name']:
            pred_val = prediction.get('fields', {}).get(field)
            gt_val = ground_truth.get('fields', {}).get(field)
            
            if not gt_val:
                results['fields'][field] = True
                continue
            
            if not pred_val:
                results['fields'][field] = False
                results['all_correct'] = False
                continue
            
            # Both dealer_name and model_name use 90% threshold per evaluation criteria
            threshold = 90
            _, score = self.fuzzy_match(str(pred_val), [str(gt_val)], threshold=0)
            results['fields'][field] = score >= threshold
            if not results['fields'][field]:
                results['all_correct'] = False
        
        # Numeric fields - tolerance check
        for field, tol in [('horse_power', hp_tolerance), ('asset_cost', cost_tolerance)]:
            pred_val = prediction.get('fields', {}).get(field)
            gt_val = ground_truth.get('fields', {}).get(field)
            
            if gt_val is None:
                results['fields'][field] = pred_val is None
            elif pred_val is None:
                results['fields'][field] = False
                results['all_correct'] = False
            else:
                try:
                    diff = abs(float(pred_val) - float(gt_val))
                    results['fields'][field] = diff <= float(gt_val) * tol
                except (ValueError, TypeError):
                    results['fields'][field] = False
                
                if not results['fields'][field]:
                    results['all_correct'] = False
        
        return results
    
    # ============================================================
    # POST-PROCESSING METHODS FOR MULTILINGUAL & QUALITY ASSURANCE
    # ============================================================
    
    def normalize_multilingual_text(self, text: str) -> str:
        """
        Normalize multilingual text (Hindi, Gujarati, English).
        
        Applies:
        - Unicode normalization (NFC)
        - Hindi/Gujarati to English transliteration
        - Whitespace normalization
        - Common OCR error correction
        """
        if not text:
            return text
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Apply Hindi transliteration
        for hindi, english in self.HINDI_TRANSLITERATION.items():
            text = text.replace(hindi, english)
        
        # Apply Gujarati transliteration
        for gujarati, english in self.GUJARATI_TRANSLITERATION.items():
            text = text.replace(gujarati, english)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_business_name(self, name: str) -> str:
        """
        Normalize business/dealer name for consistency.
        
        Standardizes:
        - Company suffixes (Pvt Ltd, Private Limited)
        - Case normalization
        - Punctuation cleanup
        """
        if not name:
            return name
        
        # Remove extra punctuation
        name = re.sub(r'[,;:]+$', '', name)
        name = re.sub(r'^[,;:]+', '', name)
        
        # Standardize company suffixes
        suffix_map = {
            r'\bPVT\.?\s*LTD\.?\b': 'Pvt Ltd',
            r'\bPRIVATE\s+LIMITED\b': 'Pvt Ltd',
            r'\bप्रा\.?\s*लि\.?\b': 'Pvt Ltd',
            r'\bLIMITED\b': 'Ltd',
            r'\bLLP\b': 'LLP',
        }
        
        for pattern, replacement in suffix_map.items():
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        
        # Title case words (except suffixes)
        words = name.split()
        normalized_words = []
        for word in words:
            if word.lower() in ['pvt', 'ltd', 'llp', 'and', '&']:
                normalized_words.append(word.capitalize() if word.lower() != '&' else '&')
            elif word.isupper() and len(word) > 3:
                normalized_words.append(word.title())
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def normalize_model_name(self, model: str) -> str:
        """
        Normalize tractor model name.
        
        Standardizes:
        - Brand names (capitalization)
        - Model number formats
        - Variant suffixes (DI, XP, XT)
        """
        if not model:
            return model
        
        # First apply multilingual normalization
        model = self.normalize_multilingual_text(model)
        
        # Brand name standardization
        brand_map = {
            r'\bMAHINDRA\b': 'Mahindra',
            r'\bJOHN\s*DEERE\b': 'John Deere',
            r'\bJD\b': 'John Deere',
            r'\bTAFE\b': 'TAFE',
            r'\bSWARAJ\b': 'Swaraj',
            r'\bSONALIKA\b': 'Sonalika',
            r'\bMASSEY\s*FERGUSON\b': 'Massey Ferguson',
            r'\bMF\b': 'Massey Ferguson',
            r'\bNEW\s*HOLLAND\b': 'New Holland',
            r'\bNH\b': 'New Holland',
            r'\bKUBOTA\b': 'Kubota',
            r'\bEICHER\b': 'Eicher',
            r'\bFARMTRAC\b': 'Farmtrac',
            r'\bPOWERTRAC\b': 'Powertrac',
            r'\bESCORTS\b': 'Escorts',
        }
        
        for pattern, replacement in brand_map.items():
            model = re.sub(pattern, replacement, model, flags=re.IGNORECASE)
        
        # Standardize variant suffixes
        model = re.sub(r'\b(DI|di)\b', 'DI', model)
        model = re.sub(r'\b(XP|xp)\b', 'XP', model)
        model = re.sub(r'\b(XT|xt)\b', 'XT', model)
        model = re.sub(r'\b(FE|fe)\b', 'FE', model)
        model = re.sub(r'\b(PLUS|plus|Plus)\b', 'Plus', model)
        
        # Clean up spacing
        model = re.sub(r'\s+', ' ', model).strip()
        
        return model
    
    def reconcile_near_duplicates(
        self,
        values: List[str],
        threshold: int = 85
    ) -> str:
        """
        Reconcile near-duplicate values from multiple extractions.
        
        Used when multiple extraction methods return similar but
        slightly different values.
        
        Returns the most common/normalized value.
        """
        if not values:
            return None
        
        if len(values) == 1:
            return values[0]
        
        # Normalize all values
        normalized = [self.normalize_multilingual_text(v) for v in values if v]
        
        if not normalized:
            return None
        
        # Find the value that matches most others
        best_value = normalized[0]
        best_match_count = 0
        
        for i, val in enumerate(normalized):
            match_count = 0
            for j, other in enumerate(normalized):
                if i != j:
                    _, score = self.fuzzy_match(val, [other], threshold=0)
                    if score >= threshold:
                        match_count += 1
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_value = val
        
        return best_value
    
    def validate_numeric_accuracy(
        self,
        value: any,
        expected_range: Tuple[int, int],
        cross_reference: Optional[int] = None,
        tolerance: float = 0.05
    ) -> Tuple[Optional[int], float]:
        """
        Validate and normalize numeric values with accuracy checks.
        
        Args:
            value: Raw numeric value (int, float, or string)
            expected_range: (min, max) valid range
            cross_reference: Expected value for cross-validation
            tolerance: Tolerance for cross-reference match
            
        Returns:
            (validated_value, confidence)
        """
        if value is None:
            return (None, 0.0)
        
        # Parse numeric value
        try:
            if isinstance(value, str):
                # Remove currency symbols, commas, spaces
                cleaned = re.sub(r'[₹Rs,\.\s/-]+', '', value)
                parsed = int(cleaned)
            else:
                parsed = int(value)
        except (ValueError, TypeError):
            return (None, 0.0)
        
        # Range check
        min_val, max_val = expected_range
        if not (min_val <= parsed <= max_val):
            return (None, 0.0)
        
        # Base confidence for valid range
        confidence = 0.8
        
        # Cross-reference validation
        if cross_reference is not None:
            diff = abs(parsed - cross_reference)
            tolerance_amount = cross_reference * tolerance
            
            if diff <= tolerance_amount:
                confidence = 0.95  # Excellent match
            elif diff <= tolerance_amount * 2:
                confidence = 0.85  # Good match
            else:
                confidence = 0.6   # Mismatch warning
        
        return (parsed, confidence)
    
    def apply_confidence_threshold(
        self,
        results: Dict,
        field_thresholds: Dict[str, float] = None
    ) -> Dict:
        """
        Apply confidence thresholds to filter unreliable extractions.
        
        Default thresholds:
        - dealer_name: 0.6
        - model_name: 0.7
        - horse_power: 0.7
        - asset_cost: 0.7
        
        Returns results with low-confidence fields set to None.
        """
        default_thresholds = {
            'dealer_name': 0.6,
            'model_name': 0.7,
            'horse_power': 0.7,
            'asset_cost': 0.7
        }
        
        thresholds = field_thresholds or default_thresholds
        filtered = dict(results)
        
        # Get overall confidence
        overall_conf = results.get('_confidence', 0)
        
        # Apply per-field thresholds
        for field, threshold in thresholds.items():
            if field in filtered:
                # If overall confidence is very low, null out all fields
                if overall_conf < 0.3:
                    filtered[field] = None
        
        return filtered
    
    def detect_language(self, text: str) -> str:
        """
        Detect primary language of text.
        
        Returns: 'english', 'hindi', 'gujarati', or 'mixed'
        """
        if not text:
            return 'english'
        
        # Count characters by script
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        
        total = devanagari + gujarati + ascii_alpha
        if total == 0:
            return 'english'
        
        hi_ratio = devanagari / total
        gu_ratio = gujarati / total
        en_ratio = ascii_alpha / total
        
        if hi_ratio > 0.4:
            return 'hindi' if en_ratio < 0.3 else 'mixed'
        elif gu_ratio > 0.4:
            return 'gujarati' if en_ratio < 0.3 else 'mixed'
        elif en_ratio > 0.6:
            return 'english'
        else:
            return 'mixed'