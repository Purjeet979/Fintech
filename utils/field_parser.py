"""
Field Parser Module
Advanced rule-based field extraction with visual + textual understanding

Handles:
- English, Hindi, Gujarati text
- Multiple invoice formats and layouts
- Key-value pair detection with spatial awareness
- Table structure interpretation
- Contextual positioning analysis

Optimized for IDFC Bank's tractor loan quotation processing.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from loguru import logger


class FieldParser:
    """
    Extract structured fields using hybrid visual-textual analysis.
    
    Advanced Extraction Strategy:
    1. Spatial key-value pair detection
    2. Table structure recognition
    3. Region-based analysis (header/body/footer)
    4. Pattern matching with multilingual support
    5. Contextual keyword proximity
    """
    
    # ============ HORSE POWER PATTERNS ============
    # Enhanced patterns to handle:
    # - Hindi numerals (०-९) and HP indicators (एचपी, HP)
    # - Handwritten text with dots/dashes as separators
    # - HP embedded in model descriptions
    # - Parenthesized HP like "(HP- ३९)" or "(39 HP)"
    HP_PATTERNS = [
        # === PRIORITY 1: HP in parentheses (very common in OCR) ===
        r'\(?HP[\-:\s]*([\d०-९]{2,3})\)?',           # (HP- 39) or (HP: 39)
        r'\(?([\d०-९]{2,3})\s*HP\)?',                 # (39 HP) or 39 HP
        r'\(?[\-:\s]*([\d०-९]{2,3})\s*HP\)?',         # (- 39 HP)
        
        # === HIGH CONFIDENCE: Explicit HP labels ===
        r'([\d०-९]{2,3})\s*(?:HP|hp|H\.P\.|Hp)\b',
        r'([\d०-९]{2,3})\s*(?:BHP|bhp|B\.H\.P\.)\b',
        r'(?:Horse\s*Power|HP|Power)[:\s]*([\d०-९]{2,3})\b',
        r'(?:Engine|Motor)[:\s]*([\d०-९]{2,3})\s*(?:HP|hp)',
        
        # === Model-embedded HP patterns (common in tractor names) ===
        # "Swaraj 744 FE 48 HP", "Mahindra 575 DI 45 HP"
        r'(?:Swaraj|Mahindra|TAFE|Sonalika|John\s*Deere|Eicher|Escorts)[\s\-]*[\d०-९]{3,4}\s*\w*\s+([\d०-९]{2,3})\s*(?:HP|H\.?P\.?)',
        r'(?:FE|XT|DI|Plus)\s+([\d०-९]{2,3})\s*(?:HP|H\.?P\.?)',
        
        # === Handwritten patterns (dots/spaces as separators) ===
        r'([\d०-९]{2,3})\s*[\.]{2,}\s*(?:HP|hp|H\.P\.|Hp)',  # 25..... H.P.
        r'([\d०-९]{2,3})\s*[\.\-_]+\s*(?:HP|hp|H\.P\.)',     # 25... H.P.
        r'TRACTOR\s*[\.]+\s*([\d०-९]{2,3})\s*[\.]*\s*H\.?P\.?',  # TRACTOR ...25... H.P.
        
        # === Table patterns (HP in row with model) ===
        r'(?:PT|Powertrac|Farmtrac)[\s\-]*\w+[\s\-]*(?:HR|DS|Plus)\s+([\d०-९]{2,3})\s*(?:HP)?',
        
        # === Hindi patterns ===
        r'([\d०-९]{2,3})\s*(?:अश्वशक्ति|एचपी|हॉर्स\s*पावर)',
        r'(?:अश्वशक्ति|पावर|शक्ति)[:\s]*([\d०-९]{2,3})',
        r'एच\.?पी\.?[:\s]*([\d०-९]{2,3})',
        r'(?:HP|H\.P\.)[\-\s]*([\d०-९]{2,3})',  # HP-३९ format
        
        # === Gujarati patterns ===
        r'([\d०-९]{2,3})\s*(?:અશ્વશક્તિ|એચપી)',
        r'(?:અશ્વશક્તિ|પાવર)[:\s]*([\d०-९]{2,3})',
        
        # === Generic: number near HP keyword ===
        r'\bHP[:\s\-]*([\d०-९]{2,3})\b',
        r'\b([\d०-९]{2,3})[:\s\-]*HP\b',
        
        # === Specification table patterns ===
        r'(?:Rated\s*)?(?:Power|HP)[:\s]+([\d०-९]{2,3})',
        r'(?:Engine\s*)?(?:Power|Capacity)[:\s]+([\d०-९]{2,3})\s*(?:HP)?',
        
        # === Looser pattern (lower priority - only if nothing else matches) ===
        r'([\d०-९]{2})\s*.{0,10}H\.?P\.?',
    ]
    
    # ============ ASSET COST PATTERNS ============
    COST_PATTERNS = [
        # Labeled totals (most reliable)
        r'(?:Total|Grand\s*Total|Net\s*Amount|Final|Payable)[:\s]*[₹Rs\.INR\s]*([\d,\.]+)',
        r'(?:कुल|टोटल|राशि|मूल्य)[:\s]*[₹Rs\.INR\s]*([\d,\.]+)',
        r'(?:કુલ|કિંમત)[:\s]*[₹Rs\.INR\s]*([\d,\.]+)',
        # Currency symbol patterns
        r'[₹]\s*([\d,]+)(?:\s*/-)?',
        r'Rs\.?\s*([\d,]+)(?:\s*/-)?',
        r'INR\s*([\d,]+)',
        # Price/Amount labels
        r'(?:Price|Amount|Cost|Value)[:\s]*[₹Rs\.INR\s]*([\d,]+)',
        r'(?:Ex[\-\s]*Showroom|On[\-\s]*Road)[:\s]*[₹Rs\.INR\s]*([\d,]+)',
    ]
    
    # ============ PIN CODE / ADDRESS PATTERNS (to filter out) ============
    PIN_CODE_CONTEXT_PATTERNS = [
        # PIN code after location names (common Indian address format)
        r'(?:Dist|District|Pin|Pincode|PIN|Taluka|Tehsil|Block)[\.\s\-:]*\d{6}',
        r'\b[A-Z][a-z]+[\s\-]+\d{6}\b',  # City/Town name followed by 6 digits
        r'(?:Rajasthan|Gujarat|Maharashtra|MP|UP|Bihar|Punjab|Haryana|HP|Uttarakhand|Karnataka|Tamil Nadu|Kerala|AP|Telangana|Odisha|WB|Assam|Jharkhand|Chhattisgarh|Goa)\s*[\(\-]?\s*\d{6}',
        r'\(\s*Raj\.?\s*\)\s*[\-\s]*\d{6}',  # (Raj.) - 306115 pattern
        r'[A-Za-z]+\s*[\-\(]\s*\d{6}\s*[\)]?',  # RANI - 306115 or RANI (306115)
    ]
    
    # ============ EXCLUSION ZONE PATTERNS (Negative Pattern Matching) ============
    # These patterns identify regions that should NOT be used for numeric field extraction
    # Using negative pattern matching prevents PIN codes, phone numbers, etc. from being 
    # mistakenly extracted as prices or other numeric fields
    #
    # IMPORTANT: We do NOT use blanket numeric patterns like \b\d{6}\b here because:
    # - Valid tractor prices are often 6 digits (e.g., 525000, 350000, 850000)
    # - Context-based patterns below identify ADDRESS regions where PIN codes appear
    # - The _is_likely_pin_code() method handles contextual PIN vs price distinction
    #
    EXCLUSION_ZONE_PATTERNS = [
        # === Phone numbers (10-11 digits are safe to exclude - never valid prices) ===
        r'\b\d{10,11}\b',                # Phone numbers (10-11 digits)
        r'\b98\d{8}\b',                  # Mobile starting with 98
        r'\b99\d{8}\b',                  # Mobile starting with 99
        r'\b[6-9]\d{9}\b',               # Indian mobile numbers
        
        # === Address/Location indicators (context-based, not just numbers) ===
        r'(?:Dist|District|Taluka|Tehsil|Block|Village|Town|City)',
        r'(?:Pin|Pincode|PIN)[\s\-:]*\d{6}',  # PIN with label (e.g., "PIN: 306115")
        r'(?:RANI|Rani|rani)[\s\-]+\d{6}',    # Specific: "RANI - 306115" pattern
        r'[\-\s]\d{6}\s*[\(\,]',              # Number followed by bracket/comma in address
        r'(?:Contact|Contect|Phone|Mobile|Tel|Fax)[\s\-:]*[\d\+]',
        r'(?:GST|GSTIN|PAN|TIN|CIN)[\s\-:]*[A-Z0-9]',
        r'\S+@\S+\.\S+',                 # Email addresses
        
        # === Address line patterns ===
        r'(?:Road|Street|Lane|Nagar|Colony|Sector|Phase|Plot|Ward|Marg)',
        r'(?:Near|Opp|Behind|Adjacent|Next to|Beside)',
        r'Kenpura|Kendra|Market|Chowk',  # Common address words
        
        # === State identifiers with numbers ===
        r'\((?:Raj|Guj|MH|MP|UP|HR|PB|HP|UK|KA|TN|KL|AP|TS|OR|WB)\.?\)\s*[\-\s]*\d{6}',
        r'(?:Rajasthan|Gujarat|Maharashtra)\s*[\-\s]*\d{6}',
        r'Dist[\.\-\s]*Pali',            # Specific district pattern from test doc
        
        # === Header/Footer metadata ===
        r'(?:Invoice|Bill|Quotation)\s*(?:No|Number|#)',
        r'(?:Date|Dated?)[\s\-:]*\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4}',
    ]
    
    # ============ MODEL NAME PATTERNS ============
    # Enhanced patterns to handle:
    # - Hindi numerals (०-९)
    # - OCR misreads (SWARA→Swaraj, SWRAJ→Swaraj)
    # - Mixed Hindi-English text
    # - Combined brands (Escorts Kubota)
    # - Model in description phrases
    MODEL_PATTERNS = [
        # === PRIORITY 1: Full brand + series + model patterns ===
        # Mahindra Yuvo Tech (with Hindi numerals and HP in parens)
        r'(MAHINDRA\s+Y[uU][vV][oO]\s*(?:TECH|Tech)?\s*\+?\s*[\d०-९]{3,4}\s*(?:DI|Dl|XP|XT)?)',
        r'(Mahindra\s+Yuvo\s*(?:Tech)?\s*\+?\s*[\d०-९]{3,4}\s*(?:DI|XP|XT)?)',
        
        # SWARA/SWRAJ → Swaraj (most common OCR error in dataset)
        r'(?:Cost\s+of|Price\s+of)?\s*(SWARA[JI]?\s*[\d०-९]{3,4}\s*(?:FE|XT|DI)?)',
        r'(SW[AR]{2,3}J?\s*[\d०-९]{3,4}\s*(?:FE|XT|DI)?)',
        
        # === PRIORITY 2: Standard brand + model patterns ===
        # Swaraj
        r'(Swaraj\s+[\d०-९]{3,4}\s*(?:FE|XT|DI)?(?:\s*TRACTOR)?)',
        r'(SWARAJ\s+[\d०-९]{3,4}\s*(?:FE|XT|DI)?)',
        
        # Mahindra variants (other series)
        r'(Mahindra\s+[\d०-९]{3,4}\s*(?:DI|XP|XT|Plus|Power)?(?:\s*Plus)?)',
        r'(MAHINDRA\s+[\d०-९]{3,4}\s*(?:DI|XP|XT|Plus|Power)?)',
        r'(Arjun\s+(?:Novo\s+)?[\d०-९]{3,4})',
        r'(Yuvo\s*(?:Tech)?\s*\+?\s*[\d०-९]{3,4}\s*(?:DI|Dl)?)',  # Mahindra Yuvo series
        
        # John Deere
        r'(John\s*Deere\s+[\d०-९]{4}[A-Z]?)',
        r'(JD[\s\-]*[\d०-९]{4}[A-Z]?)',
        r'(?:Model\s+[Nn]ame[:\s]*)?(John\s*Deere\s+[\d०-९]{4}\s*(?:Gear\s*Pro)?)',
        
        # Escorts Kubota (combined brand)
        r'(Escorts\s*Kubota\s+[\d०-९]{2,4})',
        r'(Escorts\s*Kubota\s+\w+)',
        
        # Sonalika / International Tractors Ltd
        r'(Sonalika\s+(?:DI\s*)?[\d०-९]+(?:\s*[A-Z]+)?)',
        r'(Sonalika\s+[A-Z]+\s*[\d०-९]+)',
        r'(Sonalika\s+GT\s*[\d०-९]+)',
        
        # TAFE / Massey Ferguson
        r'(TAFE\s+[\d०-९]{4})',
        r'(Massey\s*Ferguson\s+[\d०-९]{3,4})',
        r'(MF[\s\-]*[\d०-९]{3,4})',
        
        # Solis (Yanmar subsidiary) - handle OCR errors
        r'(S[o0][l1][iIu][s5]\s+[\d०-९]{3,4}\s*(?:2WD|4WD|DI|XT)?)',
        r'(SOLIS\s+[\d०-९]{3,4}\s*(?:2WD|4WD|DI|XT)?)',
        
        # New Holland
        r'(New\s*Holland\s+[\d०-९]{4})',
        r'(NH[\s\-]*[\d०-९]{4})',
        
        # Kubota
        r'(Kubota\s+\w+[\s\-]?[\d०-९]+)',
        
        # Eicher
        r'(Eicher\s+[\d०-९]{3,4})',
        
        # Escorts/Farmtrac/Powertrac
        r'(Farmtrac\s+[\d०-९]+)',
        r'(Powertrac\s+[\d०-९]+)',
        r'(Faemtrac\s+[\d०-९]+)',  # Common OCR misread
        r'(Escorts\s+[\d०-९]+)',
        
        # VST / Zetor
        r'(VST\s+(?:Zetor\s+)?\w*\s*[\d०-९]*)',
        r'(Zetor\s+[\d०-९]+)',
        
        # Indo Farm
        r'(Indo\s*Farm\s+[\d०-९]+)',
        
        # Captain
        r'(Captain\s+[\d०-९]{3,4})',
        
        # Force Motors
        r'(Force\s+(?:Orchard|Sanman|Abhiman)\s*[\d०-९]*)',
        
        # Preet
        r'(Preet\s+[\d०-९]{3,4})',
        
        # HMT
        r'(HMT\s+[\d०-९]{3,4})',
        
        # ACE / Standard
        r'(ACE\s+[\d०-९]+)',
        r'(Standard\s+[\d०-९]+)',
        
        # === PRIORITY 3: Context-based patterns ===
        # "Cost of X Tractor" - common quotation format
        r'(?:Cost\s+of|Price\s+of)\s+([A-Za-z]+[\s\-]*[\d०-९]{3,4}[\s\-]*(?:FE|XT|DI|HP)?)\s*(?:Tractor)?',
        # Model line with HP embedded
        r'([A-Za-z]+\s+[\d०-९]{3,4}\s*(?:FE|XT|DI)?)\s*(?:Tractor)?\s*[\.\-]+\s*[\d०-९]{2,3}\s*[\.\-]*\s*H\.?P\.?',
        # Generic "Model: X" pattern
        r'(?:Model|Tractor)[:\s]+([A-Za-z]+[\s\-]*[\d०-९]{3,4}[A-Za-z]*)',
    ]
    
    # ============ DEALER PATTERNS ============
    DEALER_PATTERNS = [
        # Explicit labels - Note: "From" removed as too generic (matches "from farmer" etc.)
        r'(?:Dealer|Seller|Sold\s*by|Authorized\s*Dealer)[:\s]+([^\n\r]+)',
        r'(?:विक्रेता|डीलर|एजेंसी)[:\s]+([^\n\r]+)',
        r'(?:વિક્રેતા|ડીલર)[:\s]+([^\n\r]+)',
    ]
    
    # Business suffixes for dealer detection
    BUSINESS_SUFFIXES = [
        'pvt', 'ltd', 'private', 'limited', 'llp',
        'tractor', 'tractors', 'motors', 'auto', 'automobiles', 'agencies',
        'enterprises', 'trading', 'corporation', 'associates', 'dealer',
        'ट्रैक्टर', 'ट्रैक्टर्स', 'मोटर्स', 'एजेंसी',  # Hindi
        'ટ્રેક્ટર', 'ટ્રેક્ટર્સ', 'મોટર્સ',  # Gujarati
    ]
    
    # Key-value keywords for spatial detection
    KEY_PATTERNS = {
        'horse_power': [
            'hp', 'h.p.', 'horse power', 'horsepower', 'bhp', 'power',
            'अश्वशक्ति', 'एचपी', 'हॉर्स पावर', 'પાવર', 'અશ્વશક્તિ'
        ],
        'asset_cost': [
            'total', 'grand total', 'amount', 'price', 'cost', 'value',
            'net amount', 'payable', 'ex-showroom', 'on-road', 'final',
            'कुल', 'राशि', 'मूल्य', 'कीमत', 'કુલ', 'કિંમત', 'રકમ'
        ],
        'model_name': [
            'model', 'tractor', 'vehicle', 'product', 'item', 'description',
            'मॉडल', 'ट्रैक्टर', 'વાહન', 'મોડલ', 'ટ્રેક્ટર'
        ],
        'dealer_name': [
            'dealer', 'seller', 'sold by', 'authorized dealer', 'agent',
            'डीलर', 'विक्रेता', 'एजेंट', 'ડીલર', 'વિક્રેતા'
        ]
    }
    
    def __init__(self):
        # Compile patterns for efficiency
        self._hp_compiled = [re.compile(p, re.IGNORECASE) for p in self.HP_PATTERNS]
        self._cost_compiled = [re.compile(p, re.IGNORECASE) for p in self.COST_PATTERNS]
        self._model_compiled = [re.compile(p, re.IGNORECASE) for p in self.MODEL_PATTERNS]
        self._pin_context_compiled = [re.compile(p, re.IGNORECASE) for p in self.PIN_CODE_CONTEXT_PATTERNS]
        self._exclusion_compiled = [re.compile(p, re.IGNORECASE) for p in self.EXCLUSION_ZONE_PATTERNS]
        
        # Cache for exclusion zones (reset per document)
        self._exclusion_zones = []
        
        # ============ ADAPTIVE SPATIAL ANALYSIS ============
        # These values are computed per-document from OCR boxes
        # Using median_text_height as the universal unit scales with resolution/font size
        self._median_text_height = 20  # Default fallback (will be computed per doc)
        self._median_text_width = 100  # Default fallback
    
    def _compute_adaptive_metrics(self, boxes: List[List[int]]) -> None:
        """
        Compute adaptive spatial metrics from OCR boxes.
        
        Uses median text height as the universal unit of measurement.
        This naturally scales with document resolution and font size.
        """
        if not boxes:
            return
        
        heights = []
        widths = []
        
        for box in boxes:
            if len(box) >= 4:
                h = abs(box[3] - box[1])
                w = abs(box[2] - box[0])
                if h > 0:
                    heights.append(h)
                if w > 0:
                    widths.append(w)
        
        if heights:
            # Use median to be robust against outliers (headers, footers)
            heights.sort()
            self._median_text_height = heights[len(heights) // 2]
        
        if widths:
            widths.sort()
            self._median_text_width = widths[len(widths) // 2]
        
        logger.debug(f"Adaptive metrics: median_text_height={self._median_text_height}px, median_text_width={self._median_text_width}px")
    
    # ============ NEGATIVE PATTERN MATCHING (EXCLUSION ZONES) ============
    
    def _identify_exclusion_zones(
        self,
        texts: List[str],
        boxes: List[List[int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Identify regions that should NOT be used for price/HP extraction.
        
        This is a preemptive filtering step that marks address, contact,
        and identification number regions as exclusion zones BEFORE
        attempting numeric field extraction.
        
        **Negative Pattern Matching Strategy:**
        - Detects address indicators (District, PIN, etc.)
        - Identifies phone/mobile numbers
        - Marks email/GST/PAN regions
        - Expands zones to catch nearby false positives
        
        Returns:
            List of (x1, y1, x2, y2) bounding boxes for exclusion zones
        """
        exclusion_zones = []
        excluded_texts = []  # For logging
        
        for text, box in zip(texts, boxes):
            is_exclusion = False
            matched_pattern = None
            
            # Check against exclusion patterns
            for pattern in self._exclusion_compiled:
                match = pattern.search(text)
                if match:
                    is_exclusion = True
                    matched_pattern = match.group()
                    break
            
            if is_exclusion:
                excluded_texts.append((text[:40], matched_pattern))
                
                # Expand zone slightly to catch nearby numbers
                x1, y1, x2, y2 = box
                padding_x = (x2 - x1) * 0.3  # 30% horizontal padding
                padding_y = (y2 - y1) * 0.5  # 50% vertical padding
                
                expanded_zone = (
                    int(x1 - padding_x),
                    int(y1 - padding_y),
                    int(x2 + padding_x),
                    int(y2 + padding_y)
                )
                exclusion_zones.append(expanded_zone)
        
        # Log excluded regions for transparency
        if exclusion_zones:
            logger.debug(f"[Negative Pattern Matching] Identified {len(exclusion_zones)} exclusion zones")
            for text_preview, pattern in excluded_texts[:5]:  # Show first 5
                logger.debug(f"  - Excluded: '{text_preview}...' (matched: '{pattern}')")
        
        return exclusion_zones
    
    def _calculate_iou(self, box1: List[int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2] first bounding box
            box2: (x1, y1, x2, y2) second bounding box
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_in_exclusion_zone(
        self,
        box: List[int],
        exclusion_zones: Optional[List[Tuple[int, int, int, int]]] = None,
        iou_threshold: float = 0.1
    ) -> bool:
        """
        Check if a bounding box overlaps with any exclusion zone.
        
        Uses IoU (Intersection over Union) for robust overlap detection
        instead of center-only check. Any significant overlap triggers exclusion.
        
        Args:
            box: [x1, y1, x2, y2] bounding box to check
            exclusion_zones: List of exclusion zones (uses cached if None)
            iou_threshold: Minimum IoU to consider as overlap (default 0.1 = 10%)
            
        Returns:
            True if box has significant overlap with any exclusion zone
        """
        zones = exclusion_zones if exclusion_zones is not None else self._exclusion_zones
        
        if not zones or not box:
            return False
        
        # Check IoU against each exclusion zone
        for zone in zones:
            iou = self._calculate_iou(box, zone)
            if iou >= iou_threshold:
                return True
        
        return False
    
    def _filter_by_exclusion_zones(
        self,
        texts: List[str],
        boxes: List[List[int]],
        exclusion_zones: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Filter out text elements that fall within exclusion zones.
        
        Returns:
            Filtered (texts, boxes) tuple with exclusion zone elements removed
        """
        zones = exclusion_zones if exclusion_zones is not None else self._exclusion_zones
        
        if not zones:
            return texts, boxes
        
        filtered_texts = []
        filtered_boxes = []
        
        for text, box in zip(texts, boxes):
            if not self._is_in_exclusion_zone(box, zones):
                filtered_texts.append(text)
                filtered_boxes.append(box)
        
        return filtered_texts, filtered_boxes
    
    def extract_all(self, ocr_result: Dict) -> Dict:
        """
        Extract all fields using hybrid visual-textual analysis.
        
        Enhanced extraction with:
        1. **Negative Pattern Matching** - Identify exclusion zones first
        2. Spatial key-value pair detection
        3. Table structure analysis
        4. Region-based contextual extraction
        5. Pattern matching fallback
        
        Args:
            ocr_result: Dict with 'texts', 'boxes', 'full_text' from OCR
            
        Returns:
            Dict with (value, confidence) tuples for each field
        """
        full_text = ocr_result.get('full_text', '')
        texts = ocr_result.get('texts', [])
        boxes = ocr_result.get('boxes', [])
        
        # === ADAPTIVE SPATIAL METRICS ===
        # Compute document-specific metrics for resolution-independent analysis
        self._compute_adaptive_metrics(boxes)
        
        # === STEP 0: Identify Exclusion Zones (Negative Pattern Matching) ===
        # This preemptively marks address/contact regions to avoid extracting
        # PIN codes, phone numbers, etc. as prices or other numeric fields
        logger.debug("Starting Negative Pattern Matching to identify exclusion zones...")
        self._exclusion_zones = self._identify_exclusion_zones(texts, boxes)
        
        # Create filtered text for numeric extraction (HP, Cost)
        # Dealer/Model extraction uses full text since they're in different regions
        filtered_texts, filtered_boxes = self._filter_by_exclusion_zones(texts, boxes)
        filtered_full_text = ' '.join(filtered_texts)
        
        filter_ratio = len(filtered_texts) / max(len(texts), 1)
        logger.debug(f"  → Text elements: {len(texts)} total, {len(filtered_texts)} after filtering ({filter_ratio:.0%})")
        
        # Build spatial index for key-value detection
        text_elements = list(zip(texts, boxes)) if texts and boxes else []
        filtered_elements = list(zip(filtered_texts, filtered_boxes)) if filtered_texts else []
        
        # Detect key-value pairs spatially (use filtered for numeric fields)
        kv_pairs = self._detect_key_value_pairs(text_elements)
        filtered_kv_pairs = self._detect_key_value_pairs(filtered_elements)
        
        # Detect table structures
        table_data = self._detect_table_structure(text_elements)
        filtered_table_data = self._detect_table_structure(filtered_elements)
        
        # Extract using multiple strategies
        results = {}
        
        # === DEALER NAME ===
        results['dealer_name'] = self._extract_with_fallback(
            'dealer_name',
            [
                lambda: self._extract_from_kv(kv_pairs, 'dealer_name'),
                lambda: self.extract_dealer(full_text, texts, boxes)
            ]
        )
        
        # === MODEL NAME ===
        results['model_name'] = self._extract_with_fallback(
            'model_name',
            [
                lambda: self._extract_model_from_table(table_data),
                lambda: self._extract_from_kv(kv_pairs, 'model_name'),
                lambda: self.extract_model(full_text, texts)
            ]
        )
        
        # === HORSE POWER (use filtered data to avoid address regions) ===
        results['horse_power'] = self._extract_with_fallback(
            'horse_power',
            [
                lambda: self._extract_hp_from_table(filtered_table_data),
                lambda: self._extract_from_kv(filtered_kv_pairs, 'horse_power'),
                lambda: self.extract_hp(filtered_full_text)
            ]
        )
        
        # === ASSET COST (use filtered data to avoid PIN codes) ===
        # NOTE: Pass full_text as context_text so _is_likely_pin_code can detect
        # address keywords even though extraction uses filtered text
        results['asset_cost'] = self._extract_with_fallback(
            'asset_cost',
            [
                lambda: self._extract_cost_from_table(filtered_table_data),
                lambda: self._extract_from_kv(filtered_kv_pairs, 'asset_cost'),
                lambda: self.extract_cost(filtered_full_text, context_text=full_text)
            ]
        )
        
        return results
    
    def _extract_with_fallback(
        self,
        field: str,
        strategies: List
    ) -> Tuple[Any, float]:
        """Try multiple extraction strategies, return best result."""
        best_result = (None, 0.0)
        
        for strategy in strategies:
            try:
                result = strategy()
                if result and result[0] is not None and result[1] > best_result[1]:
                    best_result = result
                    if result[1] >= 0.9:  # High confidence, stop searching
                        break
            except Exception:
                continue
        
        return best_result
    
    def _detect_key_value_pairs(
        self,
        text_elements: List[Tuple[str, List[int]]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Detect key-value pairs using spatial proximity analysis.
        
        Strategy:
        - Find text elements containing key words
        - Look for values to the right or below the key
        - Use spatial distance to determine association
        """
        kv_pairs = defaultdict(list)
        
        if not text_elements:
            return kv_pairs
        
        for i, (text, box) in enumerate(text_elements):
            text_lower = text.lower().strip()
            
            # Check if this text contains a key
            for field, keywords in self.KEY_PATTERNS.items():
                if any(kw in text_lower for kw in keywords):
                    # Found a key, look for value
                    value, conf = self._find_value_near_key(
                        text, box, text_elements, field
                    )
                    if value:
                        kv_pairs[field].append((value, conf))
        
        return kv_pairs
    
    def _find_value_near_key(
        self,
        key_text: str,
        key_box: List[int],
        text_elements: List[Tuple[str, List[int]]],
        field: str
    ) -> Tuple[Optional[str], float]:
        """
        Find value associated with a key based on spatial position.
        
        Uses ADAPTIVE thresholds based on median_text_height for resolution independence.
        
        Looks for values:
        1. On the same line, to the right of the key
        2. On the next line, below the key
        3. After colon/separator in the same text
        """
        # Check if value is in the same text (after colon)
        if ':' in key_text:
            parts = key_text.split(':', 1)
            if len(parts) > 1 and parts[1].strip():
                return (parts[1].strip(), 0.9)
        
        key_x1, key_y1, key_x2, key_y2 = key_box
        key_cx = (key_x1 + key_x2) / 2
        key_cy = (key_y1 + key_y2) / 2
        key_height = key_y2 - key_y1
        key_width = key_x2 - key_x1
        
        # === ADAPTIVE THRESHOLDS ===
        # Use median_text_height as the universal unit of measurement
        # This scales naturally with document resolution and font size
        mth = self._median_text_height
        
        # Max horizontal distance: ~8x median text height (adapts to resolution)
        max_horizontal_distance = mth * 8
        
        # Alignment tolerance for "below" detection: ~1.5x key width
        alignment_tolerance = max(key_width * 1.5, mth * 3)
        
        candidates = []
        
        for text, box in text_elements:
            if text.lower().strip() == key_text.lower().strip():
                continue
            
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Check if on same line (y overlap) and to the right
            # Use key_height for same-line detection (more precise than fixed threshold)
            if abs(cy - key_cy) < key_height * 0.8 and x1 > key_x2 - 10:
                distance = x1 - key_x2
                if distance < max_horizontal_distance:
                    # Score: closer = higher confidence
                    proximity_score = 1.0 - (distance / max_horizontal_distance)
                    conf = 0.75 + (proximity_score * 0.15)  # 0.75 to 0.90
                    candidates.append((text, conf, distance))
            
            # Check if on next line and aligned
            elif y1 > key_y2 and y1 - key_y2 < key_height * 2.5:
                if abs(cx - key_cx) < alignment_tolerance:
                    vertical_distance = y1 - key_y2
                    candidates.append((text, 0.70, vertical_distance))
        
        # Return closest candidate
        if candidates:
            candidates.sort(key=lambda x: x[2])  # Sort by distance
            best = candidates[0]
            return (best[0], best[1])
        
        return (None, 0.0)
    
    def _extract_from_kv(
        self,
        kv_pairs: Dict,
        field: str
    ) -> Tuple[Any, float]:
        """Extract field from detected key-value pairs."""
        values = kv_pairs.get(field, [])
        if not values:
            return (None, 0.0)
        
        # Get highest confidence value
        best = max(values, key=lambda x: x[1])
        raw_value, conf = best
        
        # Post-process based on field type
        if field == 'horse_power':
            hp = self._extract_hp_from_text(raw_value)
            if hp:
                return (hp, conf)
        elif field == 'asset_cost':
            cost = self._parse_indian_number(raw_value)
            if cost and 50000 <= cost <= 5000000:
                return (cost, conf)
        elif field == 'dealer_name':
            # Validate dealer name - must be reasonable business name
            cleaned = raw_value.strip()
            # Reject single words or very short names (likely noise)
            if len(cleaned) < 5 or ' ' not in cleaned:
                return (None, 0.0)
            # Reject common noise words/generic phrases
            noise_phrases = {
                'sales service', 'sales & service', 'sales services',
                'spare parts', 'spares parts', 'service & spare',
                'authorized dealer', 'authorized dealer for'
            }
            if cleaned.lower().strip() in noise_phrases:
                return (None, 0.0)
            # Reject if starts with generic label words
            if re.match(r'^(?:for|to|from|authorized)\s+', cleaned, re.IGNORECASE):
                # Check if meaningful business name follows
                stripped = re.sub(r'^(?:for|to|from|authorized\s+dealer\s+for?)\s*', '', cleaned, flags=re.IGNORECASE)
                if len(stripped.strip()) < 5:
                    return (None, 0.0)
                cleaned = stripped.strip()  # Use the stripped version
            # Reject patterns that are clearly service descriptions, not names
            service_patterns = [
                r'^Sales\s+Service',
                r'^Service\s+&?\s*Spare',
                r'^Spare\s+Parts',
                r'^(?:Near|Opp|Behind|Adjacent)\s+',  # Address context
            ]
            for pattern in service_patterns:
                if re.match(pattern, cleaned, re.IGNORECASE):
                    return (None, 0.0)
            return (cleaned, conf)
        elif field == 'model_name':
            cleaned = raw_value.strip()
            # Reject single characters or very short names
            if len(cleaned) < 3:
                return (None, 0.0)
            # Reject values that look like prices (Rs, ₹, numbers with commas)
            if re.match(r'^[Rr][sS][\.\-\s]*[\d,]+', cleaned):
                return (None, 0.0)
            if re.match(r'^₹[\d,]+', cleaned):
                return (None, 0.0)
            if re.match(r'^[\d,]+\s*[/\-]', cleaned):  # "9,11,769.00 /-" style
                return (None, 0.0)
            # Reject if it's purely numeric (with commas)
            if re.match(r'^[\d,\.]+$', cleaned.replace(' ', '')):
                return (None, 0.0)
            # Reject phone numbers and contact info
            if re.search(r'\b\d{10}\b', cleaned):  # 10-digit phone
                return (None, 0.0)
            if re.match(r'^(?:Cell|Phone|Mobile|Tel|Contact)[:\s]', cleaned, re.IGNORECASE):
                return (None, 0.0)
            # Reject patterns that are clearly not model names
            noise_patterns = [
                r'^Customer\s*[$#@]',  # "Customer $" type
                r'^(?:Mr|Mrs|Ms|Shri|Smt)[\s\.]',  # Person names
                r'^(?:Date|Dated?)[:\s]',  # Date fields
                r'^(?:Invoice|Bill|Quotation)[\s#]',  # Doc references
                r'@.*\.com',  # Email
                r'^\d+[\-/]\d+',  # Date or ref numbers
                r'^[A-Z]{2,5}\d{10,}',  # GST/PAN type
                r'^(?:Near|Opp|Behind|Adjacent)\s+',  # Address context
                r'Bus\s*Stand',  # Address landmark
                r'Pin[\s\-:]*\d{5,6}',  # PIN code in text
                r'[\-,]\s*\d{6}\b',  # PIN code pattern
            ]
            for pattern in noise_patterns:
                if re.match(pattern, cleaned, re.IGNORECASE):
                    return (None, 0.0)
            # Reject generic category words (not specific model names)
            generic_words = {
                'tractor', 'tractors', 'spare', 'spares', 'parts', 'accessories',
                'service', 'sales', 'dealer', 'agency', 'centre', 'center',
                'equipment', 'machinery', 'agricultural', 'farm', 'vehicle',
                'customer', 'client', 'buyer', 'purchaser', 'neftrta'
            }
            if cleaned.lower() in generic_words:
                return (None, 0.0)
            # Reject if mostly corrupted Unicode (broken Hindi/Gujarati encoding)
            # Check for high ratio of replacement characters or unusual Unicode blocks
            alpha_ratio = sum(1 for c in cleaned if c.isalpha() or c.isspace() or c in '.-&') / max(len(cleaned), 1)
            if alpha_ratio < 0.5:
                return (None, 0.0)
            return (cleaned, conf)
        else:
            return (raw_value.strip(), conf)
        
        return (None, 0.0)
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text is valid (not corrupted Unicode)."""
        if not text:
            return False
        # Check for common encoding corruption patterns
        # These appear when UTF-8 is misread as latin-1 or similar
        corruption_patterns = [
            r'[\u0080-\u009f]',  # C1 control characters (rarely valid)
            r'α[ñÑ][^\s]{2,}',  # Common corruption pattern
        ]
        for pattern in corruption_patterns:
            if re.search(pattern, text):
                return False
        return True
    
    def _extract_hp_from_text(self, text: str) -> Optional[int]:
        """Extract HP number from text."""
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            hp = int(num)
            if 15 <= hp <= 150:
                return hp
        return None
    
    def _detect_table_structure(
        self,
        text_elements: List[Tuple[str, List[int]]]
    ) -> Dict:
        """
        Detect table structures in document.
        
        Strategy:
        - Group text elements by rows (similar y-coordinates)
        - Identify column alignment
        - Extract header-value relationships
        """
        if not text_elements or len(text_elements) < 3:
            return {'rows': [], 'columns': [], 'cells': {}}
        
        # Group by rows
        rows = self._group_by_rows(text_elements)
        
        # Identify potential table regions
        table_data = {
            'rows': rows,
            'header_row': None,
            'data_rows': [],
            'cells': {}
        }
        
        # Find header row (often contains keywords)
        table_keywords = ['model', 'hp', 'power', 'price', 'amount', 'description',
                         'मॉडल', 'मूल्य', 'मॉडেલ', 'કિંમત']
        
        for i, row in enumerate(rows):
            row_text = ' '.join([t for t, _ in row]).lower()
            if any(kw in row_text for kw in table_keywords):
                table_data['header_row'] = i
                table_data['data_rows'] = rows[i+1:min(i+6, len(rows))]
                break
        
        return table_data
    
    def _group_by_rows(
        self,
        text_elements: List[Tuple[str, List[int]]],
        y_tolerance: int = 15
    ) -> List[List[Tuple[str, List[int]]]]:
        """Group text elements into rows based on y-coordinate."""
        if not text_elements:
            return []
        
        # Sort by y then x
        sorted_elements = sorted(text_elements, key=lambda x: (x[1][1], x[1][0]))
        
        rows = []
        current_row = [sorted_elements[0]]
        current_y = sorted_elements[0][1][1]
        
        for text, box in sorted_elements[1:]:
            y = box[1]
            if abs(y - current_y) <= y_tolerance:
                current_row.append((text, box))
            else:
                rows.append(sorted(current_row, key=lambda x: x[1][0]))
                current_row = [(text, box)]
                current_y = y
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x[1][0]))
        
        return rows
    
    def _extract_model_from_table(self, table_data: Dict) -> Tuple[Optional[str], float]:
        """Extract model name from table structure."""
        for row in table_data.get('data_rows', []):
            row_text = ' '.join([t for t, _ in row])
            
            # Try model patterns on row text
            for pattern in self._model_compiled:
                matches = pattern.findall(row_text)
                if matches:
                    model = re.sub(r'\s+', ' ', matches[0]).strip()
                    return (self._normalize_model(model), 0.92)
        
        return (None, 0.0)
    
    def _extract_hp_from_table(self, table_data: Dict) -> Tuple[Optional[int], float]:
        """Extract HP from table structure."""
        for row in table_data.get('data_rows', []):
            row_text = ' '.join([t for t, _ in row])
            
            # Look for HP patterns
            hp_match = re.search(r'(\d{2,3})\s*(?:HP|hp|H\.P\.)', row_text)
            if hp_match:
                hp = int(hp_match.group(1))
                if 15 <= hp <= 150:
                    return (hp, 0.92)
        
        return (None, 0.0)
    
    def _extract_cost_from_table(self, table_data: Dict) -> Tuple[Optional[int], float]:
        """Extract cost from table, looking for totals in last rows."""
        # Check data rows in reverse (totals usually at bottom)
        for row in reversed(table_data.get('data_rows', [])):
            row_text = ' '.join([t for t, _ in row])
            
            # Look for total patterns
            if any(kw in row_text.lower() for kw in ['total', 'grand', 'कुल', 'કુલ']):
                numbers = re.findall(r'[\d,\.]+', row_text)
                for num in reversed(numbers):  # Usually last number
                    cost = self._parse_indian_number(num)
                    if cost and 50000 <= cost <= 5000000:
                        return (cost, 0.92)
        
        return (None, 0.0)
    
    def extract_hp(self, text: str) -> Tuple[Optional[int], float]:
        """
        Extract horse power value.
        
        Returns: (value, confidence) or (None, 0.0)
        """
        # Normalize Hindi numerals first
        text = self._normalize_hindi_numerals(text)
        
        # Try each pattern
        for pattern in self._hp_compiled:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    # Also normalize the match itself
                    match_normalized = self._normalize_hindi_numerals(str(match))
                    hp = int(match_normalized)
                    if 15 <= hp <= 150:  # Valid HP range for tractors
                        return (hp, 0.9)
                except (ValueError, TypeError):
                    continue
        
        # Fallback: find numbers near HP keywords
        hp_context = re.search(
            r'([\d०-९]{2,3})\s*.{0,15}(?:HP|hp|H\.P\.|अश्वशक्ति|અશ્વશક્તિ|एचपी)|(?:HP|hp|H\.P\.|अश्वशक्ति|અશ્વશક્તિ|एचपी).{0,15}([\d०-९]{2,3})',
            text, re.IGNORECASE
        )
        if hp_context:
            try:
                hp_str = hp_context.group(1) or hp_context.group(2)
                hp_normalized = self._normalize_hindi_numerals(hp_str)
                hp = int(hp_normalized)
                if 15 <= hp <= 150:
                    return (hp, 0.7)
            except (ValueError, TypeError):
                pass
        
        return (None, 0.0)
    
    def extract_cost(self, text: str, context_text: Optional[str] = None) -> Tuple[Optional[int], float]:
        """
        Extract asset cost/price.
        
        Args:
            text: Text to extract costs from (may be filtered to exclude address regions)
            context_text: Original unfiltered text for PIN code context checking.
                          If None, uses `text` for context (less accurate PIN detection).
                          
        IMPORTANT: PIN code detection requires address keywords like "Dist", "Pin", "Road"
        to properly identify 6-digit numbers as PIN codes vs prices. If `text` has been
        filtered to remove address regions, pass the original text as `context_text`.
        
        Returns: (value, confidence) or (None, 0.0)
        """
        valid_costs = []
        
        # Use original text for PIN context checking if provided
        # This ensures address keywords are available even if text is filtered
        pin_context = context_text if context_text is not None else text
        
        # Try labeled patterns first (most reliable)
        for pattern in self._cost_compiled[:4]:  # First 4 are labeled patterns
            matches = pattern.findall(text)
            for match in matches:
                cost = self._parse_indian_number(match)
                if cost and 50000 <= cost <= 5000000:
                    # Filter out PIN codes using full context text
                    if not self._is_likely_pin_code(cost, pin_context):
                        return (cost, 0.9)
        
        # Try currency patterns
        for pattern in self._cost_compiled[4:]:
            matches = pattern.findall(text)
            for match in matches:
                cost = self._parse_indian_number(match)
                if cost and 50000 <= cost <= 5000000:
                    # Filter out PIN codes using full context text
                    if not self._is_likely_pin_code(cost, pin_context):
                        valid_costs.append((cost, 0.8))
        
        # If we found valid costs from currency patterns, return highest confidence (then largest)
        if valid_costs:
            valid_costs.sort(key=lambda x: (x[1], x[0]), reverse=True)
            return valid_costs[0]
        
        # Fallback: find large numbers in Indian format (with commas - typical price format)
        # Prefer numbers with Indian comma formatting (X,XX,XXX) over plain 6-digit numbers
        indian_format_nums = re.findall(r'(\d{1,2},\d{2},\d{3})', text)
        for num in indian_format_nums:
            cost = self._parse_indian_number(num)
            if cost and 50000 <= cost <= 5000000:
                if not self._is_likely_pin_code(cost, pin_context):
                    valid_costs.append((cost, 0.7))
        
        # Also check for plain large numbers, but with lower confidence
        plain_nums = re.findall(r'(?<![,\d])(\d{5,7})(?![,\d])', text)
        for num in plain_nums:
            cost = self._parse_indian_number(num)
            if cost and 50000 <= cost <= 5000000:
                if not self._is_likely_pin_code(cost, pin_context):
                    valid_costs.append((cost, 0.5))
        
        if valid_costs:
            # Sort by confidence first, then by value (prefer larger amounts)
            valid_costs.sort(key=lambda x: (x[1], x[0]), reverse=True)
            return valid_costs[0]
        
        return (None, 0.0)
    
    def _parse_indian_number(self, num_str: str) -> Optional[int]:
        """Parse Indian number format (e.g., 5,25,000)."""
        if not num_str:
            return None
        try:
            # Remove commas, dots (thousands separator), spaces
            cleaned = re.sub(r'[,\.\s]', '', str(num_str))
            return int(cleaned)
        except ValueError:
            return None
    
    def _is_likely_pin_code(self, number: int, full_text: str) -> bool:
        """
        Check if a number is likely a PIN code rather than a price.
        
        Indian PIN codes are 6-digit numbers (100000-999999).
        We check context to determine if the number appears near address keywords.
        """
        # PIN codes are exactly 6 digits
        if not (100000 <= number <= 999999):
            return False
        
        num_str = str(number)
        
        # Check if number appears in PIN code context
        for pattern in self._pin_context_compiled:
            if pattern.search(full_text):
                # Check if this specific number is in the match
                matches = pattern.findall(full_text)
                for match in matches:
                    if num_str in str(match):
                        return True
        
        # Check proximity to address keywords
        address_keywords = [
            'dist', 'district', 'pin', 'pincode', 'taluka', 'tehsil', 'block',
            'road', 'street', 'nagar', 'colony', 'sector', 'phase',
            'raj', 'rajasthan', 'gujarat', 'maharashtra', 'mp', 'up',
            'contact', 'contect', 'phone', 'mobile', 'email', 'e-mail',
            'address', 'location', 'office'
        ]
        
        # Find the number in text and check surrounding context (100 chars)
        for match in re.finditer(re.escape(num_str), full_text, re.IGNORECASE):
            start = max(0, match.start() - 100)
            end = min(len(full_text), match.end() + 50)
            context = full_text[start:end].lower()
            
            if any(kw in context for kw in address_keywords):
                return True
        
        # Check if the number doesn't have typical price formatting (commas)
        # Prices are usually written as 5,50,000 or 5,25,000, PIN codes as 306115
        price_formatted = re.search(rf'\d{{1,2}},\d{{2}},?{num_str[-3:]}', full_text)
        if not price_formatted:
            # Number appears without Indian price formatting, more likely PIN
            plain_match = re.search(rf'(?<![,\d]){num_str}(?![,\d])', full_text)
            if plain_match:
                # Check if near address-like patterns
                start = max(0, plain_match.start() - 80)
                context = full_text[start:plain_match.end()].lower()
                if any(kw in context for kw in ['dist', 'pin', 'raj', 'road', 'contect', 'contact']):
                    return True
        
        return False
    
    def _normalize_hindi_numerals(self, text: str) -> str:
        """Convert Hindi/Devanagari numerals to Arabic numerals."""
        hindi_to_arabic = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }
        for hindi, arabic in hindi_to_arabic.items():
            text = text.replace(hindi, arabic)
        return text
    
    def extract_model(self, text: str, texts: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
        """
        Extract tractor model name.
        
        Returns: (value, confidence) or (None, 0.0)
        """
        # Normalize Hindi numerals to Arabic
        text = self._normalize_hindi_numerals(text)
        
        # Try brand-specific patterns
        for pattern in self._model_compiled:
            matches = pattern.findall(text)
            if matches:
                model = re.sub(r'\s+', ' ', matches[0]).strip()
                # Normalize common variations
                model = self._normalize_model(model)
                return (model, 0.9)
        
        # Try to find model in individual text lines
        if texts:
            for line in texts:
                line_normalized = self._normalize_hindi_numerals(line)
                for pattern in self._model_compiled:
                    matches = pattern.findall(line_normalized)
                    if matches:
                        model = re.sub(r'\s+', ' ', matches[0]).strip()
                        return (self._normalize_model(model), 0.85)
        
        return (None, 0.0)
    
    def _normalize_model(self, model: str) -> str:
        """Normalize model name for consistency."""
        if not model:
            return model
        
        # First normalize Hindi numerals
        model = self._normalize_hindi_numerals(model)
        
        # Fix OCR misreads of brand names
        model = re.sub(r'\b(?:SWARA|SWRAJ|SWARAI|SWRAR)\b', 'Swaraj', model, flags=re.IGNORECASE)
        model = re.sub(r'\b(?:solis|solus|s0lis|s[o0]l[iI1]s)\b', 'Solis', model, flags=re.IGNORECASE)
        model = re.sub(r'\bmahindra\b', 'Mahindra', model, flags=re.IGNORECASE)
        model = re.sub(r'\bjohn\s*deere\b', 'John Deere', model, flags=re.IGNORECASE)
        model = re.sub(r'\btafe\b', 'TAFE', model, flags=re.IGNORECASE)
        model = re.sub(r'\bswaraj\b', 'Swaraj', model, flags=re.IGNORECASE)
        model = re.sub(r'\bsonalika\b', 'Sonalika', model, flags=re.IGNORECASE)
        model = re.sub(r'\bmassey\s*ferguson\b', 'Massey Ferguson', model, flags=re.IGNORECASE)
        model = re.sub(r'\bnew\s*holland\b', 'New Holland', model, flags=re.IGNORECASE)
        model = re.sub(r'\beicher\b', 'Eicher', model, flags=re.IGNORECASE)
        model = re.sub(r'\b(?:farmtrac|faemtrac)\b', 'Farmtrac', model, flags=re.IGNORECASE)
        model = re.sub(r'\bpowertrac\b', 'Powertrac', model, flags=re.IGNORECASE)
        model = re.sub(r'\bkubota\b', 'Kubota', model, flags=re.IGNORECASE)
        model = re.sub(r'\bescorts\s*kubota\b', 'Escorts Kubota', model, flags=re.IGNORECASE)
        model = re.sub(r'\byuvo\b', 'Yuvo', model, flags=re.IGNORECASE)
        model = re.sub(r'\bарjun\b', 'Arjun', model, flags=re.IGNORECASE)
        model = re.sub(r'\b2wd\b', '2WD', model, flags=re.IGNORECASE)
        model = re.sub(r'\b4wd\b', '4WD', model, flags=re.IGNORECASE)
        model = re.sub(r'\bFE\b', 'FE', model, flags=re.IGNORECASE)
        model = re.sub(r'\bDI\b', 'DI', model, flags=re.IGNORECASE)
        model = re.sub(r'\bXT\b', 'XT', model, flags=re.IGNORECASE)
        model = re.sub(r'\bgear\s*pro\b', 'Gear Pro', model, flags=re.IGNORECASE)
        
        # Remove trailing "TRACTOR" word
        model = re.sub(r'\s*TRACTOR\s*$', '', model, flags=re.IGNORECASE)
        
        return model.strip()
    
    def extract_dealer(
        self,
        text: str,
        texts: List[str],
        boxes: Optional[List[List[int]]] = None
    ) -> Tuple[Optional[str], float]:
        """
        Extract dealer/seller name.
        
        Strategy:
        1. Look for labeled dealer name
        2. Find business names with "Ltd", "Pvt" etc. (header region first, then full doc)
        3. Pattern match company names with tractor/motor keywords
        4. Look for prominent uppercase names with business keywords
        5. Fall back to first meaningful line
        
        Returns: (value, confidence) or (None, 0.0)
        """
        # Strategy 1: Explicit labels
        for pattern in self.DEALER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dealer = match.group(1).strip()
                dealer = re.sub(r'[:\-\|]+$', '', dealer).strip()
                if self._validate_dealer_name(dealer):
                    return (dealer, 0.95)
        
        # Strategy 2: Check header lines (first 10 lines) for business names  
        for line in texts[:10]:
            line_lower = line.lower()
            
            # Check for business suffixes
            if any(suffix in line_lower for suffix in self.BUSINESS_SUFFIXES):
                dealer = self._clean_dealer_name(line)
                if self._validate_dealer_name(dealer):
                    return (dealer, 0.85)
        
        # Strategy 2b: Search full document for "... Ltd" or "... Pvt Ltd" patterns
        # These are high-confidence indicators of company names
        ltd_pattern = r'([A-Z][A-Za-z\s&]+(?:Ltd\.?|Limited|Pvt\.?\s*Ltd\.?|Private\s*Limited))'
        for match in re.finditer(ltd_pattern, text):
            candidate = match.group(1).strip()
            # Filter out generic phrases like "Full Cost of Tractor Ltd" (unlikely)
            if len(candidate) > 8 and not re.search(r'(cost|price|amount|total|full)', candidate, re.I):
                if self._validate_dealer_name(candidate):
                    return (candidate, 0.80)
        
        # Strategy 3: Pattern match company names with tractor/motor keywords
        company_patterns = [
            r'([A-Za-z][A-Za-z\s&]+(?:TRACTOR|Tractor|TRACTORS|Tractors|MOTORS|Motors|AUTO|Auto|AGENCIES|Agencies|ENTERPRISES|Enterprises))',
            r'([A-Z][A-Za-z\s&]+(?:Tractors?|Motors?|Auto|Agencies|Enterprises)[A-Za-z\s&]*(?:Pvt\.?|Private)?\.?\s*(?:Ltd\.?|Limited)?)',
            r'([A-Z][A-Za-z\s]+(?:Pvt\.?|Private)\s*(?:Ltd\.?|Limited))',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                dealer = self._clean_dealer_name(match.group(1))
                if self._validate_dealer_name(dealer):
                    return (dealer, 0.75)
        
        # Strategy 4: Look for uppercase words with TRACTOR/MOTOR in text
        uppercase_pattern = r'\b([A-Z][A-Z\s]+(?:TRACTOR|MOTOR|AUTO)[A-Z\s]*)\b'
        match = re.search(uppercase_pattern, text)
        if match:
            dealer = match.group(1).strip().title()
            if self._validate_dealer_name(dealer):
                return (dealer, 0.7)
        
        # Strategy 5: First line often contains dealer name (letterhead)
        if texts and len(texts[0]) > 5:
            first_line = texts[0].strip()
            # Skip common document headers
            skip_patterns = [
                r'^(Date|Invoice|Quotation|GST|Bill|Tax|Ref)',
                r'^[\(\[]',  # Lines starting with brackets
                r'^\d',      # Lines starting with numbers
            ]
            if not any(re.match(p, first_line, re.I) for p in skip_patterns):
                if self._validate_dealer_name(first_line):
                    return (first_line, 0.6)
        
        return (None, 0.0)
    
    def _clean_dealer_name(self, name: str) -> str:
        """Clean and normalize dealer name."""
        if not name:
            return name
        
        # Remove leading/trailing punctuation
        name = re.sub(r'^[:\-\|\s]+|[:\-\|\s]+$', '', name)
        
        # Remove phone numbers, emails
        name = re.sub(r'\b\d{10,}\b', '', name)
        name = re.sub(r'\S+@\S+', '', name)
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _validate_dealer_name(self, name: str) -> bool:
        """
        Validate if a string is a valid dealer name.
        Returns True if valid, False otherwise.
        """
        if not name or len(name) < 5:
            return False
        
        name_lower = name.lower().strip()
        
        # Must have at least one space (multi-word name) OR be a known company suffix
        if ' ' not in name and not re.search(r'(?:ltd|limited|pvt|tractors?|motors?)$', name_lower):
            return False
        
        # Reject email addresses or email-like patterns
        if '@' in name or re.search(r'[a-z]+@[a-z]+|gmail|yahoo|email|\.com', name_lower):
            return False
        
        # Reject generic service descriptions
        noise_phrases = {
            'sales service', 'sales & service', 'sales services',
            'spare parts', 'spares parts', 'service & spare',
            'authorized dealer', 'authorized dealer for',
            'sales service & spare parts', 'service & spare parts',
            'spare accesories', 'spare accessories', 'tractors spare',
            'service spare parts', 'sales & spare parts'
        }
        # Check if the name IS just a noise phrase
        if name_lower.strip() in noise_phrases:
            return False
        
        # Reject if name CONTAINS only generic service words (no actual company name)
        # Pattern: mostly generic words with maybe tractor brand thrown in
        generic_words = {'sales', 'service', 'spare', 'parts', 'authorized', 'dealer', 
                        'for', 'the', 'and', '&', 'accesories', 'accessories', 'autho',
                        'autho.', 'स्पेयर', 'पार्टस्', 'एवं', 'सेवा'}
        words = name_lower.split()
        if len(words) > 0:
            generic_count = sum(1 for w in words if w.rstrip('.,:') in generic_words)
            if generic_count / len(words) > 0.7 and len(words) >= 3:
                return False
        
        # Reject if starts with generic label words
        if re.match(r'^(?:for|to|from|authorized\s+dealer)\s+', name, re.IGNORECASE):
            stripped = re.sub(r'^(?:for|to|from|authorized\s+dealer\s+for?)\s*', '', name, flags=re.IGNORECASE)
            if len(stripped.strip()) < 5:
                return False
        
        # Reject patterns that are clearly service descriptions
        service_patterns = [
            r'^Sales\s+Service',
            r'^Service\s+&?\s*Spare',
            r'^Spare\s+Parts',
            r'^(?:Near|Opp|Behind|Adjacent)\s+',
            r'^Descri[pn]tion',  # OCR misread of "Description"
            r'^Autho\.?\s+Dealer',  # "Autho. Dealer..."
            r'Spare\s+Acce[s]*ories',  # "Spare Accesories"
            r'^\s*ट्रेक्टर्स\s+एवं\s+स्पेयर',  # Hindi "Tractors and Spare Parts"
        ]
        for pattern in service_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        
        # Reject corrupted Unicode (broken Hindi/Gujarati)
        if re.search(r'α[ñÑ][^\s]{2,}', name):
            return False
        
        return True
    
    def get_text_regions(
        self,
        texts: List[str],
        boxes: List[List[int]],
        image_height: int
    ) -> Dict[str, List[str]]:
        """
        Divide text by document region.
        
        Returns dict with 'header', 'body', 'footer' text lists.
        """
        header_texts = []
        body_texts = []
        footer_texts = []
        
        header_cutoff = image_height * 0.2
        footer_cutoff = image_height * 0.75
        
        for text, box in zip(texts, boxes):
            y_center = (box[1] + box[3]) / 2
            
            if y_center < header_cutoff:
                header_texts.append(text)
            elif y_center > footer_cutoff:
                footer_texts.append(text)
            else:
                body_texts.append(text)
        
        return {
            'header': header_texts,
            'body': body_texts,
            'footer': footer_texts
        }