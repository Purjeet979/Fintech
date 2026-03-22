"""
OCR Engine Module
EasyOCR-based multilingual text extraction with layout analysis

Key Features:
- Multi-language OCR (English, Hindi, Gujarati)
- Layout-aware text extraction with reading order
- Table structure detection
- Key-value pair spatial analysis
- Confidence scoring with quality metrics

Optimized for diverse invoice layouts and unknown formats.
"""

import os
import re
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger

# ============================================================
# OFFLINE DEPLOYMENT: Hardcoded settings (no .env required)
# ============================================================

# Suppress harmless pin_memory warning from PyTorch DataLoader
warnings.filterwarnings('ignore', message=".*pin_memory.*")

# Disable EasyOCR model source connectivity check for offline deployment
os.environ['EASYOCR_MODULE_PATH'] = os.environ.get('EASYOCR_MODULE_PATH', os.path.expanduser('~/.EasyOCR'))
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Primary OCR: EasyOCR (stable, accurate, multilingual)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install: pip install easyocr")

# Fallback: PaddleOCR
PADDLE_AVAILABLE = False
try:
    import os
    os.environ.setdefault('FLAGS_use_mkldnn', '0')
    os.environ.setdefault('FLAGS_enable_pir_api', '0')
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    pass


class OCREngine:
    """
    Multilingual OCR Engine using EasyOCR (primary) with PaddleOCR fallback.
    
    Supports: English, Hindi (Devanagari), Gujarati, and mixed text.
    Uses GPU acceleration when available for best performance.
    """
    
    # EasyOCR language codes
    LANG_CODES = {
        'english': 'en',
        'hindi': 'hi',
        'gujarati': 'gu',
    }
    
    def __init__(
        self,
        use_gpu: bool = True,
        use_angle_cls: bool = True,
        drop_score: float = 0.4,
        enable_multilingual: bool = True
    ):
        """
        Initialize OCR engine with multilingual support.
        
        Args:
            use_gpu: Use GPU acceleration (CUDA)
            use_angle_cls: Enable angle classification for rotated text
            drop_score: Minimum confidence threshold
            enable_multilingual: Enable Hindi/Gujarati in addition to English
        """
        self.drop_score = drop_score
        self.enable_multilingual = enable_multilingual
        self.use_gpu = use_gpu
        self.reader = None
        
        if not EASYOCR_AVAILABLE and not PADDLE_AVAILABLE:
            raise ImportError(
                "No OCR engine available. Install: pip install easyocr "
                "or pip install paddleocr paddlepaddle"
            )
        
        # Initialize EasyOCR (primary)
        if EASYOCR_AVAILABLE:
            languages = ['en']
            if enable_multilingual:
                languages.extend(['hi'])  # Hindi (Devanagari) - Gujarati not supported by EasyOCR yet
            
            logger.info(f"Loading EasyOCR with languages: {languages}")
            try:
                self.reader = easyocr.Reader(
                    languages,
                    gpu=use_gpu,
                    verbose=False
                )
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.reader = None
        
        # Fallback: PaddleOCR (if EasyOCR fails)
        self.paddle_engine = None
        if self.reader is None and PADDLE_AVAILABLE:
            logger.info("Falling back to PaddleOCR...")
            try:
                self.paddle_engine = self._create_paddleocr(use_gpu=use_gpu)
            except Exception as e:
                logger.warning(f"PaddleOCR fallback failed: {e}")
        
        if self.reader is None and self.paddle_engine is None:
            raise RuntimeError("Failed to initialize any OCR engine")

    def _create_paddleocr(self, use_gpu: bool):
        """Create PaddleOCR with compatible arguments."""
        try:
            return PaddleOCR(lang='en', use_gpu=use_gpu, show_log=False)
        except (TypeError, ValueError):
            try:
                return PaddleOCR(lang='en', device='gpu' if use_gpu else 'cpu')
            except Exception:
                return PaddleOCR(lang='en')
    
    def extract_text(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        languages: List[str] = None
    ) -> Dict:
        """
        Extract text from image with multi-language support.
        
        Uses EasyOCR (primary) or PaddleOCR (fallback) for accurate extraction.
        
        Args:
            image: Input image (PIL, numpy array, or path)
            languages: Not used (EasyOCR handles multilingual automatically)
            
        Returns:
            Dict with 'texts', 'boxes', 'confidences', 'full_text', 'language'
        """
        img_array = self._to_numpy(image)
        
        result = None
        
        # Primary: EasyOCR
        if self.reader is not None:
            try:
                result = self._run_easyocr(img_array)
                logger.debug(f"EasyOCR extracted {result.get('count', 0)} text elements")
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
                result = None
        
        # Fallback: PaddleOCR
        if (result is None or result.get('count', 0) == 0) and self.paddle_engine is not None:
            logger.info("Trying PaddleOCR fallback...")
            try:
                result = self._run_paddleocr(img_array)
                logger.debug(f"PaddleOCR extracted {result.get('count', 0)} text elements")
            except Exception as e:
                logger.warning(f"PaddleOCR fallback failed: {e}")
        
        if result is None:
            result = self._empty_result()
        
        # Detect primary language
        result['language'] = self.detect_language(result.get('full_text', ''))
        
        return result
    
    def _run_easyocr(self, img_array: np.ndarray) -> Dict:
        """Run EasyOCR and return standardized result."""
        # EasyOCR returns: [[bbox, text, confidence], ...]
        # bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        raw_results = self.reader.readtext(img_array)
        
        texts = []
        boxes = []
        confidences = []
        
        for detection in raw_results:
            bbox, text, conf = detection
            
            if conf >= self.drop_score and text.strip():
                texts.append(text.strip())
                # Convert polygon to [x1, y1, x2, y2]
                boxes.append(self._normalize_box(bbox))
                confidences.append(float(conf))
        
        # Sort by position (top-to-bottom, left-to-right)
        if texts:
            sorted_data = self._sort_by_position(texts, boxes, confidences)
            texts, boxes, confidences = sorted_data
        
        return {
            'texts': texts,
            'boxes': boxes,
            'confidences': confidences,
            'full_text': '\n'.join(texts),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'count': len(texts)
        }
    
    def _run_paddleocr(self, img_array: np.ndarray) -> Dict:
        """Run PaddleOCR fallback and return standardized result."""
        try:
            result = self.paddle_engine.ocr(img_array)
        except TypeError:
            result = self.paddle_engine.ocr(img_array, cls=True)
        
        return self._parse_paddle_results(result)
    
    def _parse_paddle_results(self, result) -> Dict:
        """Parse PaddleOCR raw results (handles both old and new API formats)."""
        texts = []
        boxes = []
        confidences = []
        
        if not result:
            return self._empty_result()
        
        # New PaddleOCR (PP-OCRv5) returns dict or list of dicts
        if isinstance(result, dict):
            rec_texts = result.get('rec_texts', result.get('texts', []))
            rec_scores = result.get('rec_scores', result.get('scores', []))
            dt_polys = result.get('dt_polys', result.get('boxes', []))
            
            for i, text in enumerate(rec_texts):
                if text and text.strip():
                    conf = rec_scores[i] if i < len(rec_scores) else 0.5
                    if conf >= self.drop_score:
                        texts.append(text.strip())
                        if i < len(dt_polys):
                            boxes.append(self._normalize_box(dt_polys[i]))
                        else:
                            boxes.append([0, 0, 0, 0])
                        confidences.append(float(conf))
        
        # Handle list of dicts (batch results)
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            for item in result:
                rec_texts = item.get('rec_texts', item.get('texts', []))
                rec_scores = item.get('rec_scores', item.get('scores', []))
                dt_polys = item.get('dt_polys', item.get('boxes', []))
                
                for i, text in enumerate(rec_texts):
                    if text and text.strip():
                        conf = rec_scores[i] if i < len(rec_scores) else 0.5
                        if conf >= self.drop_score:
                            texts.append(text.strip())
                            if i < len(dt_polys):
                                boxes.append(self._normalize_box(dt_polys[i]))
                            else:
                                boxes.append([0, 0, 0, 0])
                            confidences.append(float(conf))
        
        # Old PaddleOCR format: [[box, (text, conf)], ...]
        elif isinstance(result, list) and result:
            data = result[0] if result[0] else result
            if isinstance(data, list):
                for line in data:
                    if line:
                        try:
                            if isinstance(line, (list, tuple)) and len(line) >= 2:
                                box = line[0]
                                text_conf = line[1]
                                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                                    text, conf = text_conf[0], text_conf[1]
                                elif isinstance(text_conf, str):
                                    text, conf = text_conf, 0.8
                                else:
                                    continue
                                
                                if conf >= self.drop_score and text and text.strip():
                                    texts.append(text.strip())
                                    boxes.append(self._normalize_box(box))
                                    confidences.append(float(conf))
                        except (ValueError, TypeError, IndexError):
                            continue
        
        return {
            'texts': texts,
            'boxes': boxes,
            'confidences': confidences,
            'full_text': '\n'.join(texts),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'count': len(texts)
        }
    
    def _empty_result(self) -> Dict:
        """Return empty OCR result structure."""
        return {
            'texts': [],
            'boxes': [],
            'confidences': [],
            'full_text': '',
            'avg_confidence': 0.0,
            'count': 0
        }
    
    def _merge_multilingual_results(self, all_results: Dict) -> Dict:
        """
        Merge OCR results from multiple languages.
        
        Strategy: 
        - Use English results as base (better for numbers, model names)
        - Add unique Hindi/Gujarati text not captured by English
        - Avoid duplicate detections using IoU
        """
        if not all_results:
            return {
                'texts': [], 'boxes': [], 'confidences': [],
                'full_text': '', 'avg_confidence': 0.0
            }
        
        # Start with English results (best for structured data)
        if 'en' in all_results:
            merged = {
                'texts': list(all_results['en']['texts']),
                'boxes': list(all_results['en']['boxes']),
                'confidences': list(all_results['en']['confidences'])
            }
        else:
            merged = {'texts': [], 'boxes': [], 'confidences': []}
        
        # Add non-duplicate results from other languages
        for lang, result in all_results.items():
            if lang == 'en':
                continue
            
            for i, (text, box, conf) in enumerate(zip(
                result['texts'], result['boxes'], result['confidences']
            )):
                # Check if this box overlaps with existing
                is_duplicate = False
                for existing_box in merged['boxes']:
                    if self._calculate_iou(box, existing_box) > 0.5:
                        is_duplicate = True
                        break
                
                # Add if unique and contains non-ASCII (likely Hindi/Gujarati)
                if not is_duplicate and self._has_non_ascii(text):
                    merged['texts'].append(text)
                    merged['boxes'].append(box)
                    merged['confidences'].append(conf)
        
        # Sort by position
        if merged['boxes']:
            sorted_data = self._sort_by_position(
                merged['texts'], merged['boxes'], merged['confidences']
            )
            merged['texts'], merged['boxes'], merged['confidences'] = sorted_data
        
        merged['full_text'] = '\n'.join(merged['texts'])
        merged['avg_confidence'] = np.mean(merged['confidences']) if merged['confidences'] else 0.0
        
        return merged
    
    def _normalize_box(self, box: List) -> List[int]:
        """Convert polygon to [x1, y1, x2, y2] format."""
        box = np.array(box)
        return [
            int(min(box[:, 0])),
            int(min(box[:, 1])),
            int(max(box[:, 0])),
            int(max(box[:, 1]))
        ]
    
    def _sort_by_position(
        self,
        texts: List[str],
        boxes: List[List[int]],
        confidences: List[float]
    ) -> Tuple[List, List, List]:
        """Sort by reading order (top-to-bottom, left-to-right)."""
        y_tolerance = 20
        
        items = list(zip(texts, boxes, confidences))
        items.sort(key=lambda x: (x[1][1] // y_tolerance, x[1][0]))
        
        return (
            [i[0] for i in items],
            [i[1] for i in items],
            [i[2] for i in items]
        )
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union."""
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
    
    def _has_non_ascii(self, text: str) -> bool:
        """Check if text contains non-ASCII characters (Hindi/Gujarati)."""
        return any(ord(c) > 127 for c in text)
    
    def _to_numpy(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """Convert various image formats to numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        else:
            return np.array(Image.open(image))
    
    def detect_language(self, text: str) -> str:
        """
        Detect primary language based on script analysis.
        
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
        
        # Determine primary language
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
    
    def get_text_in_region(
        self,
        ocr_result: Dict,
        region: Tuple[int, int, int, int]
    ) -> List[str]:
        """Get text elements within a specific image region."""
        rx1, ry1, rx2, ry2 = region
        texts_in_region = []
        
        for text, box in zip(ocr_result.get('texts', []), ocr_result.get('boxes', [])):
            # Check if box center is in region
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                texts_in_region.append(text)
        
        return texts_in_region
    
    def get_structured_output(self, ocr_result: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Get structured OCR output with layout analysis.
        
        Returns:
            Dict with:
            - lines: Text grouped by reading lines
            - regions: Header/body/footer segmentation
            - key_value_candidates: Detected key-value patterns
            - tables: Detected table structures
        """
        texts = ocr_result.get('texts', [])
        boxes = ocr_result.get('boxes', [])
        
        if not texts or not boxes:
            return {'lines': [], 'regions': {}, 'key_value_candidates': [], 'tables': []}
        
        width, height = image_size
        
        # Group into lines
        lines = self._group_into_lines(texts, boxes)
        
        # Segment into regions
        regions = self._segment_regions(texts, boxes, height)
        
        # Detect key-value candidates
        kv_candidates = self._detect_kv_candidates(texts, boxes)
        
        # Detect table structures
        tables = self._detect_tables(texts, boxes)
        
        return {
            'lines': lines,
            'regions': regions,
            'key_value_candidates': kv_candidates,
            'tables': tables
        }
    
    def _group_into_lines(
        self,
        texts: List[str],
        boxes: List[List[int]],
        y_tolerance: int = None  # Now computed adaptively if not provided
    ) -> List[Dict]:
        """
        Group text into logical lines based on y-coordinates.
        
        Uses ADAPTIVE y_tolerance based on median text height for resolution independence.
        
        Returns list of lines with text and bounding box.
        """
        if not texts:
            return []
        
        # === ADAPTIVE Y-TOLERANCE ===
        # Compute from median text height if not explicitly provided
        if y_tolerance is None:
            heights = [abs(b[3] - b[1]) for b in boxes if len(b) >= 4 and b[3] > b[1]]
            if heights:
                heights.sort()
                median_height = heights[len(heights) // 2]
                # Use 50% of median text height as tolerance
                y_tolerance = max(5, int(median_height * 0.5))
            else:
                y_tolerance = 15  # Fallback
        
        # Sort by y-coordinate
        elements = sorted(zip(texts, boxes), key=lambda x: (x[1][1], x[1][0]))
        
        lines = []
        current_line_texts = []
        current_line_boxes = []
        current_y = elements[0][1][1]
        
        for text, box in elements:
            if abs(box[1] - current_y) <= y_tolerance:
                current_line_texts.append(text)
                current_line_boxes.append(box)
            else:
                if current_line_texts:
                    # Sort by x within line
                    sorted_items = sorted(zip(current_line_texts, current_line_boxes),
                                         key=lambda x: x[1][0])
                    lines.append({
                        'text': ' '.join([t for t, _ in sorted_items]),
                        'elements': [{'text': t, 'box': b} for t, b in sorted_items],
                        'bbox': self._get_line_bbox(current_line_boxes)
                    })
                current_line_texts = [text]
                current_line_boxes = [box]
                current_y = box[1]
        
        # Add last line
        if current_line_texts:
            sorted_items = sorted(zip(current_line_texts, current_line_boxes),
                                 key=lambda x: x[1][0])
            lines.append({
                'text': ' '.join([t for t, _ in sorted_items]),
                'elements': [{'text': t, 'box': b} for t, b in sorted_items],
                'bbox': self._get_line_bbox(current_line_boxes)
            })
        
        return lines
    
    def _get_line_bbox(self, boxes: List[List[int]]) -> List[int]:
        """Get bounding box encompassing all boxes in a line."""
        if not boxes:
            return [0, 0, 0, 0]
        
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        
        return [x1, y1, x2, y2]
    
    def _segment_regions(
        self,
        texts: List[str],
        boxes: List[List[int]],
        image_height: int
    ) -> Dict[str, List[str]]:
        """
        Segment document into header, body, footer regions.
        
        Typical invoice layout:
        - Header (top 20%): Dealer name, logo, contact info
        - Body (20-75%): Items, specifications, prices
        - Footer (bottom 25%): Totals, signatures, stamps
        """
        header_cutoff = image_height * 0.20
        footer_cutoff = image_height * 0.75
        
        regions = {'header': [], 'body': [], 'footer': []}
        
        for text, box in zip(texts, boxes):
            y_center = (box[1] + box[3]) / 2
            
            if y_center < header_cutoff:
                regions['header'].append(text)
            elif y_center > footer_cutoff:
                regions['footer'].append(text)
            else:
                regions['body'].append(text)
        
        return regions
    
    def _detect_kv_candidates(
        self,
        texts: List[str],
        boxes: List[List[int]]
    ) -> List[Dict]:
        """
        Detect potential key-value pairs based on spatial patterns.
        
        Patterns detected:
        1. "Key: Value" in same text
        2. Key on left, value on right (same line)
        3. Key above, value below
        """
        candidates = []
        
        # Pattern 1: Colon-separated in same text
        colon_pattern = re.compile(r'^([^:]+):\s*(.+)$')
        for i, text in enumerate(texts):
            match = colon_pattern.match(text)
            if match:
                candidates.append({
                    'type': 'colon_separated',
                    'key': match.group(1).strip(),
                    'value': match.group(2).strip(),
                    'box': boxes[i],
                    'confidence': 0.9
                })
        
        # Pattern 2: Adjacent horizontal pairs
        for i, (text1, box1) in enumerate(zip(texts, boxes)):
            # Skip if already has value (contains colon)
            if ':' in text1 and len(text1.split(':')[1].strip()) > 0:
                continue
            
            # Look for text to the right
            for j, (text2, box2) in enumerate(zip(texts, boxes)):
                if i == j:
                    continue
                
                # Check if same line and to the right
                y_overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])
                height = min(box1[3] - box1[1], box2[3] - box2[1])
                
                if y_overlap > height * 0.5 and box2[0] > box1[2]:
                    distance = box2[0] - box1[2]
                    if distance < 200:  # Close enough
                        candidates.append({
                            'type': 'horizontal_pair',
                            'key': text1,
                            'value': text2,
                            'key_box': box1,
                            'value_box': box2,
                            'confidence': 0.8
                        })
        
        return candidates
    
    def _detect_tables(
        self,
        texts: List[str],
        boxes: List[List[int]]
    ) -> List[Dict]:
        """
        Detect table structures based on column alignment.
        
        Tables are characterized by:
        - Multiple rows with similar y-coordinates
        - Consistent column x-alignments
        - Header row with descriptive text
        """
        if len(texts) < 4:
            return []
        
        # Group by rows
        rows = []
        elements = sorted(zip(texts, boxes), key=lambda x: x[1][1])
        
        current_row = []
        current_y = elements[0][1][1]
        y_tolerance = 20
        
        for text, box in elements:
            if abs(box[1] - current_y) <= y_tolerance:
                current_row.append({'text': text, 'box': box})
            else:
                if len(current_row) >= 2:  # Multi-column row
                    rows.append(current_row)
                current_row = [{'text': text, 'box': box}]
                current_y = box[1]
        
        if len(current_row) >= 2:
            rows.append(current_row)
        
        # Identify tables (consecutive rows with similar column count)
        tables = []
        if len(rows) >= 2:
            table_rows = []
            expected_cols = len(rows[0])
            
            for row in rows:
                if abs(len(row) - expected_cols) <= 1:
                    table_rows.append(row)
                else:
                    if len(table_rows) >= 2:
                        tables.append({
                            'rows': table_rows,
                            'num_columns': expected_cols,
                            'header': table_rows[0] if table_rows else None
                        })
                    table_rows = [row]
                    expected_cols = len(row)
            
            if len(table_rows) >= 2:
                tables.append({
                    'rows': table_rows,
                    'num_columns': expected_cols,
                    'header': table_rows[0] if table_rows else None
                })
        
        return tables
    
    def extract_with_layout(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Dict:
        """
        Extract text with full layout analysis.
        
        Returns enhanced OCR result with:
        - Standard OCR output
        - Layout structure
        - Key-value pairs
        - Table data
        """
        # Get base OCR result
        ocr_result = self.extract_text(image)
        
        # Get image size
        img_array = self._to_numpy(image)
        height, width = img_array.shape[:2]
        
        # Add structured output
        structured = self.get_structured_output(ocr_result, (width, height))
        ocr_result.update(structured)
        
        return ocr_result
