"""
YOLO Detector Module
Signature and stamp detection using YOLOv8 or intelligent fallback heuristics

Features:
- YOLO model inference (if available)
- Fallback: Color-based stamp detection
- Fallback: Contour-based signature detection
- Full document search (not just fixed regions)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
from loguru import logger

# ============================================================
# OFFLINE DEPLOYMENT: Hardcoded settings (no .env required)
# ============================================================
# Disable pin_memory for DataLoaders (required for some GPU configurations)
os.environ['PIN_MEMORY'] = 'False'

# Disable Ultralytics telemetry and online checks for offline deployment
os.environ['YOLO_OFFLINE'] = 'True'

try:
    from ultralytics import YOLO
    # Configure ultralytics settings for offline deployment
    try:
        from ultralytics import settings
        settings.update({'sync': False})  # Disable online sync
    except Exception:
        pass
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Detection will be limited.")


class YOLODetector:
    """
    Detect signatures and stamps in invoice documents.
    
    Uses YOLO model if provided, otherwise intelligent heuristics.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
            device: 'cuda' or 'cpu'
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.use_fallback = True
        
        # Try loading YOLO model
        if model_path and Path(model_path).exists() and YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.use_fallback = False
                logger.info(f"Loaded YOLO model: {model_path}")
            except Exception as e:
                logger.warning(f"YOLO load failed: {e}")
    
    def detect(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict:
        """
        Detect signature and stamp in image.
        
        Returns:
            Dict with 'signature' and 'stamp' containing 'present', 'bbox', 'confidence'
        """
        img_array = self._to_numpy(image)
        
        if not self.use_fallback and self.model:
            return self._yolo_detection(img_array)
        else:
            return self._intelligent_fallback(img_array)
    
    def _yolo_detection(self, image: np.ndarray) -> Dict:
        """Run YOLO model detection."""
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = {
            'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
        }
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = [int(x) for x in box.xyxy[0].tolist()]
                
                cls_name = 'signature' if cls_id == 0 else 'stamp'
                
                if conf > detections[cls_name]['confidence']:
                    detections[cls_name] = {
                        'present': True,
                        'bbox': bbox,
                        'confidence': round(conf, 3)
                    }
        
        return detections
    
    def _intelligent_fallback(self, image: np.ndarray) -> Dict:
        """
        Intelligent heuristic detection for signatures and stamps.
        
        Strategy:
        1. Search bottom 50% for signature (dark strokes)
        2. Search entire document for colored stamps (blue/red)
        3. Use contour analysis with multiple criteria
        """
        if not CV2_AVAILABLE:
            return {
                'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
                'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
            }
        
        h, w = image.shape[:2]
        
        detections = {
            'signature': {'present': False, 'bbox': None, 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': None, 'confidence': 0.0}
        }
        
        # === SIGNATURE DETECTION ===
        # Signatures typically in bottom half
        sig_region_start = int(h * 0.5)
        sig_region = image[sig_region_start:, :]
        
        sig_result = self._detect_signature_region(sig_region, 0, sig_region_start)
        if sig_result:
            detections['signature'] = sig_result
        
        # === STAMP DETECTION ===
        # Stamps can be anywhere but usually bottom, check full image
        stamp_result = self._detect_stamp_full(image)
        if stamp_result:
            detections['stamp'] = stamp_result
        
        return detections
    
    def _detect_signature_region(
        self,
        region: np.ndarray,
        x_offset: int,
        y_offset: int
    ) -> Optional[Dict]:
        """
        Detect signature using ink stroke analysis.
        
        Signatures are characterized by:
        - Dark strokes on light background
        - Horizontal bias (wider than tall)
        - Low-medium density
        - Connected components
        """
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Adaptive threshold for varying backgrounds
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 10
        )
        
        # Morphological operations to connect signature strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Skip tiny or huge regions
            if area < 400 or area > 80000:
                continue
            
            # Signature heuristics
            aspect = cw / max(ch, 1)
            
            # Signatures are typically wider than tall (0.5 to 8 aspect ratio)
            if not (0.5 < aspect < 8):
                continue
            
            # Minimum width
            if cw < 40:
                continue
            
            # Calculate ink density
            roi = thresh[y:y+ch, x:x+cw]
            density = np.sum(roi > 0) / (cw * ch + 1e-6)
            
            # Signatures have low-medium density (not solid blocks)
            if not (0.03 < density < 0.4):
                continue
            
            # Score based on multiple factors
            score = 0.5
            
            # Bonus for typical signature aspect ratio (2-5)
            if 1.5 < aspect < 6:
                score += 0.15
            
            # Bonus for reasonable size
            if 1000 < area < 30000:
                score += 0.1
            
            # Bonus for moderate density
            if 0.08 < density < 0.25:
                score += 0.1
            
            candidates.append({
                'bbox': [x + x_offset, y + y_offset, x + cw + x_offset, y + ch + y_offset],
                'confidence': min(0.85, score),
                'score': score * area  # Larger signatures preferred
            })
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda c: c['score'])
            return {
                'present': True,
                'bbox': best['bbox'],
                'confidence': round(best['confidence'], 3)
            }
        
        return None
    
    def _detect_stamp_full(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect stamp in full image using color analysis.
        
        Stamps are characterized by:
        - Distinct color (blue, red, purple)
        - Roughly circular/oval shape
        - Usually in corners or bottom
        """
        if len(image.shape) != 3:
            return None
        
        h, w = image.shape[:2]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Blue stamp detection (most common in India)
        blue_lower = np.array([100, 30, 30])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Red stamp detection
        red_lower1 = np.array([0, 30, 30])
        red_upper1 = np.array([12, 255, 255])
        red_lower2 = np.array([165, 30, 30])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        
        # Purple stamp detection
        purple_lower = np.array([130, 30, 30])
        purple_upper = np.array([165, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        
        # Combine all color masks
        stamp_mask = blue_mask | red_mask | purple_mask
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Minimum area for stamp
            if area < 500 or area > 150000:
                continue
            
            # Stamps are roughly square/circular
            aspect = cw / max(ch, 1)
            if not (0.3 < aspect < 3.0):
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            
            # Calculate position score (bottom half preferred)
            y_center = y + ch / 2
            position_bonus = 0.1 if y_center > h * 0.5 else 0
            
            # Score calculation
            score = 0.5 + circularity * 0.3 + position_bonus
            
            # Bonus for reasonable stamp size
            if 2000 < area < 50000:
                score += 0.1
            
            candidates.append({
                'bbox': [x, y, x + cw, y + ch],
                'confidence': min(0.9, score),
                'area': area
            })
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda c: c['area'] * c['confidence'])
            return {
                'present': True,
                'bbox': best['bbox'],
                'confidence': round(best['confidence'], 3)
            }
        
        return None
    
    def _to_numpy(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """Convert various image formats to numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        else:
            return np.array(Image.open(image))
    
    def visualize(
        self,
        image: np.ndarray,
        detections: Dict,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: Detection results
            output_path: Optional save path
            
        Returns:
            Annotated image
        """
        if not CV2_AVAILABLE:
            return image
        
        img_copy = image.copy()
        
        colors = {
            'signature': (0, 255, 0),  # Green
            'stamp': (255, 0, 0)       # Red (in RGB)
        }
        
        for det_type, color in colors.items():
            det = detections.get(det_type, {})
            if det.get('present') and det.get('bbox'):
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
                
                label = f"{det_type}: {det.get('confidence', 0):.2f}"
                cv2.putText(
                    img_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
        
        if output_path:
            # Convert RGB to BGR for OpenCV save
            cv2.imwrite(output_path, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        
        return img_copy
