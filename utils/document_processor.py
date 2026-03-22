"""
Document Processor Module
Handles PDF/Image ingestion, preprocessing, and normalization
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from loguru import logger

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class DocumentProcessor:
    """
    Handles document ingestion and preprocessing.
    Supports PDF, PNG, JPG, TIFF formats.
    """
    
    def __init__(
        self,
        target_dpi: int = 300,
        enhance_quality: bool = True,
        max_dimension: int = 4096
    ):
        self.target_dpi = target_dpi
        self.enhance_quality = enhance_quality
        self.max_dimension = max_dimension
        self.supported_formats = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}
    
    def process(self, file_path: Union[str, Path]) -> List[Tuple[Image.Image, dict]]:
        """
        Process document and return list of (image, metadata) tuples.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Convert to images
        if suffix == '.pdf':
            images = self._process_pdf(file_path)
        else:
            images = self._process_image(file_path)
        
        # Process each page
        processed = []
        for idx, img in enumerate(images):
            img = self._resize_if_needed(img)
            
            if self.enhance_quality:
                img = self._enhance_image(img)
            
            metadata = {
                'source_file': str(file_path),
                'page_number': idx + 1,
                'total_pages': len(images),
                'size': img.size
            }
            processed.append((img, metadata))
        
        return processed
    
    def _process_pdf(self, file_path: Path) -> List[Image.Image]:
        """Convert PDF to images."""
        # Try pdf2image first only when Poppler is available in PATH.
        # This prevents avoidable runtime warnings on systems without Poppler.
        poppler_available = bool(shutil.which('pdftoppm') or shutil.which('pdftoppm.exe'))
        if PDF2IMAGE_AVAILABLE and poppler_available:
            try:
                images = convert_from_path(str(file_path), dpi=self.target_dpi)
                return [img.convert('RGB') for img in images]
            except Exception as e:
                logger.warning(f"pdf2image failed: {e}")
        
        # Fallback to PyMuPDF
        if PYMUPDF_AVAILABLE:
            return self._process_pdf_pymupdf(file_path)
        
        raise ImportError("No PDF library available. Install pdf2image or PyMuPDF.")
    
    def _process_pdf_pymupdf(self, file_path: Path) -> List[Image.Image]:
        """Convert PDF using PyMuPDF."""
        doc = fitz.open(str(file_path))
        images = []
        
        zoom = self.target_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        doc.close()
        return images
    
    def _process_image(self, file_path: Path) -> List[Image.Image]:
        """Process image file."""
        img = Image.open(file_path)
        
        # Convert to RGB, handling alpha channels properly
        if img.mode in ('RGBA', 'LA', 'P', 'PA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            
            # Convert palette and grayscale+alpha modes to RGBA for uniform handling
            if img.mode == 'P':
                img = img.convert('RGBA')
            elif img.mode == 'LA':
                # LA (grayscale + alpha) also needs conversion to preserve alpha
                img = img.convert('RGBA')
            elif img.mode == 'PA':
                # PA (palette + alpha) needs conversion too
                img = img.convert('RGBA')
            
            # Now img is either RGBA (converted) or was already RGBA
            if img.mode == 'RGBA':
                # Use alpha channel as mask for proper transparency handling
                alpha_mask = img.split()[-1]  # Get alpha channel
                background.paste(img, mask=alpha_mask)
            else:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        return [img]
    
    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        """Resize if exceeds maximum dimension."""
        w, h = img.size
        if max(w, h) > self.max_dimension:
            ratio = self.max_dimension / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img
    
    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Enhance image for better OCR."""
        # Sharpen
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        return img
    
    def to_numpy(self, img: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array."""
        return np.array(img)
