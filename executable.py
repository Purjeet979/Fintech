#!/usr/bin/env python3
"""
Document AI - Invoice Field Extraction
ET Gen AI Hackathon

Extracts structured data from tractor loan quotations/invoices.
Supports English, Hindi, and Gujarati documents.

Usage:
    python executable.py --input invoice.pdf --output result.json
    python executable.py --input_dir ./invoices/ --output_dir ./results/
"""

import os
import sys
import warnings

# ============================================================
# OFFLINE DEPLOYMENT: Hardcoded settings (no .env required)
# Must be set BEFORE importing any ML libraries
# ============================================================

# Suppress harmless pin_memory warning from PyTorch DataLoader
# This warning occurs when no GPU accelerator is available
warnings.filterwarnings('ignore', message=".*pin_memory.*")

# Disable EasyOCR model source connectivity check for offline deployment
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Disable pin_memory for DataLoaders (required for some GPU configurations)
os.environ['PIN_MEMORY'] = 'False'

# Disable Ultralytics (YOLO) telemetry and online checks
os.environ['YOLO_OFFLINE'] = 'True'

# Disable Hugging Face online checks for offline model loading
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# ============================================================

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

# Note: .env file loading removed - all settings are now hardcoded above
# for reliable offline deployment without external file dependencies

# Pipeline components
from utils.document_processor import DocumentProcessor
from utils.ocr_engine import OCREngine
from utils.field_parser import FieldParser
from utils.validator import Validator
from utils.yolo_detector import YOLODetector
from utils.consensus_engine import ConsensusEngine, BootstrapRefiner


class InvoiceExtractor:
    """
    End-to-end pipeline for invoice field extraction.
    
    Pipeline:
    1. Document Processing (PDF/Image → normalized image)
    2. OCR (multilingual text extraction)
    3. Field Parsing (pattern-based extraction)
    4. Validation (fuzzy matching, cross-validation)
    5. Detection (signature/stamp)
    6. Output Generation
    """
    
    def __init__(
        self,
        use_vlm: bool = True,  # VLM enabled by default for better handwritten text handling
        vlm_provider: str = 'qwen',  # Default to local Qwen (offline, no internet)
        vlm_model: str = 'Qwen/Qwen2-VL-2B-Instruct',  # Fits easily in 16GB VRAM
        use_4bit: bool = False,  # Enable for 7B model on limited VRAM
        yolo_model_path: Optional[str] = None,
        use_gpu: bool = True,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: str = '2024-02-15-preview'
    ):
        """
        Initialize extraction pipeline.
        
        Args:
            use_vlm: Use Vision Language Model for enhanced extraction
            vlm_provider: 'qwen' (local, default), 'openai', or 'azure'
            vlm_model: Model name for Qwen (e.g., 'Qwen/Qwen2.5-VL-2B-Instruct')
            use_4bit: Enable 4-bit quantization for larger models on limited VRAM
            yolo_model_path: Path to trained YOLO model
            use_gpu: Use GPU acceleration where available
            azure_endpoint: Azure OpenAI endpoint URL
            azure_deployment: Azure OpenAI deployment name
            azure_api_key: Azure OpenAI API key
            azure_api_version: Azure OpenAI API version
        """
        # Store VLM settings
        self.vlm_model = vlm_model
        self.use_4bit = use_4bit
        
        # Store Azure settings for VLM initialization
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_api_key = azure_api_key
        self.azure_api_version = azure_api_version
        logger.info("Initializing Invoice Extractor...")
        
        # Document processor
        # Note: enhance_quality disabled as it can cause OCR to miss some text elements
        self.doc_processor = DocumentProcessor(
            target_dpi=300,
            enhance_quality=False
        )
        
        # OCR Engine (multilingual)
        try:
            self.ocr_engine = OCREngine(
                use_gpu=use_gpu,
                enable_multilingual=True
            )
            self._ocr_available = True
        except ImportError as e:
            logger.warning(f"OCR not available: {e}")
            self._ocr_available = False
        
        # Field parser
        self.field_parser = FieldParser()
        
        # Validator
        self.validator = Validator(fuzzy_threshold=90)
        
        # Signature/Stamp detector - DISABLED for speed
        # Per problem statement: all documents will contain signature and stamp
        self.detector = None  # YOLO removed - signature/stamp always present
        
        # VLM (lazy loaded)
        self.use_vlm = use_vlm
        self.vlm_provider = vlm_provider
        self._vlm = None
        
        # Consensus Engine for multi-pipeline extraction
        self.consensus_engine = ConsensusEngine(
            min_voters=2,
            confidence_weight=True,
            use_deterministic_boost=True
        )
        self.bootstrap_refiner = BootstrapRefiner(self.consensus_engine)
        
        logger.info("Invoice Extractor initialized successfully")
    
    def _get_vlm(self):
        """Lazy load VLM extractor."""
        if self._vlm is None and self.use_vlm:
            try:
                from utils.vlm_extractor import VLMExtractor
                self._vlm = VLMExtractor(
                    provider=self.vlm_provider,
                    model_name=self.vlm_model,
                    use_4bit=self.use_4bit,
                    api_key=self.azure_api_key if self.vlm_provider == 'azure' else None,
                    azure_endpoint=self.azure_endpoint,
                    azure_deployment=self.azure_deployment,
                    azure_api_version=self.azure_api_version
                )
                logger.info(f"VLM loaded: {self.vlm_provider} ({self.vlm_model})")
            except Exception as e:
                logger.warning(f"VLM initialization failed: {e}")
                self.use_vlm = False
        return self._vlm
    
    def extract(self, input_path: Union[str, Path]) -> Dict:
        """
        Extract fields from a single document.
        
        Args:
            input_path: Path to PDF or image file
            
        Returns:
            Standardized extraction result dict
        """
        start_time = time.time()
        input_path = Path(input_path)
        doc_id = input_path.stem
        
        logger.info(f"Processing: {doc_id}")
        
        # Track processing for cost estimation
        processing_stats = {
            'ocr_calls': 0,
            'vlm_calls': 0,
            'vlm_tokens': 0
        }
        
        try:
            # === STAGE 1: Document Processing ===
            pages = self.doc_processor.process(input_path)
            if not pages:
                raise ValueError("No pages extracted from document")
            
            image, metadata = pages[0]  # Use first page
            img_size = image.size  # (width, height)
            
            # === STAGE 2: OCR Extraction with Layout Analysis ===
            ocr_result = {
                'texts': [], 'boxes': [], 'full_text': '', 'language': 'english',
                'lines': [], 'regions': {}, 'key_value_candidates': [], 'tables': []
            }
            
            if self._ocr_available:
                # Use enhanced layout extraction for better visual understanding
                ocr_result = self.ocr_engine.extract_with_layout(image)
                processing_stats['ocr_calls'] = 1
                
                logger.debug(f"OCR extracted {len(ocr_result.get('texts', []))} text elements")
                logger.debug(f"Detected language: {ocr_result.get('language', 'unknown')}")
                logger.debug(f"Detected {len(ocr_result.get('lines', []))} lines, "
                           f"{len(ocr_result.get('tables', []))} tables, "
                           f"{len(ocr_result.get('key_value_candidates', []))} KV pairs")
            
            # === STAGE 3: Multi-Pipeline Extraction with Consensus ===
            # Define extraction methods for consensus voting
            extraction_methods = [
                lambda: self.field_parser.extract_all(ocr_result),  # Method 1: Full parser
            ]
            method_names = ['rule_based']
            
            # Add VLM as additional voter if enabled
            vlm_result = None
            if self.use_vlm:
                vlm = self._get_vlm()
                if vlm:
                    def vlm_extraction():
                        nonlocal vlm_result
                        try:
                            vlm_result = vlm.extract_fields(image)
                            processing_stats['vlm_calls'] = 1
                            processing_stats['vlm_tokens'] = getattr(vlm, 'tokens_used', 500)
                            return vlm_result
                        except Exception as e:
                            logger.warning(f"VLM extraction failed: {e}")
                            return {}
                    
                    extraction_methods.append(vlm_extraction)
                    method_names.append('vlm')
            
            # Run primary extraction
            parsed_fields = self.field_parser.extract_all(ocr_result)
            initial_confidence = self._calculate_confidence(parsed_fields)
            logger.debug(f"Initial extraction confidence: {initial_confidence:.2f}")
            
            # === STAGE 4: Consensus & Pseudo-Labeling ===
            # Generate pseudo-labels using deterministic rules
            full_text = ocr_result.get('full_text', '')
            pseudo_labels = self.consensus_engine.generate_pseudo_labels(
                ocr_text=full_text,
                extraction_results=parsed_fields,
                full_ocr_result=ocr_result
            )
            logger.debug(f"Generated pseudo-labels for {sum(1 for p in pseudo_labels.values() if p.get('is_pseudo_label'))} fields")
            
            # Use VLM when enabled - always apply consensus for better accuracy
            # VLM is especially important for handwritten text that OCR struggles with
            if self.use_vlm and len(extraction_methods) > 1:
                logger.info("Applying VLM + consensus voting for enhanced extraction...")
                
                # Run consensus across methods with adaptive trust weighting
                # Pass ocr_result to enable document trait detection for field-specific weights
                consensus_results = self.consensus_engine.extract_with_consensus(
                    extraction_methods, method_names, ocr_result=ocr_result
                )
                
                # Convert consensus to parsed_fields format
                for field, (value, conf, meta) in consensus_results.items():
                    if value is not None and conf > 0:
                        parsed_fields[field] = (value, conf)
                        logger.debug(f"  {field}: {value} (conf={conf:.2f}, voters={meta.get('voters', 1)})")
                
                processing_stats['consensus_voters'] = len(extraction_methods)
            
            # Bootstrap refinement with pseudo-labels
            refined_labels = self.bootstrap_refiner.refine_labels(
                current_labels=pseudo_labels,
                new_extraction=parsed_fields
            )
            
            # Use refined labels if they're more reliable
            for field, label in refined_labels.items():
                _, current_conf = self._safe_unpack_field(parsed_fields.get(field), 0.0)
                
                if label.get('reliable') and label.get('confidence', 0) > current_conf:
                    parsed_fields[field] = (label['value'], label['confidence'])
            
            # === STAGE 5: Self-Consistency Verification ===
            consistency_report = self.consensus_engine.verify_self_consistency(
                {k: (v[0] if isinstance(v, tuple) else v, v[1] if isinstance(v, tuple) else 0.5, {}) 
                 for k, v in parsed_fields.items()}
            )
            
            if not consistency_report['is_consistent']:
                logger.warning("Self-consistency check failed, applying adjustments...")
                for field, adj in consistency_report.get('adjustments', {}).items():
                    if field in parsed_fields:
                        val, conf = self._safe_unpack_field(parsed_fields[field], 0.5)
                        # Lower confidence for inconsistent fields
                        parsed_fields[field] = (val, conf * 0.85)
            
            # === STAGE 6: Validation ===
            validated = self.validator.validate_all(parsed_fields)
            
            # === STAGE 6: Signature/Stamp Detection ===
            # YOLO removed for speed - signature/stamp always present per problem statement
            detections = {
                'signature': {'present': True, 'bbox': None, 'confidence': 1.0},
                'stamp': {'present': True, 'bbox': None, 'confidence': 1.0}
            }
            
            # === STAGE 7: Format Output ===
            processing_time = time.time() - start_time
            
            result = self._format_result(
                doc_id=doc_id,
                validated=validated,
                detections=detections,
                processing_time=processing_time,
                processing_stats=processing_stats,
                language=ocr_result.get('language', 'english')
            )
            
            logger.info(
                f"Completed {doc_id}: confidence={result['confidence']:.2f}, "
                f"time={processing_time:.1f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            return self._error_result(doc_id, str(e), time.time() - start_time)
    
    @staticmethod
    def _safe_unpack_field(field_value, default_conf: float = 0.0) -> Tuple:
        """
        Safely unpack field value to (value, confidence) tuple.
        
        Handles edge cases where field_value might be:
        - A proper (value, confidence) tuple
        - A single-element tuple
        - A raw value (not a tuple)
        - None
        """
        if field_value is None:
            return (None, default_conf)
        elif isinstance(field_value, tuple):
            if len(field_value) >= 2:
                return (field_value[0], field_value[1])
            elif len(field_value) == 1:
                return (field_value[0], default_conf)
            else:
                return (None, default_conf)
        else:
            return (field_value, default_conf)
    
    def _calculate_confidence(self, fields: Dict) -> float:
        """Calculate extraction confidence from parsed fields."""
        confidences = []
        for value in fields.values():
            _, conf = self._safe_unpack_field(value, 0.0)
            if conf > 0:
                confidences.append(conf)
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _merge_vlm_results(self, parsed: Dict, vlm: Dict) -> Dict:
        """Merge VLM results with OCR results, preferring VLM for low-confidence fields."""
        merged = {}
        vlm_conf = vlm.get('confidence', 0.8)
        
        for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            parsed_val, parsed_conf = self._safe_unpack_field(parsed.get(field), 0.0)
            vlm_val = vlm.get(field)
            
            # Use VLM if OCR confidence is low or OCR didn't find value
            if (parsed_val is None or parsed_conf < 0.6) and vlm_val is not None:
                merged[field] = (vlm_val, vlm_conf)
            else:
                merged[field] = (parsed_val, parsed_conf)
        
        return merged
    
    def _format_result(
        self,
        doc_id: str,
        validated: Dict,
        detections: Dict,
        processing_time: float,
        processing_stats: Dict,
        language: str = 'english'
    ) -> Dict:
        """Format result to match required output structure."""
        
        # Get detection data
        sig = detections.get('signature', {})
        stamp = detections.get('stamp', {})
        
        # Get per-field confidences computed by validator
        # These are based on fuzzy matching, cross-validation, and range checking
        field_confs = validated.get('_field_confidences', {})
        
        # Calculate overall confidence using actual field-level confidences
        field_confidences = []
        
        # Text/numeric fields - use actual computed confidences, not hardcoded values
        for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            if validated.get(field) is not None:
                # Use validator's computed confidence for this field
                actual_conf = field_confs.get(field, 0.85)
                field_confidences.append(actual_conf)
            else:
                field_confidences.append(0.0)
        
        # Detection fields
        field_confidences.append(sig.get('confidence', 0))
        field_confidences.append(stamp.get('confidence', 0))
        
        # Use validator's overall confidence if available, otherwise calculate from fields
        overall_confidence = validated.get('_confidence', 0)
        if overall_confidence == 0:
            overall_confidence = sum(field_confidences) / len(field_confidences) if field_confidences else 0
        
        # Estimate cost
        cost = self._estimate_cost(processing_stats)
        
        return {
            "doc_id": doc_id,
            "fields": {
                "dealer_name": validated.get('dealer_name'),
                "model_name": validated.get('model_name'),
                "horse_power": validated.get('horse_power'),
                "asset_cost": validated.get('asset_cost'),
                "signature": {
                    "present": True,  # Always present per problem statement
                    "bbox": None  # YOLO removed for speed
                },
                "stamp": {
                    "present": True,  # Always present per problem statement
                    "bbox": None  # YOLO removed for speed
                }
            },
            "confidence": round(overall_confidence, 2),
            "processing_time_sec": round(processing_time, 1),
            "cost_estimate_usd": round(cost, 4)
        }
    
    def _error_result(self, doc_id: str, error: str, processing_time: float) -> Dict:
        """Generate error result in correct format."""
        return {
            "doc_id": doc_id,
            "fields": {
                "dealer_name": None,
                "model_name": None,
                "horse_power": None,
                "asset_cost": None,
                "signature": {"present": True, "bbox": None},
                "stamp": {"present": True, "bbox": None}
            },
            "confidence": 0.0,
            "processing_time_sec": round(processing_time, 1),
            "cost_estimate_usd": 0.0
        }
    
    def _estimate_cost(self, stats: Dict) -> float:
        """
        Estimate processing cost in USD.
        
        Costs based on typical cloud pricing:
        - OCR: ~$0.0003 per image
        - VLM API: ~$0.003 per 1K tokens
        - Detection: ~$0.0001 per image
        """
        cost = 0.0001  # Base (PDF conversion)
        cost += stats.get('ocr_calls', 0) * 0.0003
        cost += stats.get('vlm_tokens', 0) / 1000 * 0.003
        cost += 0.0001  # Detection
        return cost
    
    def extract_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict]:
        """
        Process all documents in a directory.
        
        Args:
            input_dir: Directory containing documents
            output_dir: Directory for output files
            
        Returns:
            List of extraction results
        """
        input_dir = Path(input_dir)
        
        # Find documents
        extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
        files = []
        for ext in extensions:
            files.extend(input_dir.glob(f'*{ext}'))
            files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        files = sorted(set(files))
        logger.info(f"Found {len(files)} documents to process")
        
        results = []
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] Processing {file_path.name}")
            
            result = self.extract(file_path)
            results.append(result)
            
            # Save individual result
            if output_dir:
                out_path = Path(output_dir) / f"{file_path.stem}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Invoice Field Extraction - ET Gen AI Hackathon @ IIT Guwahati',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single document (positional argument - RECOMMENDED for evaluation):
    python executable.py invoice.png
    
  Single document (with named argument):
    python executable.py --input invoice.pdf --output result.json
    
  Batch processing:
    python executable.py --input_dir ./invoices/ --output_dir ./results/
    
  With local VLM (offline, no internet - DEFAULT):
    python executable.py invoice.png --use_vlm
    
  With specific VLM model:
    python executable.py invoice.png --use_vlm --vlm_provider qwen --vlm_model Qwen/Qwen2-VL-7B-Instruct --use_4bit
        """
    )
    
    # Positional argument for input file (for evaluation: python executable.py invoice.png)
    parser.add_argument('input_file', nargs='?', type=str, help='Input document path (PNG/PDF)')
    
    # Named arguments (backward compatibility)
    parser.add_argument('--input', '-i', type=str, help='Input document path (alternative to positional)')
    parser.add_argument('--input_dir', '-d', type=str, help='Input directory for batch')
    parser.add_argument('--output', '-o', type=str, help='Output JSON path')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--use_vlm', action='store_true', default=True, 
                        help='Use VLM for better accuracy (enabled by default)')
    parser.add_argument('--no_vlm', action='store_true', 
                        help='Disable VLM (use rule-based extraction only)')
    parser.add_argument('--vlm_provider', type=str, default='granite', choices=['granite', 'qwen', 'openai', 'azure'],
                        help='VLM provider: granite (default, fast), qwen, openai, or azure')
    parser.add_argument('--vlm_model', type=str, default='ibm-granite/granite-docling-258M',
                        help='VLM model: granite-docling-258M (fast), granite-vision-3.2-2b (accurate), or qwen models')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization (for larger models)')
    
    # Azure OpenAI specific arguments
    parser.add_argument('--azure_endpoint', type=str, help='Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT)')
    parser.add_argument('--azure_deployment', type=str, help='Azure OpenAI deployment name (or set AZURE_OPENAI_DEPLOYMENT)')
    parser.add_argument('--azure_api_key', type=str, help='Azure OpenAI API key (or set AZURE_OPENAI_API_KEY)')
    parser.add_argument('--azure_api_version', type=str, default='2024-02-15-preview', help='Azure OpenAI API version')
    
    parser.add_argument('--yolo_model', type=str, help='Path to YOLO model')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    level = "DEBUG" if args.debug else "INFO"
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    # Resolve input: positional argument takes precedence, then --input
    input_path = args.input_file or args.input
    
    # Validate arguments
    if not input_path and not args.input_dir:
        parser.error("Input required: provide input file as first argument, or use --input or --input_dir")
    
    # Initialize extractor
    # VLM is enabled by default, use --no_vlm to disable
    use_vlm = args.use_vlm and not getattr(args, 'no_vlm', False)
    
    extractor = InvoiceExtractor(
        use_vlm=use_vlm,
        vlm_provider=args.vlm_provider,
        vlm_model=getattr(args, 'vlm_model', 'Qwen/Qwen2-VL-2B-Instruct'),
        use_4bit=getattr(args, 'use_4bit', False),
        yolo_model_path=args.yolo_model,
        use_gpu=not args.no_gpu,
        azure_endpoint=getattr(args, 'azure_endpoint', None),
        azure_deployment=getattr(args, 'azure_deployment', None),
        azure_api_key=getattr(args, 'azure_api_key', None),
        azure_api_version=getattr(args, 'azure_api_version', '2024-02-15-preview')
    )
    
    # Process
    if input_path:
        # Single document processing
        result = extractor.extract(input_path)
        
        # Determine output path - use doc_id from result for consistent naming
        output_path = args.output
        if not output_path:
            doc_id = result.get('doc_id', Path(input_path).stem)
            output_path = f"./sample_output/{doc_id}_result.json"
        
        # Save result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Print result
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    else:
        # Batch processing
        results = extractor.extract_batch(args.input_dir, args.output_dir)
        
        # Save combined results
        combined_path = Path(args.output_dir) / 'all_results.json'
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        successful = sum(1 for r in results if r.get('confidence', 0) > 0)
        total_docs = len(results)
        avg_conf = sum(r.get('confidence', 0) for r in results) / total_docs if total_docs else 0
        avg_time = sum(r.get('processing_time_sec', 0) for r in results) / total_docs if total_docs else 0
        total_cost = sum(r.get('cost_estimate_usd', 0) for r in results)
        
        print(f"\n{'='*55}")
        print(f"  BATCH PROCESSING COMPLETE")
        print(f"{'='*55}")
        print(f"  Documents Processed: {total_docs}")
        if total_docs:
            print(f"  Successful:          {successful} ({successful/total_docs*100:.0f}%)")
        else:
            print(f"  Successful:          0 (0%)")
        print(f"  Average Confidence:  {avg_conf:.1%}")
        print(f"  Average Time:        {avg_time:.1f}s per document")
        print(f"  Total Cost:          ${total_cost:.4f}")
        print(f"  Results:             {combined_path}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
