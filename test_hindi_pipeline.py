import json
import os
import sys
# Redirect loguru to stderr for cleaner output
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from utils.field_parser import FieldParser
from utils.validator import Validator
from utils.consensus_engine import ConsensusEngine

def test_hindi_pipeline():
    # 1. Mock OCR Result for a Hindi Sonalika Invoice
    ocr_result = {
        'full_text': """
सोनालीका ट्रैक्टर्स
अधिकृत विक्रेता: सोनालीका इंटरनेशनल
बिल प्रति: श्री राम लाल
दिनांक: १०/०५/२०२४
विवरण: सोनालीका डीआई ७४५ III
अश्वशक्ति: ५० एचपी
कुल मूल्य: ₹७,५०,०००
""",
        'texts': [
            "सोनालीका ट्रैक्टर्स",
            "अधिकृत विक्रेता: सोनालीका इंटरनेशनल",
            "बिल प्रति: श्री राम लाल",
            "दिनांक: १०/०५/२०२४",
            "विवरण: सोनालीका डीआई ७४५ III",
            "अश्वशक्ति: ५० एचपी",
            "कुल मूल्य: ₹७,५०,०००"
        ],
        'language': 'hindi',
        'confidences': [0.9] * 7
    }

    print("\n[STEP 1] Field Parser Extraction...")
    parser = FieldParser()
    parsed_fields = parser.extract_all(ocr_result)
    for k, v in parsed_fields.items():
        print(f"  {k}: {v}")

    print("\n[STEP 2] Consensus Engine...")
    vlm_result = {
        'dealer_name': 'Sonalika International',
        'model_name': 'Sonalika DI 745',
        'horse_power': 50,
        'asset_cost': 750000,
        'confidence': 0.85
    }
    
    engine = ConsensusEngine()
    consensus_results = engine.extract_with_consensus(
        [lambda: parsed_fields, lambda: vlm_result], 
        ['rule_based', 'vlm'],
        ocr_result=ocr_result
    )
    
    final_parsed = {}
    for field, (val, conf, meta) in consensus_results.items():
        print(f"  {field}: {val} (conf={conf}, voters={meta.get('voters')})")
        final_parsed[field] = (val, conf)

    print("\n[STEP 3] Validator...")
    validator = Validator()
    # Check normalization directly
    raw_model = final_parsed.get('model_name', (None, 0))[0]
    norm_model = validator.normalize_multilingual_text(raw_model)
    print(f"  Raw Model: {raw_model} -> Norm Model: {norm_model}")
    
    validated = validator.validate_all(final_parsed)
    print("\n[FINAL OUTPUT]:")
    print(json.dumps({k: v for k, v in validated.items() if not k.startswith('_')}, indent=2, ensure_ascii=False))
    print(f"\nOverall Confidence: {validated.get('_confidence')}")

if __name__ == "__main__":
    test_hindi_pipeline()
