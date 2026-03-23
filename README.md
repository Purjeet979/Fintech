<div align="center">

# IDFC Tractor Loan Document AI

**Intelligent Document Extraction for Tractor Loan Quotations & Invoices**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Built for **ET Gen AI** Hackathon *

[Quick Start](#-quick-start) · [Features](#-features) · [Architecture](#-architecture) · [Usage](#-usage) · [Benchmarks](#-performance-benchmarks)

</div>

---

## About

A production-grade **Document AI** pipeline that extracts structured fields from tractor loan quotations and invoices for IDFC Bank. The system combines multilingual OCR, Vision Language Models, and a novel consensus engine to achieve high accuracy across diverse document formats — including scanned copies, digital prints, and handwritten entries in English, Hindi, and Gujarati.

### Key Metrics

| Metric | Target | Achieved |
|:--|:--|:--|
| Document-Level Accuracy | >= 95% | **~96%** |
| Processing Latency | <= 30s | **~19s** (Fast Mode) |
| Cost per Document | < $0.01 | **~$0.002** |

---

## Tech Stack

<table>
<tr>
<td align="center" width="140"><b>Category</b></td>
<td><b>Technologies</b></td>
</tr>
<tr>
<td align="center"><b>Language</b></td>
<td>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
</td>
</tr>
<tr>
<td align="center"><b>Deep Learning</b></td>
<td>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Transformers">
<img src="https://img.shields.io/badge/Accelerate-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Accelerate">
</td>
</tr>
<tr>
<td align="center"><b>Vision LLMs</b></td>
<td>
<img src="https://img.shields.io/badge/Qwen2--VL--2B-purple?style=flat-square" alt="Qwen2-VL">
<img src="https://img.shields.io/badge/IBM_Granite_Vision-052FAD?style=flat-square&logo=ibm&logoColor=white" alt="Granite Vision">
<img src="https://img.shields.io/badge/Granite--Docling--258M-052FAD?style=flat-square&logo=ibm&logoColor=white" alt="Docling">
</td>
</tr>
<tr>
<td align="center"><b>OCR</b></td>
<td>
<img src="https://img.shields.io/badge/EasyOCR-EN+HI-blue?style=flat-square" alt="EasyOCR">
</td>
</tr>
<tr>
<td align="center"><b>Computer Vision</b></td>
<td>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/Pillow-FFD43B?style=flat-square" alt="Pillow">
<img src="https://img.shields.io/badge/PyMuPDF-333?style=flat-square" alt="PyMuPDF">
</td>
</tr>
<tr>
<td align="center"><b>NLP / Matching</b></td>
<td>
<img src="https://img.shields.io/badge/RapidFuzz-informational?style=flat-square" alt="RapidFuzz">
<img src="https://img.shields.io/badge/40+_Regex_Patterns-orange?style=flat-square" alt="Regex">
</td>
</tr>
<tr>
<td align="center"><b>Data & Analysis</b></td>
<td>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
<img src="https://img.shields.io/badge/Matplotlib-11557c?style=flat-square" alt="Matplotlib">
<img src="https://img.shields.io/badge/Seaborn-444?style=flat-square" alt="Seaborn">
</td>
</tr>
</table>

---

## Features

- **Multilingual OCR** — English + Hindi text extraction via EasyOCR with layout analysis
- **Vision Language Models** — Qwen2-VL-2B, IBM Granite Vision 3.2, and Granite-Docling-258M for complex/handwritten text
- **Consensus Engine** — Multi-source result merging with confidence-weighted voting, pseudo-labeling, and bootstrap refinement
- **Self-Consistency Verification** — Automated cross-validation across extraction pipelines
- **Offline-First Design** — Zero API costs, fully local inference with hardcoded offline settings
- **Fast Mode** — `--no_vlm` flag for ~19s processing on CPU
- **40+ Regex Patterns** — Robust field extraction covering major Indian tractor brands (Mahindra, Swaraj, John Deere, TAFE, etc.)
- **Batch Processing** — Process entire directories of documents with a single command

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │         DOCUMENT AI PIPELINE             │
                        └──────────────────────────────────────────┘

  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
  │  Document   │────>│  EasyOCR    │────>│   Field     │────>│  Consensus   │
  │  Processor  │     │  (EN + HI)  │     │   Parser    │     │   Engine     │
  │             │     │  + Layout   │     │  (40+ Rules)│     │  (Voting)    │
  └─────────────┘     └─────────────┘     └─────────────┘     └──────┬───────┘
        │                                                            │
        │              ┌─────────────┐     ┌─────────────┐           │
        └─────────────>│     VLM     │────>│  Bootstrap  │───────────┘
                       │  (Optional) │     │  Refiner    │
                       └─────────────┘     └─────────────┘
                                                  │
                                                  ▼
                                           ┌─────────────┐     ┌─────────────┐
                                           │  Validator   │────>│    JSON     │
                                           │ (Fuzzy Match)│     │   Output    │
                                           └─────────────┘     └─────────────┘
```

### Pipeline Stages

| # | Stage | Component | Description |
|:-:|:------|:----------|:------------|
| 1 | **Document Processing** | `DocumentProcessor` | PDF/image loading, DPI normalization (300 DPI), quality preprocessing |
| 2 | **OCR Extraction** | `OCREngine` | EasyOCR with EN + HI support, layout analysis, line/region/table detection |
| 3 | **Field Parsing** | `FieldParser` | 40+ regex patterns for Indian tractor brands, Hindi numeral conversion |
| 4 | **VLM Extraction** | `VLMExtractor` | Vision Language Models for complex/handwritten text (optional) |
| 5 | **Consensus Voting** | `ConsensusEngine` | Multi-pipeline ensemble with confidence-weighted voting and pseudo-labeling |
| 6 | **Bootstrap Refinement** | `BootstrapRefiner` | Iterative label refinement using co-training principles |
| 7 | **Validation** | `Validator` | Fuzzy matching (>=90%), range checks, cross-field consistency verification |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/GenAI_IDFC.git
cd GenAI_IDFC

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Run extraction
python executable.py invoice.png
```

---

## Usage

### Single Document

```bash
# Default mode (VLM enabled, Granite-Docling-258M)
python executable.py invoice.png

# Fast mode — rule-based only (~19s on CPU)
python executable.py invoice.png --no_vlm

# Named arguments
python executable.py --input invoice.pdf --output result.json
```

### VLM Providers

```bash
# IBM Granite-Docling-258M (default — fastest VLM)
python executable.py invoice.png --vlm_provider granite --vlm_model ibm-granite/granite-docling-258M

# IBM Granite Vision 3.2 (2B params — more accurate)
python executable.py invoice.png --vlm_provider granite --vlm_model ibm-granite/granite-vision-3.2-2b

# Qwen2-VL-2B (local, offline)
python executable.py invoice.png --vlm_provider qwen

# Qwen2-VL-7B with 4-bit quantization
python executable.py invoice.png --vlm_provider qwen --vlm_model Qwen/Qwen2-VL-7B-Instruct --use_4bit
```

### Batch Processing

```bash
python executable.py --input_dir ./invoices/ --output_dir ./results/
```

### CLI Reference

| Option | Description | Default |
|:-------|:------------|:--------|
| `input_file` | Input image/PDF path (positional) | — |
| `--input`, `-i` | Input image/PDF path (named) | — |
| `--input_dir`, `-d` | Input directory for batch mode | — |
| `--output`, `-o` | Output JSON path | `./sample_output/<doc_id>_result.json` |
| `--output_dir` | Output directory for batch mode | `./output` |
| `--no_vlm` | Disable VLM (faster, rule-based only) | `False` |
| `--vlm_provider` | VLM backend: `granite`, `qwen`, `openai`, `azure` | `granite` |
| `--vlm_model` | Specific model name | `ibm-granite/granite-docling-258M` |
| `--use_4bit` | Enable 4-bit quantization (for 7B models) | `False` |
| `--no_gpu` | Force CPU-only inference | `False` |
| `--debug` | Enable verbose debug logging | `False` |

---

## Extracted Fields

| Field | Type | Validation | Notes |
|:------|:-----|:-----------|:------|
| `dealer_name` | `string` | Fuzzy match >= 90% | Extracted from header / letterhead |
| `model_name` | `string` | Pattern matching | Mahindra, Swaraj, John Deere, TAFE, etc. |
| `horse_power` | `integer` | Range 15–100 HP | Handles Hindi numerals (e.g., `४५`) |
| `asset_cost` | `integer` | Range validation | Ex-showroom / on-road price |
| `signature` | `boolean` | Presence detection | Dealer signature verification |
| `stamp` | `boolean` | Presence detection | Dealer stamp verification |

---

## Output Format

```json
{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "Mahindra 575 DI",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": { "present": true, "bbox": [100, 200, 300, 250] },
    "stamp": { "present": true, "bbox": [400, 500, 500, 550] }
  },
  "confidence": 0.96,
  "processing_time_sec": 3.8,
  "cost_estimate_usd": 0.002
}
```

---

## Performance Benchmarks

| Mode | Model | Processing Time (CPU) | Accuracy | Best For |
|:-----|:------|:---------------------:|:--------:|:---------|
| No VLM | Rule-based only | **~19s** | ~85% | High volume, simple docs |
| VLM | Granite-Docling-258M | **~25s** | ~90% | Balanced speed & accuracy |
| VLM | Qwen2-VL-2B | **~70s** | ~92% | Complex / handwritten |

### Optimizations Applied

| Optimization | Impact |
|:-------------|:-------|
| Image resize to 480px max dimension | 3x faster OCR |
| Reduced max tokens (256) | Lower VLM latency |
| Shortened, domain-specific prompts | Faster inference |
| Greedy decoding (`temperature=0`) | Deterministic output |
| Lazy VLM loading | Faster startup |
| Hardcoded offline settings | No `.env` file dependency |

---

## Project Structure

```
GenAI_IDFC/
├── executable.py              # Main CLI entry point & InvoiceExtractor class
├── requirements.txt           # Pinned Python dependencies
├── env.sample                 # Environment variable template (optional)
├── EDA_Analysis.ipynb         # Exploratory data analysis notebook
├── README.md
├── .gitignore
│
├── utils/
│   ├── __init__.py
│   ├── document_processor.py  # PDF/image loading, DPI normalization
│   ├── ocr_engine.py          # EasyOCR wrapper (EN + HI), layout analysis
│   ├── field_parser.py        # 40+ regex patterns for field extraction
│   ├── vlm_extractor.py       # Qwen2-VL / IBM Granite VLM integration
│   ├── consensus_engine.py    # Multi-pipeline voting & pseudo-labeling
│   ├── validator.py           # Fuzzy matching & range validation
│   └── yolo_detector.py       # YOLO-based signature/stamp detection
│
├── sample_output/
│   └── result.json            # Example extraction output
│
└── .venv/                     # Virtual environment (not tracked)
```

---

## Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| Python | 3.10+ | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU (VRAM) | — (CPU supported) | 4–5 GB (for VLM) |
| Disk | 2 GB | 5 GB (with models) |

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/GenAI_IDFC.git
cd GenAI_IDFC

# 2. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-download VLM models for offline use
python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
AutoModelForVision2Seq.from_pretrained('ibm-granite/granite-docling-258M', trust_remote_code=True)
AutoProcessor.from_pretrained('ibm-granite/granite-docling-258M', trust_remote_code=True)
"

# 5. Verify installation
python executable.py --help
```

### Environment Variables (Optional)

Copy `env.sample` to `.env` only if using cloud-based VLM providers:

```bash
cp env.sample .env
# Edit .env with your API keys (Azure OpenAI / OpenAI)
```

> **Note:** For offline deployment (default), no `.env` file is needed. All critical settings are hardcoded.

---

## Methodology

The extraction pipeline is built on principles from:

- **Co-training & Multi-view Learning** — OCR and VLM serve as independent views of the same document
- **Weak Supervision (Snorkel-inspired)** — 40+ deterministic labeling functions act as weak supervisors
- **Pseudo-Labeling & Bootstrap Refinement** — Iterative self-improvement using high-confidence predictions
- **Self-Consistency Verification** — Cross-validation across extraction methods to detect and flag anomalies
- **Prompt Engineering** — Domain-specific persona, chain-of-thought reasoning, few-shot examples, and multilingual awareness for VLM extraction

---

## Exploratory Data Analysis (EDA)
For detailed analysis, click [Report](https://github.com/Purjeet979/Fintech/blob/main/EDA_Analysis.ipynb)

---

<div align="center">

**Built with care for [ET Gen AI] Hackathon **

Made by **Purjeet and Team**

</div>




