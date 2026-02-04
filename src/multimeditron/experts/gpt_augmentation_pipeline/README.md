# Dataset Augmentation with OpenAI Batch API

This directory contains a **reproducible, batch-based pipeline** for augmenting multimodal expert datasets (e.g. dermatology, ophthalmology) using the OpenAI Chat Completions API.

The pipeline is designed for:
- large-scale dataset augmentation,
- deterministic execution,
- cost estimation and tracking,
- separation of data processing and model interaction.

No datasets or API outputs are committed to the repository.

---

## Overview

The pipeline takes an existing multimodal dataset (JSONL manifest with image paths and text),
builds structured GPT prompts, submits them via the OpenAI **Batch API**, and collects the
generated outputs together with exact token usage and cost.

All steps are CLI-driven and configurable via environment variables.

---

## Directory Structure

```
.
├── config.py              # Central configuration (paths, model, pricing)
├── utils.py               # Dataset loading and image base64 encoding
├── make_batches.py        # Build Batch API request files (JSONL)
├── submit_batches.py      # Submit batch request files to OpenAI
├── collect_all.py         # Retrieve batch outputs, merge results, compute actual cost
├── estimate_price.py      # Cost projection from a small sample
└── README.md
```

---

## Pipeline Steps

### 1. Dataset Preparation (external)

Raw datasets are first converted into a **standard multimediset format**:

```json
{
  "text": "...",
  "modalities": [
    { "type": "image", "value": "relative/path/to/image.jpg" }
  ]
}
```

This preprocessing step is dataset-specific and not included in this pipeline. The pipeline assumes the datasets already exists.

### 2. Build Batch Requests

```bash
python make_batches.py
```

* Loads dataset entries via `utils.py`
* Base64-encodes images
* Builds structured multimodal prompts
* Splits requests into size-limited JSONL files (`part_*.jsonl`)
* Output is written to `BATCHES_DIR`

### 3. (Optional) Estimate Cost

```bash
python submit_batches.py --estimate
python collect_all.py
python estimate_price.py
```

* Runs a small number of samples
* Estimates cost per example
* Projects total cost for the full dataset

No assumptions are made about token counts.

### 4. Submit Full Batches

```bash
python submit_batches.py
```

* Uploads each batch file to OpenAI
* Submits Batch API jobs
* Saves returned batch IDs for later retrieval

### 5. Collect Outputs and Compute Actual Cost

```bash
python collect_all.py
```

* Polls batch jobs until completion
* Downloads raw API responses
* Merges all parts into a single JSONL
* Extracts generated texts
* Computes actual prompt and completion token usage
* Reports exact monetary cost based on configured pricing

---

## Configuration

All configuration is centralized in `config.py` and controlled via environment variables:

* `OPENAI_API_KEY`
* `OPENAI_MODEL`
* dataset paths
* batch/output directories
* pricing per million tokens

---

## Notes

* The pipeline currently uses the Chat Completions API via the Batch endpoint.
* Prompts are intentionally verbose and domain-specific to maximize annotation quality.
* Generated outputs are intended for dataset curation and annotation, not clinical use.

---

## Disclaimer

This pipeline is intended for research and dataset development purposes only.