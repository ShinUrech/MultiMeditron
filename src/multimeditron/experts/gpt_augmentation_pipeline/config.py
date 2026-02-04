"""
config.py

Central configuration for GPT-based dataset augmentation pipelines.

This file is intentionally lightweight and fully environment-driven to ensure:
- reproducibility
- safety for public repositories
- easy reuse across datasets and experiments
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# OpenAI / API configuration
# ---------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Please export your OpenAI API key before running the pipeline."
    )

# Model used for generation
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ---------------------------------------------------------------------
# Pricing assumptions (per 1M tokens)
# Used ONLY for cost estimation and reporting
# ---------------------------------------------------------------------

PRICING_CURRENCY = os.getenv("PRICING_CURRENCY", "CHF")

PRICE_PER_MILLION_INPUT = float(
    os.getenv("PRICE_PER_MILLION_INPUT", "2.50")
)
PRICE_PER_MILLION_OUTPUT = float(
    os.getenv("PRICE_PER_MILLION_OUTPUT", "7.50")
)

# ---------------------------------------------------------------------
# Experiment / run identification
# ---------------------------------------------------------------------

# Change this to separate experiments cleanly
RUN_NAME = os.getenv(
    "RUN_NAME",
    "fitzpatrick_skin_gpt4o_v1"
)

# ---------------------------------------------------------------------
# Dataset paths (dataset-agnostic)
# ---------------------------------------------------------------------

# Root directory of the processed dataset (contains the JSONL manifest)
DATASET_ROOT = Path(
    os.getenv(
        "DATASET_ROOT",
        "/mloscratch/users/turan/datasets/Fitzpatrick"
    )
)

# Root directory for images (defaults to DATASET_ROOT)
DATASET_IMG_ROOT = Path(
    os.getenv(
        "DATASET_IMG_ROOT",
        str(DATASET_ROOT)
    )
)

# ---------------------------------------------------------------------
# Batch + output directories (scoped by RUN_NAME)
# ---------------------------------------------------------------------

BASE_RUN_DIR = Path(
    os.getenv(
        "BASE_RUN_DIR",
        "/mloscratch/users/turan/gpt4_dataset_generation"
    )
) / RUN_NAME

BATCHES_DIR = Path(
    os.getenv(
        "BATCHES_DIR",
        str(BASE_RUN_DIR / "batches")
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        str(BASE_RUN_DIR / "outputs")
    )
)

# Ensure directories exist
BATCHES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Estimation settings
# ---------------------------------------------------------------------

# Number of samples used when estimating total cost
N_ITER_ESTIMATE = int(
    os.getenv("N_ITER_ESTIMATE", "5")
)
