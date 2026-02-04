"""
utils.py

Utility functions for loading multimodal datasets (text + images)
and preparing inputs for OpenAI batch-based augmentation.

This module is dataset-agnostic and expects a JSONL manifest with:
{
  "text": "...",
  "modalities": [
      {"type": "image", "value": "relative/path/to/image.jpg"}
  ]
}
"""

import json
import base64
from typing import Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm

import config

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATASET_ROOT = Path(config.DATASET_ROOT)
IMAGE_ROOT = Path(config.DATASET_IMG_ROOT)

MANIFEST_NAME = os.getenv(
    "MANIFEST_NAME",
    "fitzpatrick_black_skin_manifest.jsonl"
)
MANIFEST_PATH = DATASET_ROOT / MANIFEST_NAME


# ---------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------

def load_data_raw() -> List[dict]:
    """
    Load the raw JSONL manifest as a list of dicts.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    records: List[dict] = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {MANIFEST_PATH}"
                ) from e

    if not records:
        raise RuntimeError("Manifest loaded but contains no records.")

    return records


def encode_image_to_base64(path: Path) -> str:
    """
    Read an image from disk and return base64-encoded string.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image_path(value: str) -> Path:
    """
    Resolve image paths safely, handling leading slashes.
    """
    rel = value.lstrip("/")
    return IMAGE_ROOT / rel


def load_data(
    nb_samples: Optional[int] = None,
    max_images_per_sample: int = 1,
) -> Tuple[List[Tuple[str, Tuple[str, ...]]], List[str]]:
    """
    Load dataset examples and encode images.

    Returns:
      examples: List of (text, (img_b64, ...))
      paths:    List of absolute image paths used
    """
    raw_records = load_data_raw()
    records = raw_records[:nb_samples] if nb_samples else raw_records

    examples: List[Tuple[str, Tuple[str, ...]]] = []
    used_paths: List[str] = []

    for rec in tqdm(records, desc="Loading dataset"):
        text = str(rec.get("text", "")).strip()
        if not text:
            continue

        image_paths: List[Path] = []
        for m in rec.get("modalities", []):
            if m.get("type") == "image" and m.get("value"):
                image_paths.append(resolve_image_path(m["value"]))
            if len(image_paths) >= max_images_per_sample:
                break

        if not image_paths:
            continue

        try:
            encoded_images = tuple(
                encode_image_to_base64(p) for p in image_paths
            )
        except Exception as e:
            # Skip corrupted or unreadable images
            print(f"[WARN] Skipping sample due to image error: {e}")
            continue

        examples.append((text, encoded_images))
        used_paths.extend(str(p) for p in image_paths)

    if not examples:
        raise RuntimeError("No valid examples loaded.")

    return examples, used_paths
