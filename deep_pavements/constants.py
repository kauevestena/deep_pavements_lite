"""
Deep Pavements Lite — Configuration constants and project settings.

Centralizes all configuration values used across the package including:
- Device selection (CUDA/CPU)
- File paths and extensions
- Surface type labels
- Pathway categories for segmentation
- Debug mode settings
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import torch

# ── Device Configuration ──────────────────────────────────────────────────────
# Use CUDA if available, otherwise fall back to CPU
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"

# ── Path Configuration ────────────────────────────────────────────────────────
# Project root is the parent of this package directory
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

params_path: Final[str] = "params.json"
data_path: Final[str] = "data"

# ── File Extensions ───────────────────────────────────────────────────────────
ext_in: Final[str] = ".jpg"
ext_out: Final[str] = ".png"

# ── Pathway Categories ───────────────────────────────────────────────────────
# Categories used in semantic segmentation with Cityscapes class mapping
pathway_categories: Final[list[str]] = ["roads", "sidewalks", "car"]

# Cityscapes class ID mapping for each pathway category
PATHWAY_CLASS_MAPPING: Final[dict[str, list[int]]] = {
    "roads": [0],       # Cityscapes 'road' class
    "sidewalks": [1],   # Cityscapes 'sidewalk' class
    "car": [13],        # Cityscapes 'car' class
}

# ── Surface Type Labels ──────────────────────────────────────────────────────
# Surface material types that can be classified by the CLIP model
default_surfaces: Final[list[str]] = [
    "asphalt",
    "concrete",
    "concrete_plates",
    "grass",
    "ground",
    "sett",
    "paving_stones",
    "cobblestone",
    "gravel",
    "sand",
    "compacted",
]

# ── Model Configuration ──────────────────────────────────────────────────────
clip_model_path: Final[str] = "deep_pavements_clip_model.pt"

CLIP_ARCHITECTURE: Final[str] = "ViT-B/32"
CLIP_EMBEDDING_DIM: Final[int] = 512

FINETUNED_REPO: Final[str] = (
    "kauevestena/clip-vit-base-patch32-finetuned-surface-materials"
)
FINETUNED_HF_FILENAME: Final[str] = "pytorch_model.bin"
FINETUNED_DIRECT_URL: Final[str] = (
    "https://huggingface.co/kauevestena/"
    "clip-vit-base-patch32-finetuned-surface-materials/"
    "resolve/main/model.pt"
)

ONEFORMER_MODEL_ID: Final[str] = "shi-labs/oneformer_cityscapes_swin_large"

# ── Model Version Info ────────────────────────────────────────────────────────
MODEL_INFO: Final[dict[str, str]] = {
    "clip_architecture": CLIP_ARCHITECTURE,
    "finetuned_repo": FINETUNED_REPO,
    "oneformer_model": ONEFORMER_MODEL_ID,
    "package_version": "1.0.0",
}

# ── Debug Mode Configuration ─────────────────────────────────────────────────
DEBUG_OUTPUT_DIR: Final[str] = "debug_outputs"

# ── Minimum Region Size ──────────────────────────────────────────────────────
MIN_REGION_SIZE: Final[int] = 32  # px — regions smaller than this get upscaled
CLIP_INPUT_SIZE: Final[int] = 224  # CLIP's native resolution
