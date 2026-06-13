"""
Pre-cache CLIP model weights for Docker builds.

Downloads and loads the CLIP model so that weights are cached
in the HuggingFace / torch cache directories during the Docker
build step, avoiding runtime downloads.
"""

import sys
sys.path.append('.')

from deep_pavements.models import load_clip_model
from deep_pavements.constants import DEVICE

# Load CLIP model — this triggers the download and caching
model, preprocess, is_real = load_clip_model(DEVICE)

if is_real:
    print("✓ CLIP model pre-cached successfully")
else:
    print("⚠ CLIP model could not be loaded, using mock")
