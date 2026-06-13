#!/usr/bin/env python3
"""
Download the fine-tuned CLIP model from HuggingFace.

This is a convenience script that delegates to the shared model
download function in deep_pavements.models.
"""

from deep_pavements.models import download_finetuned_model
from deep_pavements.constants import clip_model_path

if __name__ == "__main__":
    download_finetuned_model(clip_model_path)