"""
Deep Pavements Lite — Automated pavement surface classification from street-level imagery.

This package provides the core functionality for analyzing pavement surfaces in
street-level imagery using computer vision and machine learning techniques.

Main entry point:
    from modules import process_images
"""

from modules.pipeline import process_images
from modules.constants import (
    DEVICE,
    default_surfaces,
    pathway_categories,
    clip_model_path,
    data_path,
)

__version__ = "1.0.0"

__all__ = [
    "process_images",
    "DEVICE",
    "default_surfaces",
    "pathway_categories",
    "clip_model_path",
    "data_path",
]
