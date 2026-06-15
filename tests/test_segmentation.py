"""Tests for deep_pavements.segmentation – heuristic segmentation, overlay,
and Cityscapes colour encoding."""

import numpy as np
import pytest
from PIL import Image

from modules.segmentation import (
    create_heuristic_segmentation,
    create_segmentation_overlay,
    get_cityscapes_color_encoding,
)


# ── create_heuristic_segmentation ─────────────────────────────────────────


def test_heuristic_segmentation_shape(mock_image):
    """Output mask must have the same (H, W) as the input image."""
    mask = create_heuristic_segmentation(mock_image)

    width, height = mock_image.size
    assert mask.shape == (height, width)


def test_heuristic_segmentation_valid_classes(mock_image):
    """Mask values must be drawn exclusively from {0, 1, 13, 255}."""
    mask = create_heuristic_segmentation(mock_image)

    unique_values = set(np.unique(mask))
    allowed = {0, 1, 13, 255}
    assert unique_values.issubset(allowed), (
        f"Unexpected class IDs in mask: {unique_values - allowed}"
    )


def test_heuristic_segmentation_has_road(mock_image):
    """The mask must contain at least some road pixels (class 0)."""
    mask = create_heuristic_segmentation(mock_image)

    assert np.any(mask == 0), "No road pixels (class 0) detected in the mask"


# ── create_segmentation_overlay ───────────────────────────────────────────


def test_segmentation_overlay_returns_image(mock_image, sample_segmentation_mask):
    """Overlay output must be a PIL Image in RGB mode."""
    mapping = {
        "roads": [0],
        "sidewalks": [1],
        "car": [13],
    }
    overlay = create_segmentation_overlay(mock_image, sample_segmentation_mask, mapping)

    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"


def test_segmentation_overlay_size_matches(mock_image, sample_segmentation_mask):
    """Overlay must be the same size as the original image."""
    mapping = {
        "roads": [0],
        "sidewalks": [1],
        "car": [13],
    }
    overlay = create_segmentation_overlay(mock_image, sample_segmentation_mask, mapping)

    assert overlay.size == mock_image.size


# ── get_cityscapes_color_encoding ─────────────────────────────────────────


def test_cityscapes_encoding_has_classes():
    """Encoding dict must have a 'classes' key containing entries for
    class IDs '0', '1', and '13'."""
    encoding = get_cityscapes_color_encoding()

    assert "classes" in encoding
    for cid in ("0", "1", "13"):
        assert cid in encoding["classes"], f"Missing class ID '{cid}' in encoding"
