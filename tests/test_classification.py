"""Tests for deep_pavements.classification – CLIP‑based surface classification."""

import numpy as np
import pytest

from modules.classification import classify_surface_type
from modules.constants import default_surfaces


# ── classify_surface_type ─────────────────────────────────────────────────


def test_classify_surface_type_returns_dict(mock_image, mock_clip_model, mock_tokenize, sample_polygon):
    """Result must be a dict with 'surface' and 'confidence' keys."""
    model, preprocess = mock_clip_model

    result = classify_surface_type(mock_image, sample_polygon, model, preprocess, "cpu")

    assert isinstance(result, dict)
    assert "surface" in result
    assert "confidence" in result


def test_classify_surface_type_valid_surface(mock_image, mock_clip_model, mock_tokenize, sample_polygon):
    """The predicted surface must be one of the canonical surface types."""
    model, preprocess = mock_clip_model

    result = classify_surface_type(mock_image, sample_polygon, model, preprocess, "cpu")

    assert result["surface"] in default_surfaces


def test_classify_surface_type_confidence_range(mock_image, mock_clip_model, mock_tokenize, sample_polygon):
    """Confidence should be a float in [0.0, 1.0]."""
    model, preprocess = mock_clip_model

    result = classify_surface_type(mock_image, sample_polygon, model, preprocess, "cpu")

    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_surface_type_small_region(mock_image, mock_clip_model, mock_tokenize):
    """A polygon smaller than 32×32 pixels must still produce a valid result
    (the implementation upsamples tiny crops to 224×224)."""
    model, preprocess = mock_clip_model
    # 20×20 rectangle
    small_polygon = np.array([10, 10, 30, 10, 30, 30, 10, 30], dtype=np.float64)

    result = classify_surface_type(mock_image, small_polygon, model, preprocess, "cpu")

    assert "surface" in result
    assert "confidence" in result


def test_classify_surface_type_invalid_polygon(mock_image, mock_clip_model, mock_tokenize):
    """An empty polygon should gracefully return surface='unknown', confidence=0.0."""
    model, preprocess = mock_clip_model
    empty_polygon = np.array([], dtype=np.float64)

    result = classify_surface_type(mock_image, empty_polygon, model, preprocess, "cpu")

    assert result["surface"] == "unknown"
    assert result["confidence"] == 0.0
