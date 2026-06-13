"""Tests for deep_pavements.models – mock CLIP model factory."""

import torch

from deep_pavements.models import create_mock_clip_model


# ── create_mock_clip_model ────────────────────────────────────────────────


def test_create_mock_clip_model_returns_tuple():
    """Factory must return a ``(model, preprocess)`` 2‑tuple."""
    result = create_mock_clip_model("cpu")

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_mock_encode_image_shape():
    """``encode_image`` on a (1, 3, 224, 224) tensor should return (1, 512)."""
    model, _ = create_mock_clip_model("cpu")
    dummy_input = torch.randn(1, 3, 224, 224)

    output = model.encode_image(dummy_input)

    assert output.shape == (1, 512)


def test_mock_encode_text_shape():
    """``encode_text`` on a (5, 77) tensor should return (5, 512)."""
    model, _ = create_mock_clip_model("cpu")
    dummy_tokens = torch.randint(0, 1000, (5, 77))

    output = model.encode_text(dummy_tokens)

    assert output.shape == (5, 512)


def test_mock_model_eval():
    """``model.eval()`` must return the model itself (fluent API)."""
    model, _ = create_mock_clip_model("cpu")

    assert model.eval() is model


def test_mock_model_to_device():
    """``model.to('cpu')`` must return the model itself (fluent API)."""
    model, _ = create_mock_clip_model("cpu")

    assert model.to("cpu") is model
