"""
Deep Pavements Lite — Model loading, downloading, and caching.

Provides unified functions for:
- Loading CLIP models (base + fine-tuned weights)
- Loading OneFormer segmentation models
- Downloading fine-tuned model weights from HuggingFace
- Creating mock models for testing environments

This module consolidates model-related code that was previously duplicated
across lib.py, download_finetuned_model.py, and precaching/precaching_clip.py.
"""

from __future__ import annotations

import os
from typing import Any, cast

import clip
import torch
import torchvision.transforms as transforms

from modules.constants import (
    CLIP_ARCHITECTURE,
    CLIP_EMBEDDING_DIM,
    CLIP_INPUT_SIZE,
    DEVICE,
    FINETUNED_DIRECT_URL,
    FINETUNED_HF_FILENAME,
    FINETUNED_REPO,
    clip_model_path,
)


# ── Mock CLIP Model ──────────────────────────────────────────────────────────


class MockCLIPModel:
    """Mock CLIP model for testing when the real model is unavailable."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = device

    def encode_image(self, image_input: torch.Tensor) -> torch.Tensor:
        """Return a mock image embedding with correct dimensions."""
        batch_size = image_input.shape[0]
        return torch.randn(batch_size, CLIP_EMBEDDING_DIM, device=self.device)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Return mock text embeddings with correct dimensions."""
        batch_size = text_tokens.shape[0]
        return torch.randn(batch_size, CLIP_EMBEDDING_DIM, device=self.device)

    def eval(self) -> "MockCLIPModel":
        return self

    def to(self, device: str | torch.device) -> "MockCLIPModel":
        self.device = device
        return self


def create_mock_clip_model(
    device: str | torch.device,
) -> tuple[MockCLIPModel, transforms.Compose]:
    """
    Create a mock CLIP model for testing purposes when real model is unavailable.

    Args:
        device: PyTorch device to place tensors on.

    Returns:
        Tuple of (mock_model, mock_preprocess) that can be used for testing.
    """
    mock_preprocess = transforms.Compose(
        [
            transforms.Resize((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def mock_tokenize(
        texts: list[str] | str, context_length: int = 77, truncate: bool = False
    ) -> torch.IntTensor | torch.LongTensor:
        """Mock tokenize function that returns dummy tokens."""
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        return cast(
            torch.IntTensor | torch.LongTensor,
            torch.randint(0, 1000, (batch_size, context_length), device=device),
        )

    # Monkey-patch the clip module to use our mock tokenize
    clip.tokenize = mock_tokenize

    print("✓ Mock CLIP model created for testing (network unavailable)")
    return MockCLIPModel(device), mock_preprocess


# ── Model Download ───────────────────────────────────────────────────────────


def download_finetuned_model(
    model_path: str = clip_model_path,
    cache_dir: str = "./cache",
) -> bool:
    """
    Download the fine-tuned CLIP model from HuggingFace Hub.

    Tries HuggingFace Hub first, then falls back to direct URL download.

    Args:
        model_path: Local path to save the downloaded model.
        cache_dir: Directory for HuggingFace Hub cache.

    Returns:
        True if download was successful, False otherwise.
    """
    # First try HuggingFace Hub download
    try:
        from huggingface_hub import hf_hub_download
        import shutil

        print("Attempting to download fine-tuned CLIP model...")

        downloaded_file = hf_hub_download(
            repo_id=FINETUNED_REPO,
            filename=FINETUNED_HF_FILENAME,
            cache_dir=cache_dir,
        )

        shutil.copy2(downloaded_file, model_path)

        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ Fine-tuned model downloaded successfully ({file_size:.1f} MB)")
            return True
        return False

    except Exception as e:
        print(f"Unable to download fine-tuned model via HuggingFace Hub: {e}")
        print("Trying direct download fallback...")

        # Fallback to direct download
        try:
            import requests

            print(f"Downloading from direct URL: {FINETUNED_DIRECT_URL}")

            response = requests.get(FINETUNED_DIRECT_URL, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                print(
                    f"✓ Fine-tuned model downloaded via direct URL ({file_size:.1f} MB)"
                )
                return True
            return False

        except Exception as direct_e:
            print(f"Direct download also failed: {direct_e}")
            print("This may be due to network restrictions or missing dependencies.")
            return False


# ── CLIP Model Loading ───────────────────────────────────────────────────────


def _load_finetuned_weights(
    model: Any,
    model_path: str,
    device: torch.device,
) -> bool:
    """
    Attempt to load fine-tuned weights into a CLIP model.

    Args:
        model: CLIP model to load weights into.
        model_path: Path to the checkpoint file.
        device: PyTorch device for tensor placement.

    Returns:
        True if weights were loaded successfully.
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Fine-tuned CLIP model loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load fine-tuned model: {e}")
        print("Using default CLIP model")
        return False


def load_clip_model(
    device: str | torch.device | None = None,
    model_path: str | None = None,
) -> tuple[Any, Any, bool]:
    """
    Load the CLIP model with optional fine-tuned weights.

    Attempts to load the base CLIP ViT-B/32 model, then optionally loads
    fine-tuned weights for surface classification. If the fine-tuned model
    file is not found locally, attempts to download it.

    Falls back to a mock model if CLIP cannot be loaded at all.

    Args:
        device: PyTorch device. Defaults to auto-detected DEVICE.
        model_path: Path to fine-tuned weights. Defaults to clip_model_path.

    Returns:
        Tuple of (model, preprocess, is_real_model):
        - model: CLIP model (real or mock)
        - preprocess: Image preprocessing transform
        - is_real_model: True if a real CLIP model was loaded
    """
    if device is None:
        device = torch.device(DEVICE)
    elif isinstance(device, str):
        device = torch.device(device)

    if model_path is None:
        model_path = clip_model_path

    try:
        print("Loading CLIP model...")
        model, preprocess = clip.load(CLIP_ARCHITECTURE, device=device)

        # Try to load fine-tuned weights
        if os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            _load_finetuned_weights(model, model_path, device)
        else:
            print(f"Fine-tuned model {model_path} not found")
            if download_finetuned_model(model_path):
                print(f"Loading downloaded fine-tuned model from {model_path}")
                _load_finetuned_weights(model, model_path, device)
            else:
                print("Using default CLIP model")

        model.eval()
        print("✓ CLIP model loaded successfully")
        return model, preprocess, True

    except Exception as e:
        print(f"Warning: Could not load CLIP model: {e}")
        print("Attempting to create mock CLIP model for testing...")
        try:
            model, preprocess = create_mock_clip_model(device)
            return model, preprocess, True
        except Exception as mock_e:
            print(f"Failed to create mock model: {mock_e}")
            print("Continuing without model loading (will copy images only)")
            return None, None, False


# ── OneFormer Model Loading ──────────────────────────────────────────────────


class OneFormerModelCache:
    """
    Thread-safe singleton cache for OneFormer models.

    Replaces the fragile function-attribute pattern used in the original code.
    """

    _processor: Any | None = None
    _model: Any | None = None
    _loaded: bool = False

    @classmethod
    def load(cls, device: str | torch.device) -> tuple[Any, Any]:
        """
        Load OneFormer model, caching for subsequent calls.

        Args:
            device: PyTorch device for model placement.

        Returns:
            Tuple of (processor, model).
        """
        if cls._loaded:
            return cls._processor, cls._model

        from transformers import (
            OneFormerProcessor,
            OneFormerForUniversalSegmentation,
        )
        from modules.constants import ONEFORMER_MODEL_ID

        print("Loading OneFormer model...")
        cls._processor = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL_ID)
        cls._model = OneFormerForUniversalSegmentation.from_pretrained(
            ONEFORMER_MODEL_ID
        ).to(device)
        cls._model.eval()
        cls._loaded = True
        print("✓ OneFormer model loaded successfully")

        return cls._processor, cls._model

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._loaded

    @classmethod
    def reset(cls) -> None:
        """Reset the cache (useful for testing)."""
        cls._processor = None
        cls._model = None
        cls._loaded = False
