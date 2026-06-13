"""
Shared pytest fixtures for the deep_pavements test suite.

Provides mock objects, sample data, and temporary directories used across
all test modules so that tests run without GPU, network, or real model access.
"""

import os
import tempfile

import clip
import geopandas as gpd
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# mock_image – synthetic 640×480 street‑scene RGB image
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_image():
    """Create a 640×480 RGB image simulating a street scene.

    Layout (top → bottom):
        - Top 50 %  → sky‑blue  (135, 206, 235)
        - 50‑60 %   → lighter gray sidewalk band (180, 180, 180)
        - Bottom 40% → dark gray road (64, 64, 64)
    """
    width, height = 640, 480
    img = Image.new("RGB", (width, height), color=(135, 206, 235))
    pixels = img.load()

    sidewalk_start = int(height * 0.50)  # row 240
    road_start = int(height * 0.60)      # row 288

    for y in range(height):
        for x in range(width):
            if y >= road_start:
                pixels[x, y] = (64, 64, 64)       # road
            elif y >= sidewalk_start:
                pixels[x, y] = (180, 180, 180)     # sidewalk band
            # else: keep sky‑blue

    return img


# ---------------------------------------------------------------------------
# mock_clip_model – lightweight stand‑in for a real CLIP model
# ---------------------------------------------------------------------------

class MockCLIPModel:
    """Minimal CLIP model stub that returns random embeddings."""

    def __init__(self, device="cpu"):
        self.device = device

    def encode_image(self, image_input):
        batch_size = image_input.shape[0]
        return torch.randn(batch_size, 512, device=self.device)

    def encode_text(self, text_tokens):
        batch_size = text_tokens.shape[0]
        return torch.randn(batch_size, 512, device=self.device)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def mock_clip_model():
    """Return ``(model, preprocess)`` using :class:`MockCLIPModel`.

    ``preprocess`` is a torchvision ``Compose`` pipeline that resizes to
    224×224, converts to tensor, and normalises with ImageNet statistics.
    """
    model = MockCLIPModel(device="cpu")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


# ---------------------------------------------------------------------------
# mock_tokenize – monkeypatch clip.tokenize
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tokenize(monkeypatch):
    """Replace ``clip.tokenize`` with a deterministic stub returning random
    integer tensors of shape ``(batch, context_length)``."""

    def _tokenize(texts, context_length=77, truncate=True):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        return torch.randint(0, 1000, (batch_size, context_length))

    monkeypatch.setattr(clip, "tokenize", _tokenize)
    return _tokenize


# ---------------------------------------------------------------------------
# sample_segmentation_mask – 480×640 Cityscapes‑style mask
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_segmentation_mask():
    """Return a 480×640 ``np.uint8`` mask with Cityscapes class IDs.

    Layout:
        - rows 288‑479 (bottom 40 %) = 0  (road)
        - rows 240‑287, cols 0‑319   = 1  (left sidewalk)
        - rows 240‑287, cols 320‑639 = 1  (right sidewalk)
        - 30×50 block at (row 200, col 300) = 13 (car)
        - everything else            = 255 (background)
    """
    height, width = 480, 640
    mask = np.full((height, width), 255, dtype=np.uint8)

    # Road – bottom 40 %
    mask[288:, :] = 0

    # Sidewalks – band between rows 240‑287
    mask[240:288, :320] = 1   # left half
    mask[240:288, 320:] = 1   # right half

    # Car – small block
    mask[200:230, 300:350] = 13

    return mask


# ---------------------------------------------------------------------------
# sample_polygon – flattened rectangle polygon
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_polygon():
    """Flattened [x1,y1, …] rectangle covering the lower‑centre of a
    640×480 image: ``[100, 300, 540, 300, 540, 480, 100, 480]``."""
    return np.array([100, 300, 540, 300, 540, 480, 100, 480], dtype=np.float64)


# ---------------------------------------------------------------------------
# sample_geodataframe – 1‑row GeoDataFrame with a temp image
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_geodataframe(tmp_path, mock_image):
    """GeoDataFrame with one row pointing to a temporary test image.

    Columns: ``id``, ``file_path``, ``geometry``.
    """
    img_path = str(tmp_path / "test_001.jpg")
    mock_image.save(img_path)

    data = {
        "id": ["test_001"],
        "file_path": [img_path],
        "geometry": [Point(0, 0)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# tmp_data_dir – temporary directory with an 'output' subdirectory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create and return a temporary directory that already contains an
    ``output/`` subdirectory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return tmp_path
