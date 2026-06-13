"""Tests for deep_pavements.pipeline – end‑to‑end image processing."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from deep_pavements.pipeline import process_images


# ── process_images ────────────────────────────────────────────────────────


def test_process_images_empty_gdf(tmp_data_dir):
    """An empty GeoDataFrame must return an empty GeoDataFrame (no crash)."""
    empty_gdf = gpd.GeoDataFrame()

    result = process_images(empty_gdf, str(tmp_data_dir))

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty


def test_process_images_missing_file(tmp_data_dir):
    """A GDF whose file_path points to a non‑existent file must not crash."""
    gdf = gpd.GeoDataFrame(
        {
            "id": ["missing_001"],
            "file_path": [str(tmp_data_dir / "does_not_exist.jpg")],
            "geometry": [Point(0, 0)],
        },
        crs="EPSG:4326",
    )

    result = process_images(gdf, str(tmp_data_dir))

    assert isinstance(result, gpd.GeoDataFrame)
