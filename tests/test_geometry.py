"""Tests for deep_pavements.geometry – polygon extraction, area, road axis,
sidewalk region classification."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from deep_pavements.geometry import (
    calculate_polygon_area,
    classify_side_surface,
    classify_sidewalk_regions,
    extract_polygons_from_mask,
    get_road_axis_line,
)


# ── extract_polygons_from_mask ────────────────────────────────────────────


def test_extract_polygons_from_binary_mask():
    """A filled circle on a 100×100 mask should yield ≥1 polygon with ≥6 coords."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    rr, cc = np.ogrid[:100, :100]
    circle = ((rr - 50) ** 2 + (cc - 50) ** 2) <= 30 ** 2
    mask[circle] = 1

    polygons = extract_polygons_from_mask(mask)

    assert len(polygons) >= 1, "Expected at least one polygon from the circle mask"
    assert len(polygons[0]) >= 6, "Polygon should have at least 3 vertices (6 coords)"


def test_extract_polygons_empty_mask():
    """An all‑zeros mask should return an empty list."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    polygons = extract_polygons_from_mask(mask)
    assert polygons == []


# ── calculate_polygon_area ────────────────────────────────────────────────


def test_calculate_polygon_area_square():
    """A 100×100 axis‑aligned square has area ≈ 10 000 sq px."""
    # (0,0) → (100,0) → (100,100) → (0,100)
    square = np.array([0, 0, 100, 0, 100, 100, 0, 100], dtype=np.float64)
    area = calculate_polygon_area(square)
    assert abs(area - 10_000) < 1, f"Expected ~10000, got {area}"


def test_calculate_polygon_area_degenerate():
    """A polygon with only 2 points (a line) should return 0.0."""
    line = np.array([0, 0, 10, 10], dtype=np.float64)
    assert calculate_polygon_area(line) == 0.0


def test_calculate_polygon_area_empty():
    """An empty coordinate array should return 0.0."""
    assert calculate_polygon_area(np.array([], dtype=np.float64)) == 0.0


# ── get_road_axis_line ────────────────────────────────────────────────────


def test_get_road_axis_horizontal_road():
    """A road wider than tall should produce a *vertical* dividing line
    (both endpoints share the same x‑coordinate)."""
    # Wide rectangle: x 0→400, y 300→480
    road = np.array([0, 300, 400, 300, 400, 480, 0, 480], dtype=np.float64)
    image_size = (640, 480)

    axis = get_road_axis_line([road], image_size)

    assert axis is not None
    coords = list(axis.coords)
    assert len(coords) == 2
    # Vertical line ⇒ same x
    assert coords[0][0] == pytest.approx(coords[1][0], abs=1e-3)


def test_get_road_axis_vertical_road():
    """A road taller than wide should produce a *horizontal* dividing line
    (both endpoints share the same y‑coordinate)."""
    # Tall rectangle: x 250→350, y 0→480
    road = np.array([250, 0, 350, 0, 350, 480, 250, 480], dtype=np.float64)
    image_size = (640, 480)

    axis = get_road_axis_line([road], image_size)

    assert axis is not None
    coords = list(axis.coords)
    assert len(coords) == 2
    # Horizontal line ⇒ same y
    assert coords[0][1] == pytest.approx(coords[1][1], abs=1e-3)


def test_get_road_axis_empty_list():
    """An empty polygon list should return ``None``."""
    assert get_road_axis_line([], (640, 480)) is None


def test_get_road_axis_invalid_polygon():
    """A polygon with fewer than 3 vertices (< 6 coords) should return ``None``."""
    tiny = np.array([0, 0, 10, 10], dtype=np.float64)
    assert get_road_axis_line([tiny], (640, 480)) is None


# ── classify_side_surface ─────────────────────────────────────────────────


def test_classify_side_surface_with_sidewalk():
    """When sidewalk polygons are present, the surface type of the largest
    sidewalk should be returned."""
    small_sw = (Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), "concrete")
    large_sw = (Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), "asphalt")
    road = (Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]), "asphalt")

    result = classify_side_surface([small_sw, large_sw], [], [road])
    assert result == "asphalt"


def test_classify_side_surface_no_sidewalk_no_car():
    """Empty sidewalk *and* car lists → ``'no_sidewalk'``."""
    road = (Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]), "asphalt")
    assert classify_side_surface([], [], [road]) == "no_sidewalk"


def test_classify_side_surface_car_hindered():
    """A car whose area ≥ ⅓ of the road area → ``'car_hindered'``."""
    # Car area = 100×100 = 10 000; road area = 150×150 = 22 500  → ratio ≈ 0.44
    car = (Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), "car")
    road = (Polygon([(0, 0), (150, 0), (150, 150), (0, 150)]), "asphalt")

    assert classify_side_surface([], [car], [road]) == "car_hindered"


def test_classify_side_surface_small_car():
    """A car whose area < ⅓ of the road area → ``'no_sidewalk'``."""
    car = (Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), "car")       # area 100
    road = (Polygon([(0, 0), (200, 0), (200, 200), (0, 200)]), "asphalt")  # area 40 000

    assert classify_side_surface([], [car], [road]) == "no_sidewalk"


# ── classify_sidewalk_regions ─────────────────────────────────────────────


def test_classify_sidewalk_regions_basic():
    """Full integration: verify output dict has the expected keys."""
    # Build a minimal segmentation_result
    road_poly = [100, 300, 540, 300, 540, 480, 100, 480]
    left_sw_poly = [10, 260, 310, 260, 310, 290, 10, 290]
    right_sw_poly = [330, 260, 630, 260, 630, 290, 330, 290]

    segmentation_result = {
        "filename": "test.jpg",
        "image_size": (640, 480),
        "pathway_segments": [
            {
                "category": "roads",
                "polygon": road_poly,
                "surface_type": {"surface": "asphalt", "confidence": 0.9},
                "segment_id": "roads_0",
            },
            {
                "category": "sidewalks",
                "polygon": left_sw_poly,
                "surface_type": {"surface": "concrete", "confidence": 0.8},
                "segment_id": "sidewalks_0",
            },
            {
                "category": "sidewalks",
                "polygon": right_sw_poly,
                "surface_type": {"surface": "paving_stones", "confidence": 0.7},
                "segment_id": "sidewalks_1",
            },
        ],
    }

    # Vertical dividing line through the centre of the road
    road_axis = LineString([(320, 0), (320, 480)])
    image_size = (640, 480)

    result = classify_sidewalk_regions(segmentation_result, road_axis, image_size)

    assert "road" in result
    assert "left_sidewalk" in result
    assert "right_sidewalk" in result
