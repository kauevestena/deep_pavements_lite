"""
Deep Pavements Lite — Geometry and spatial operations.

Provides functions for:
- Extracting polygon contours from segmentation masks
- Calculating polygon areas
- Finding road axis lines for left/right image splitting
- Classifying sidewalk regions relative to the road axis
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split
from skimage import measure


def extract_polygons_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """
    Extract polygon contours from a binary segmentation mask.

    Uses marching squares for contour detection and Douglas-Peucker for
    polygon simplification.

    Args:
        mask: Binary mask where True/1 indicates object pixels.

    Returns:
        List of flattened polygon coordinate arrays, each shaped (n*2,)
        in format [x1, y1, x2, y2, ...] (image pixel coordinates).
    """
    polygons: list[np.ndarray] = []

    # Find contours using marching squares algorithm
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)

    for contour in contours:
        # Simplify with Douglas-Peucker algorithm (tolerance=2.0 pixels)
        simplified = measure.approximate_polygon(contour, tolerance=2.0)

        if len(simplified) >= 3:
            # Swap from (row, col) to (x, y) and flatten
            polygon = simplified[:, [1, 0]].flatten()
            polygons.append(polygon)

    return polygons


def calculate_polygon_area(polygon_coords: np.ndarray) -> float:
    """
    Calculate the area of a polygon in square pixels.

    Args:
        polygon_coords: Flattened polygon coordinates [x1, y1, x2, y2, ...].

    Returns:
        Area of the polygon in square pixels, or 0.0 for invalid polygons.
    """
    try:
        coords = [
            (polygon_coords[i], polygon_coords[i + 1])
            for i in range(0, len(polygon_coords), 2)
        ]
        if len(coords) < 3:
            return 0.0
        polygon = Polygon(coords)
        return polygon.area
    except Exception:
        return 0.0


def get_road_axis_line(
    road_polygons: list[np.ndarray],
    image_size: tuple[int, int],
) -> LineString | None:
    """
    Find the main axis of road polygons and create a dividing line.

    Determines whether the road is primarily horizontal or vertical,
    then creates a perpendicular line through the centroid extended
    to the image boundaries.

    Args:
        road_polygons: List of road polygon coordinate arrays.
        image_size: Image dimensions (width, height).

    Returns:
        LineString dividing the image into left/right (or top/bottom)
        regions, or None if no valid road is found.
    """
    if not road_polygons:
        return None

    try:
        # Find the largest road polygon
        largest_road: np.ndarray | None = None
        max_area: float = 0

        for polygon_coords in road_polygons:
            area = calculate_polygon_area(polygon_coords)
            if area > max_area:
                max_area = area
                largest_road = polygon_coords

        if largest_road is None:
            return None

        # Convert to Shapely polygon
        coords = [
            (largest_road[i], largest_road[i + 1])
            for i in range(0, len(largest_road), 2)
        ]
        if len(coords) < 3:
            return None

        road_polygon = Polygon(coords)
        minx, miny, maxx, maxy = road_polygon.bounds
        centroid = road_polygon.centroid

        # Road wider than tall → vertical dividing line
        # Road taller than wide → horizontal dividing line
        if (maxx - minx) > (maxy - miny):
            x = centroid.x
            line = LineString([(x, 0), (x, image_size[1])])
        else:
            y = centroid.y
            line = LineString([(0, y), (image_size[0], y)])

        return line

    except Exception as e:
        print(f"Warning: Could not determine road axis: {e}")
        return None


def classify_sidewalk_regions(
    segmentation_result: dict,
    road_axis: LineString,
    image_size: tuple[int, int],
) -> dict[str, str]:
    """
    Classify surfaces on the left and right sides of the road axis.

    Splits the image into two regions using the road axis line,
    assigns each detected sidewalk/car polygon to a side,
    and determines the surface type for each side.

    Args:
        segmentation_result: Results from segment_and_classify().
        road_axis: Line dividing the image into left and right regions.
        image_size: Image dimensions (width, height).

    Returns:
        Dict with keys 'road', 'left_sidewalk', 'right_sidewalk',
        each containing the classified surface type string.
    """
    result: dict[str, str] = {
        "road": "unknown",
        "left_sidewalk": "no_sidewalk",
        "right_sidewalk": "no_sidewalk",
    }

    try:
        # Create image boundary polygon
        image_polygon = Polygon(
            [
                (0, 0),
                (image_size[0], 0),
                (image_size[0], image_size[1]),
                (0, image_size[1]),
            ]
        )

        # Split image into left and right regions
        left_region, right_region = _split_image_regions(
            image_polygon, road_axis, image_size
        )

        # Categorize segments by type and side
        road_polygons: list[tuple[Polygon, str]] = []
        left_sidewalk_polygons: list[tuple[Polygon, str]] = []
        right_sidewalk_polygons: list[tuple[Polygon, str]] = []
        left_car_polygons: list[tuple[Polygon, str]] = []
        right_car_polygons: list[tuple[Polygon, str]] = []

        for segment in segmentation_result.get("pathway_segments", []):
            category = segment.get("category", "")
            polygon_coords = np.array(segment.get("polygon", []))
            surface_type = segment.get("surface_type", {}).get("surface", "unknown")

            if len(polygon_coords) < 6:  # At least 3 points
                continue

            coords = [
                (polygon_coords[i], polygon_coords[i + 1])
                for i in range(0, len(polygon_coords), 2)
            ]
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    continue

                centroid = poly.centroid

                if category == "roads":
                    road_polygons.append((poly, surface_type))
                elif category == "sidewalks":
                    if left_region.contains(centroid):
                        left_sidewalk_polygons.append((poly, surface_type))
                    elif right_region.contains(centroid):
                        right_sidewalk_polygons.append((poly, surface_type))
                elif category == "car":
                    if left_region.contains(centroid):
                        left_car_polygons.append((poly, surface_type))
                    elif right_region.contains(centroid):
                        right_car_polygons.append((poly, surface_type))

            except Exception:
                continue

        # Classify road surface (use largest road polygon)
        if road_polygons:
            largest_road = max(road_polygons, key=lambda x: x[0].area)
            result["road"] = largest_road[1]

        # Classify each side
        result["left_sidewalk"] = classify_side_surface(
            left_sidewalk_polygons, left_car_polygons, road_polygons
        )
        result["right_sidewalk"] = classify_side_surface(
            right_sidewalk_polygons, right_car_polygons, road_polygons
        )

        return result

    except Exception as e:
        print(f"Warning: Could not classify sidewalk regions: {e}")
        return result


def _split_image_regions(
    image_polygon: Polygon,
    road_axis: LineString,
    image_size: tuple[int, int],
) -> tuple[Polygon, Polygon]:
    """Split the image polygon into left and right regions using the road axis."""
    try:
        split_result = split(image_polygon, road_axis)
        if len(split_result.geoms) >= 2:
            return split_result.geoms[0], split_result.geoms[1]
    except Exception:
        pass

    # Fallback: simple vertical/horizontal division
    centroid = road_axis.centroid
    if road_axis.coords[0][0] == road_axis.coords[1][0]:  # Vertical line
        x = centroid.x
        left = Polygon([(0, 0), (x, 0), (x, image_size[1]), (0, image_size[1])])
        right = Polygon(
            [
                (x, 0),
                (image_size[0], 0),
                (image_size[0], image_size[1]),
                (x, image_size[1]),
            ]
        )
    else:  # Horizontal line
        y = centroid.y
        left = Polygon([(0, 0), (image_size[0], 0), (image_size[0], y), (0, y)])
        right = Polygon(
            [
                (0, y),
                (image_size[0], y),
                (image_size[0], image_size[1]),
                (0, image_size[1]),
            ]
        )
    return left, right


def classify_side_surface(
    sidewalk_polygons: list[tuple[Polygon, str]],
    car_polygons: list[tuple[Polygon, str]],
    road_polygons: list[tuple[Polygon, str]],
) -> str:
    """
    Classify the surface type for one side of the road.

    Decision logic:
    1. If sidewalks are present, return the surface type of the largest one.
    2. If no sidewalks but cars are present with area ≥ 1/3 of road area,
       return 'car_hindered'.
    3. Otherwise, return 'no_sidewalk'.

    Args:
        sidewalk_polygons: List of (polygon, surface_type) for sidewalks on this side.
        car_polygons: List of (polygon, surface_type) for cars on this side.
        road_polygons: List of (polygon, surface_type) for all roads.

    Returns:
        Surface classification string.
    """
    if sidewalk_polygons:
        largest_sidewalk = max(sidewalk_polygons, key=lambda x: x[0].area)
        return largest_sidewalk[1]

    if not car_polygons:
        return "no_sidewalk"

    total_car_area = sum(poly.area for poly, _ in car_polygons)
    total_road_area = sum(poly.area for poly, _ in road_polygons)

    if total_road_area == 0:
        return "no_sidewalk"

    car_road_ratio = total_car_area / total_road_area

    if car_road_ratio < 1 / 3:
        return "no_sidewalk"
    else:
        return "car_hindered"
