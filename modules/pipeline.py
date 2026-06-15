"""
Deep Pavements Lite — Main processing pipeline.

Orchestrates the full image processing workflow:
1. Load models (CLIP + OneFormer)
2. Iterate over images in the input GeoDataFrame
3. Perform segmentation and classification per image
4. Determine road axis and classify left/right sidewalk surfaces
5. Export results as GeoJSON and optionally GeoPackage
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from deep_pavements.classification import segment_and_classify
from deep_pavements.constants import (
    DEBUG_OUTPUT_DIR,
    DEVICE,
    MODEL_INFO,
    ext_in,
    ext_out,
)
from deep_pavements.debug import (
    generate_debug_html_report,
    save_debug_image_metadata,
)
from deep_pavements.geometry import (
    classify_sidewalk_regions,
    get_road_axis_line,
)
from deep_pavements.models import load_clip_model


def process_images(
    input_gdf: gpd.GeoDataFrame,
    data_path: str,
    debug_mode: bool = False,
    workers: int = 1,
) -> gpd.GeoDataFrame:
    """
    Process images using metadata from a GeoDataFrame with CLIP and OneFormer models.

    This function implements the main image processing pipeline that:
    1. Loads CLIP model (with optional fine-tuned weights for surface classification)
    2. Processes images specified in the input GeoDataFrame
    3. Performs semantic segmentation and surface material classification
    4. Analyzes road/sidewalk layout and classifies surfaces on both sides
    5. Returns structured geodataframe with surface classifications including GPS coordinates

    Args:
        input_gdf: GeoDataFrame containing image metadata with columns:
                   'id', 'file_path', 'geometry' (Point with GPS coordinates).
        data_path: Path to directory containing input images and output directory.
        debug_mode: If True, save all intermediary results to debug_outputs folder.
        workers: Number of parallel workers for I/O operations (default 1 = sequential).

    Returns:
        GeoDataFrame with surface classifications: filename, image_id,
        road, road_confidence, left_sidewalk, left_confidence,
        right_sidewalk, right_confidence, geometry.
    """
    device = torch.device(DEVICE)

    # Load CLIP model
    model, preprocess, model_available = load_clip_model(device)

    # Create output directory
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)
    print(f"Created output directory: {output_path}")

    # Create debug directories if needed
    debug_dirs = _setup_debug_dirs(data_path, debug_mode)

    # Check for empty input
    if input_gdf.empty:
        print("No images found in input GeoDataFrame")
        return gpd.GeoDataFrame()

    print(f"Found {len(input_gdf)} images to process")

    # Process each image
    surface_results: list[dict[str, Any]] = []
    debug_data: list[dict[str, Any]] = []

    for idx, row in tqdm(
        input_gdf.iterrows(), total=len(input_gdf), desc="Processing images"
    ):
        image_id = row["id"]
        filename = f"{image_id}.jpg"
        image_path = row["file_path"]
        coordinates = row["geometry"]

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue

        image = Image.open(image_path)

        if model_available:
            # Preprocess for CLIP
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)

            # Build debug info if needed
            current_debug_info = None
            if debug_mode and debug_dirs:
                current_debug_info = {
                    "debug_path": debug_dirs["root"],
                    "debug_images_path": debug_dirs["images"],
                    "debug_segmented_path": debug_dirs["segmented"],
                    "debug_metadata_path": debug_dirs["metadata"],
                    "image_id": image_id,
                    "filename": filename,
                    "image_size": image.size,
                    "coordinates": coordinates,
                }

            # Segmentation + classification
            segmentation_result = segment_and_classify(
                image, model, preprocess, device, filename, current_debug_info
            )

            # Road axis detection and sidewalk classification
            result_entry = _classify_road_and_sidewalks(
                segmentation_result, image, image_id, filename, coordinates
            )

            if result_entry:
                surface_results.append(result_entry)

                if debug_mode:
                    debug_entry = {
                        "image_id": image_id,
                        "filename": filename,
                        "segmentation_result": segmentation_result,
                        "surface_classification": result_entry,
                        "road_axis": result_entry.get("_road_axis_wkt"),
                        "coordinates": (
                            f"{coordinates.x}, {coordinates.y}"
                            if hasattr(coordinates, "x")
                            else str(coordinates)
                        ),
                    }
                    debug_data.append(debug_entry)

        # Save processed image to output directory
        output_filename = filename.replace(ext_in, ext_out)
        output_filepath = os.path.join(output_path, output_filename)
        image.save(output_filepath)

        # Save debug outputs
        if debug_mode and debug_dirs:
            debug_image_path = os.path.join(debug_dirs["images"], filename)
            image.save(debug_image_path)
            save_debug_image_metadata(
                image_id,
                filename,
                image.size,
                coordinates,
                image_path,
                debug_dirs["metadata"],
            )

    print("Image processing complete.")

    # Generate debug report
    if debug_mode and debug_data and debug_dirs:
        generate_debug_html_report(debug_data, debug_dirs["reports"])

    # Build and export results
    return _export_results(surface_results, output_path)


def _setup_debug_dirs(
    data_path: str, debug_mode: bool
) -> dict[str, str] | None:
    """Create debug output directory structure if debug mode is enabled."""
    if not debug_mode:
        return None

    debug_path = os.path.join(data_path, DEBUG_OUTPUT_DIR)
    dirs = {
        "root": debug_path,
        "images": os.path.join(debug_path, "images"),
        "segmented": os.path.join(debug_path, "segmented_images"),
        "metadata": os.path.join(debug_path, "metadata"),
        "reports": os.path.join(debug_path, "reports"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    print(f"Debug mode enabled — created debug output directory: {debug_path}")
    return dirs


def _classify_road_and_sidewalks(
    segmentation_result: dict,
    image: Image.Image,
    image_id: str,
    filename: str,
    coordinates: Any,
) -> dict[str, Any] | None:
    """Extract road polygons, find axis, classify sidewalk regions."""
    road_polygons: list[np.ndarray] = []
    for segment in segmentation_result.get("pathway_segments", []):
        if segment.get("category") == "roads":
            road_polygons.append(np.array(segment.get("polygon", [])))

    if not road_polygons:
        return None

    road_axis = get_road_axis_line(road_polygons, image.size)

    if not road_axis:
        return None

    surface_classification = classify_sidewalk_regions(
        segmentation_result, road_axis, image.size
    )

    # Extract confidence values from segmentation result segments
    road_confidence = _get_segment_confidence(segmentation_result, "roads")
    left_confidence = _get_segment_confidence(segmentation_result, "sidewalks", side="left")
    right_confidence = _get_segment_confidence(segmentation_result, "sidewalks", side="right")

    result_entry = {
        "filename": filename,
        "image_id": image_id,
        "road": surface_classification["road"],
        "road_confidence": road_confidence,
        "left_sidewalk": surface_classification["left_sidewalk"],
        "left_confidence": left_confidence,
        "right_sidewalk": surface_classification["right_sidewalk"],
        "right_confidence": right_confidence,
        "geometry": coordinates,
        "_road_axis_wkt": road_axis.wkt if road_axis else None,
    }
    return result_entry


def _get_segment_confidence(
    segmentation_result: dict,
    category: str,
    side: str | None = None,
) -> float:
    """Extract the confidence score for the largest segment of a given category."""
    best_confidence = 0.0
    for segment in segmentation_result.get("pathway_segments", []):
        if segment.get("category") == category:
            surface_type = segment.get("surface_type", {})
            if isinstance(surface_type, dict):
                confidence = surface_type.get("confidence", 0.0)
                if confidence > best_confidence:
                    best_confidence = confidence
    return best_confidence


def _export_results(
    surface_results: list[dict[str, Any]],
    output_path: str,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame from results and export as GeoJSON + GeoPackage."""
    if not surface_results:
        print("No valid road detections found in any images.")
        return gpd.GeoDataFrame()

    # Remove internal fields before export
    export_results = []
    for r in surface_results:
        entry = {k: v for k, v in r.items() if not k.startswith("_")}
        export_results.append(entry)

    gdf = gpd.GeoDataFrame(export_results, crs="EPSG:4326")

    # Save as GeoJSON
    geojson_path = os.path.join(output_path, "surface_classifications.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved surface classifications to {geojson_path}")

    # Save as GeoPackage
    gpkg_path = os.path.join(output_path, "surface_classifications.gpkg")
    try:
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"Saved surface classifications to {gpkg_path}")
    except Exception as e:
        print(f"Warning: Could not save GeoPackage: {e}")

    # Save model version info
    model_info_path = os.path.join(output_path, "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(MODEL_INFO, f, indent=2)

    return gdf
