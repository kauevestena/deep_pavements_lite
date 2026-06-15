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

from modules.classification import segment_and_classify
from modules.constants import (
    DEBUG_OUTPUT_DIR,
    DEVICE,
    MODEL_INFO,
    ext_in,
    ext_out,
)
from modules.debug import (
    generate_debug_html_report,
    save_debug_image_metadata,
)
from modules.geometry import (
    classify_sidewalk_regions,
    get_road_axis_line,
)
from modules.models import load_clip_model


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


def main() -> None:
    import argparse
    from my_mappilary_api.mapillary_api import (
        get_mapillary_images_metadata,
        mapillary_data_to_gdf,
        download_all_pictures_from_gdf,
    )
    from modules.visualization import generate_map
    from modules.constants import data_path as DEFAULT_DATA_PATH

    parser = argparse.ArgumentParser(description="Deep Pavements Lite CLI")
    parser.add_argument("--lat_min", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--lon_min", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--lat_max", type=float, required=True, help="Maximum latitude")
    parser.add_argument("--lon_max", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediary results")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to randomly sample for processing",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=DEFAULT_DATA_PATH,
        help="Directory to store downloaded images and generated outputs",
    )
    parser.add_argument(
        "--mapillary_token",
        default=None,
        help="Mapillary access token used to authenticate requests",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for I/O operations (default: 1)",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="Generate an interactive Leaflet map of results",
    )
    resolution_group = parser.add_mutually_exclusive_group()
    resolution_group.add_argument(
        "--half_res",
        action="store_true",
        help="Downscale downloaded images to half of the original resolution",
    )
    resolution_group.add_argument(
        "--quarter_res",
        action="store_true",
        help="Downscale downloaded images to one quarter of the original resolution",
    )
    args = parser.parse_args()

    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be a positive integer")

    scale_factor = 0.5 if args.half_res else 0.25 if args.quarter_res else 1.0
    output_dir = os.path.abspath(args.output_dir)

    # Read Mapillary token
    mapillary_token = _get_mapillary_token(args.mapillary_token)

    if not mapillary_token:
        print("Error: No Mapillary token found. Please create a file named 'mapillary_token' with your token.")
        print("Expected locations: mapillary_token, workspace/data/mapillary_token, data/mapillary_token")
        print("Or pass --mapillary_token <token> on the command line.")
        return

    # Get features
    print("Getting features from Mapillary...")
    metadata = get_mapillary_images_metadata(args.lon_min, args.lat_min, args.lon_max, args.lat_max, token=mapillary_token)

    if metadata.get("data"):
        print(f"Found {len(metadata['data'])} features.")

        # Convert to GeoDataFrame
        if args.max_images and len(metadata.get("data", [])) > args.max_images:
            print(
                f"Sampling up to {args.max_images} image(s) from metadata for processing."
            )
            metadata["data"] = metadata["data"][:args.max_images]

        gdf = mapillary_data_to_gdf(metadata)

        # Download images and get GDF with file paths
        print("Downloading images...")
        os.makedirs(output_dir, exist_ok=True)
        gdf['file_path'] = gdf['id'].apply(lambda x: os.path.join(output_dir, f"{x}.jpg"))
        download_all_pictures_from_gdf(gdf, output_dir, scale_factor=scale_factor)
        print("Image download complete.")

        # Process images using the GDF with metadata
        print("Processing images...")
        result_gdf = process_images(gdf, output_dir, debug_mode=args.debug, workers=args.workers)

        if not result_gdf.empty:
            print(f"Generated surface classifications for {len(result_gdf)} images.")
            print("Surface classification summary:")
            print(result_gdf[['filename', 'image_id', 'road', 'left_sidewalk', 'right_sidewalk']].to_string())

            # Generate interactive map if requested
            if args.map:
                map_output = os.path.join(output_dir, "output")
                generate_map(result_gdf, map_output)
        else:
            print("No surface classifications generated.")

        print("Image processing complete.")
    else:
        print("No features found for the given bounding box.")


def _get_mapillary_token(cli_token: str | None = None) -> str | None:
    """
    Get Mapillary API token from CLI argument, environment, or file.

    Priority: CLI arg → environment variable → token files.
    """
    if cli_token:
        return cli_token.strip()

    # Check environment variable
    env_token = os.environ.get("MAPILLARY_API")
    if env_token:
        return env_token.strip()

    # Check token files
    token_files = ["mapillary_token", "workspace/data/mapillary_token", "data/mapillary_token"]
    for token_file in token_files:
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                token = f.read().strip()
            print(f"Found token in {token_file}")
            return token

    return None

