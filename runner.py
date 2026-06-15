import argparse
import os
from my_mappilary_api.mapillary_api import (
    get_mapillary_images_metadata,
    mapillary_data_to_gdf,
    download_all_pictures_from_gdf,
)
from deep_pavements import process_images
from deep_pavements.constants import data_path as DEFAULT_DATA_PATH
from deep_pavements.visualization import generate_map

def main():
    parser = argparse.ArgumentParser(description="Deep Pavements Lite Runner")
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
    import os
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


if __name__ == "__main__":
    main()
