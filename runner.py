import argparse
import os
from my_mappilary_api.mapillary_api import (
    get_mapillary_images_metadata,
    mapillary_data_to_gdf,
    download_all_pictures_from_gdf,
)
from lib import *

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

    # Read Mapillary token
    token_files = ["mapillary_token", "workspace/data/mapillary_token", "data/mapillary_token"]
    mapillary_token = None
    
    for token_file in token_files:
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                mapillary_token = f.read().strip()
            print(f"Found token in {token_file}")
            break
    
    if not mapillary_token:
        print("Error: No Mapillary token found. Please create a file named 'mapillary_token' with your token.")
        print("Expected locations:", token_files)
        return

    # Get features
    print("Getting features from Mapillary...")
    metadata = get_mapillary_images_metadata(args.lon_min, args.lat_min, args.lon_max, args.lat_max, token=mapillary_token)

    if metadata.get("data"):
        print(f"Found {len(metadata['data'])} features.")

        # Convert to GeoDataFrame
        if args.max_images:
            print(
                f"Sampling up to {args.max_images} image(s) from metadata for processing."
            )

        gdf = mapillary_data_to_gdf(metadata, max_images=args.max_images)

        if args.max_images and len(gdf) < len(metadata.get("data", [])):
            print(f"Selected {len(gdf)} image(s) for download and analysis.")

        # Download images and get GDF with file paths
        print("Downloading images...")
        # Ensure data directory exists before saving any images
        os.makedirs(data_path, exist_ok=True)
        # Add file paths to the GDF
        gdf['file_path'] = gdf['id'].apply(lambda x: os.path.join(data_path, f"{x}.jpg"))
        download_all_pictures_from_gdf(gdf, data_path, scale_factor=scale_factor)
        print("Image download complete.")

        # Process images using the GDF with metadata
        print("Processing images...")
        result_gdf = process_images(gdf, data_path, debug_mode=args.debug)
        
        if not result_gdf.empty:
            print(f"Generated surface classifications for {len(result_gdf)} images.")
            print("Surface classification summary:")
            print(result_gdf[['filename', 'image_id', 'road', 'left_sidewalk', 'right_sidewalk']].to_string())
        else:
            print("No surface classifications generated.")
            
        print("Image processing complete.")
    else:
        print("No features found for the given bounding box.")

if __name__ == "__main__":
    main()
