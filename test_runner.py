#!/usr/bin/env python3
"""
Test runner for deep_pavements_lite with single image processing

This script is similar to runner.py but limits processing to a single image
for testing purposes, particularly useful in resource-constrained environments.
"""

import argparse
import os
from my_mappilary_api.mapillary_api import get_mapillary_images_metadata, mapillary_data_to_gdf, download_all_pictures_from_gdf
from lib import *

def main():
    parser = argparse.ArgumentParser(description="Deep Pavements Lite Test Runner (Single Image)")
    parser.add_argument("--lat_min", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--lon_min", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--lat_max", type=float, required=True, help="Maximum latitude")
    parser.add_argument("--lon_max", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--max_images", type=int, default=1, help="Maximum number of images to process (default: 1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediary results")
    args = parser.parse_args()

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
        
        # Limit to max_images for testing
        if len(metadata['data']) > args.max_images:
            print(f"Limiting to {args.max_images} image(s) for testing")
            metadata['data'] = metadata['data'][:args.max_images]

        # Convert to GeoDataFrame
        gdf = mapillary_data_to_gdf(metadata)

        # Download images and get GDF with file paths
        print("Downloading images...")
        # Add file paths to the GDF
        gdf['file_path'] = gdf['id'].apply(lambda x: os.path.join(data_path, f"{x}.jpg"))
        download_all_pictures_from_gdf(gdf, data_path)
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