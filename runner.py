import argparse
import os
from my_mappilary_api.mapillary_api import get_mapillary_images_metadata, mapillary_data_to_gdf, download_all_pictures_from_gdf
from lib import *

def main():
    parser = argparse.ArgumentParser(description="Deep Pavements Lite Runner")
    parser.add_argument("--lat_min", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--lon_min", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--lat_max", type=float, required=True, help="Maximum latitude")
    parser.add_argument("--lon_max", type=float, required=True, help="Maximum longitude")
    args = parser.parse_args()

    # Read Mapillary token
    with open("mapillary_token", "r") as f:
        mapillary_token = f.read().strip()

    # Get features
    print("Getting features from Mapillary...")
    metadata = get_mapillary_images_metadata(args.lon_min, args.lat_min, args.lon_max, args.lat_max, token=mapillary_token)

    if metadata.get("data"):
        print(f"Found {len(metadata['data'])} features.")

        # Convert to GeoDataFrame
        gdf = mapillary_data_to_gdf(metadata)

        # Download images
        print("Downloading images...")
        download_all_pictures_from_gdf(gdf, data_path)
        print("Image download complete.")

        # Process images
        print("Processing images...")
        process_images(data_path)
        print("Image processing complete.")
    else:
        print("No features found for the given bounding box.")

if __name__ == "__main__":
    main()
