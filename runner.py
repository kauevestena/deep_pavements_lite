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
        gdf = mapillary_data_to_gdf(metadata)

        # Download images
        print("Downloading images...")
        download_all_pictures_from_gdf(gdf, data_path)
        print("Image download complete.")

        # Process images
        print("Processing images...")
        result_gdf = process_images(data_path)
        
        if not result_gdf.empty:
            print(f"Generated surface classifications for {len(result_gdf)} images.")
            print("Surface classification summary:")
            print(result_gdf[['filename', 'road', 'left_sidewalk', 'right_sidewalk']].to_string())
        else:
            print("No surface classifications generated.")
            
        print("Image processing complete.")
    else:
        print("No features found for the given bounding box.")

if __name__ == "__main__":
    main()
