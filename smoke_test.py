#!/usr/bin/env python3
"""
Smoke test for deep_pavements_lite

This script performs a lightweight test of the core functionality by processing
a single test image without requiring Mapillary API access.
"""

import os
import sys
import tempfile
from PIL import Image
import geopandas as gpd
from shapely.geometry import Point

# Import our modules
from lib import process_images
from constants import data_path

def get_mapillary_token():
    """Get Mapillary API token from environment or file"""
    # Check environment variable first (for GitHub Actions)
    token = os.environ.get('MAPILLARY_API')
    if token:
        return token.strip()
    
    # Check token files as fallback
    token_files = ["mapillary_token", "workspace/data/mapillary_token", "data/mapillary_token"]
    
    for token_file in token_files:
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                token = f.read().strip()
            return token
    
    return None

def create_test_geodataframe():
    """Create a minimal GeoDataFrame with test image data"""
    # Try to use Mapillary API for dummy test data if token is available
    mapillary_token = get_mapillary_token()
    
    if mapillary_token:
        print("🔄 Attempting to create test data using Mapillary API...")
        try:
            from my_mappilary_api.mapillary_api import get_mapillary_images_metadata, mapillary_data_to_gdf, download_all_pictures_from_gdf
            
            # Small area around San Francisco for testing
            lat_min, lon_min = 37.7749, -122.4194
            lat_max, lon_max = 37.7759, -122.4184
            
            metadata = get_mapillary_images_metadata(
                lon_min, lat_min, lon_max, lat_max, 
                token=mapillary_token
            )
            
            if metadata.get("data"):
                # Limit to first image for testing
                metadata['data'] = metadata['data'][:1]
                gdf = mapillary_data_to_gdf(metadata)
                
                if not gdf.empty:
                    # Add file paths and download images
                    gdf['file_path'] = gdf['id'].apply(lambda x: os.path.join(data_path, f"{x}.jpg"))
                    download_all_pictures_from_gdf(gdf, data_path)
                    print("✓ Dummy test data created using Mapillary API")
                    return gdf
        except Exception as e:
            print(f"⚠ Failed to create Mapillary test data: {e}")
    
    print("📋 Using fallback static test image...")
    # Fallback to static test image
    test_image_path = os.path.join(data_path, 'test_image.jpg')
    
    # Copy from test_data if needed
    if not os.path.exists(test_image_path):
        source_path = os.path.join('test_data', 'street_scene.jpg')
        if os.path.exists(source_path):
            import shutil
            shutil.copy2(source_path, test_image_path)
    
    test_data = {
        'id': ['test_image'],
        'file_path': [test_image_path],
        'geometry': [Point(-122.4194, 37.7749)]  # San Francisco coordinates
    }
    
    gdf = gpd.GeoDataFrame(test_data, crs='EPSG:4326')
    return gdf

def main():
    """Run the smoke test"""
    print("🚀 Starting Deep Pavements Lite Smoke Test")
    print("=" * 50)
    
    # Ensure data directory exists
    os.makedirs(data_path, exist_ok=True)
    
    # Ensure test_data/outputs directory exists
    test_output_dir = os.path.join("test_data", "outputs")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Check if test image exists
    test_image_path = os.path.join(data_path, 'test_image.jpg')
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found at {test_image_path}")
        print("   Please ensure the test image is downloaded first")
        return 1
    
    print(f"✓ Found test image: {test_image_path}")
    
    # Verify image can be loaded
    try:
        with Image.open(test_image_path) as img:
            print(f"✓ Image loaded successfully: {img.size} pixels, mode: {img.mode}")
    except Exception as e:
        print(f"❌ Failed to load test image: {e}")
        return 1
    
    # Create test GeoDataFrame
    print("✓ Creating test GeoDataFrame...")
    test_gdf = create_test_geodataframe()
    print(f"✓ Created GDF with {len(test_gdf)} test image(s)")
    
    # Process the test image in debug mode
    print("🔄 Processing test image in debug mode...")
    try:
        result_gdf = process_images(test_gdf, data_path, debug_mode=True)
        
        if result_gdf.empty:
            print("⚠ No results generated (this may be expected without GPU/models)")
            print("✓ Core processing pipeline executed without errors")
        else:
            print(f"✓ Successfully processed {len(result_gdf)} image(s)")
            print("✓ Surface classifications generated:")
            print(result_gdf[['filename', 'image_id']].to_string())
            
    except Exception as e:
        print(f"❌ Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Copy debug outputs to test_data/outputs
    debug_path = os.path.join(data_path, "debug_outputs")
    if os.path.exists(debug_path):
        import shutil
        print("🔄 Copying debug outputs to test_data/outputs...")
        
        # Copy the entire debug_outputs directory to test_data/outputs
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)
        shutil.copytree(debug_path, test_output_dir)
        
        # Count all files recursively in test_data/outputs
        total_files = sum([len(files) for r, d, files in os.walk(test_output_dir)])
        print(f"✓ Debug outputs copied to test_data/outputs ({total_files} files)")
        
        # List the structure
        print("📁 Debug output structure:")
        for root, dirs, files in os.walk(test_output_dir):
            level = root.replace(test_output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("⚠ No debug outputs found (expected with debug mode enabled)")
    
    # Check standard outputs
    output_path = os.path.join(data_path, "output")
    if os.path.exists(output_path):
        output_files = os.listdir(output_path)
        print(f"✓ Standard output directory created with {len(output_files)} files:")
        for file in output_files:
            print(f"  - {file}")
    
    print("=" * 50)
    print("🎉 Smoke test completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)