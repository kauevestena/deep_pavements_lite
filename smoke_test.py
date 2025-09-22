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

def create_test_geodataframe():
    """Create a minimal GeoDataFrame with test image data"""
    # Create test data for a single image
    test_data = {
        'id': ['test_image'],
        'file_path': [os.path.join(data_path, 'test_image.jpg')],
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
    
    # Process the test image
    print("🔄 Processing test image...")
    try:
        result_gdf = process_images(test_gdf, data_path)
        
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
    
    # Check outputs
    output_path = os.path.join(data_path, "output")
    if os.path.exists(output_path):
        output_files = os.listdir(output_path)
        print(f"✓ Output directory created with {len(output_files)} files:")
        for file in output_files:
            print(f"  - {file}")
    
    print("=" * 50)
    print("🎉 Smoke test completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)