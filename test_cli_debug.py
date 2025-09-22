#!/usr/bin/env python3
"""
Simple CLI test for debug mode without requiring Mapillary API.
"""

import os
import sys
import tempfile
import argparse
from PIL import Image
import geopandas as gpd
from shapely.geometry import Point

# Import our modules
from lib import process_images
from constants import data_path

def main():
    parser = argparse.ArgumentParser(description="Test Deep Pavements Lite Debug Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediary results")
    args = parser.parse_args()

    print(f"🚀 Starting CLI Debug Test (debug_mode={args.debug})")
    print("=" * 60)
    
    # Ensure test image exists
    test_image_path = os.path.join(data_path, 'test_image.jpg')
    if not os.path.exists(test_image_path):
        source_path = os.path.join('test_data', 'street_scene.jpg')
        if os.path.exists(source_path):
            import shutil
            shutil.copy2(source_path, test_image_path)
            print(f"✓ Copied test image to {test_image_path}")
        else:
            print("❌ Test image not found")
            return 1
    
    # Create test GeoDataFrame
    test_data = {
        'id': ['cli_test'],
        'file_path': [test_image_path],
        'geometry': [Point(-122.4194, 37.7749)]  # San Francisco coordinates
    }
    
    test_gdf = gpd.GeoDataFrame(test_data, crs='EPSG:4326')
    print(f"✓ Created test GeoDataFrame with {len(test_gdf)} image(s)")
    
    # Process the test image with specified debug mode
    print(f"🔄 Processing test image with debug_mode={args.debug}...")
    try:
        result_gdf = process_images(test_gdf, data_path, debug_mode=args.debug)
        
        if args.debug:
            print("✅ Debug mode processing completed!")
            
            # Check debug outputs
            debug_path = os.path.join(data_path, "debug_outputs")
            if os.path.exists(debug_path):
                print(f"✓ Debug output directory: {debug_path}")
                
                subdirs = ['images', 'segmented_images', 'metadata', 'reports']
                for subdir in subdirs:
                    subdir_path = os.path.join(debug_path, subdir)
                    if os.path.exists(subdir_path):
                        files = os.listdir(subdir_path)
                        print(f"  📁 {subdir}/: {len(files)} files")
                        for file in sorted(files):
                            file_path = os.path.join(subdir_path, file)
                            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                            print(f"    📄 {file} ({file_size:,} bytes)")
            else:
                print("❌ Debug output directory not found")
        else:
            print("✅ Standard processing completed (no debug outputs)")
        
        # Check standard outputs
        output_path = os.path.join(data_path, "output")
        if os.path.exists(output_path):
            output_files = os.listdir(output_path)
            print(f"✓ Standard output directory: {output_path}")
            print(f"  📁 output/: {len(output_files)} files")
            for file in sorted(output_files):
                file_path = os.path.join(output_path, file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"    📄 {file} ({file_size:,} bytes)")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 60)
    print("🎉 CLI debug test completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)