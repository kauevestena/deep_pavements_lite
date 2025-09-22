#!/usr/bin/env python3
"""
Test script for debug mode functionality.
"""

import os
import geopandas as gpd
from shapely.geometry import Point
from lib import process_images
from constants import data_path

def test_debug_mode():
    """Test debug mode functionality"""
    print("🔧 Testing Debug Mode")
    print("=" * 50)
    
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
        'id': ['test_debug'],
        'file_path': [test_image_path],
        'geometry': [Point(-122.4194, 37.7749)]  # San Francisco coordinates
    }
    
    test_gdf = gpd.GeoDataFrame(test_data, crs='EPSG:4326')
    print(f"✓ Created test GeoDataFrame with {len(test_gdf)} image(s)")
    
    # Test with debug mode enabled
    print("🔄 Processing with debug mode enabled...")
    try:
        result_gdf = process_images(test_gdf, data_path, debug_mode=True)
        
        print("✓ Debug mode processing completed")
        
        # Check if debug outputs were created
        debug_path = os.path.join(data_path, "debug_outputs")
        if os.path.exists(debug_path):
            print(f"✓ Debug output directory created: {debug_path}")
            
            # Check subdirectories
            subdirs = ['images', 'segmented_images', 'metadata', 'reports']
            for subdir in subdirs:
                subdir_path = os.path.join(debug_path, subdir)
                if os.path.exists(subdir_path):
                    files = os.listdir(subdir_path)
                    print(f"  ✓ {subdir}/: {len(files)} files")
                    for file in files:
                        file_path = os.path.join(subdir_path, file)
                        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                        print(f"    - {file} ({file_size} bytes)")
                else:
                    print(f"  ⚠ {subdir}/: directory not found")
        else:
            print("❌ Debug output directory not created")
            
    except Exception as e:
        print(f"❌ Error during debug processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 50)
    print("🎉 Debug mode test completed!")
    return 0

if __name__ == "__main__":
    exit_code = test_debug_mode()
    exit(exit_code)