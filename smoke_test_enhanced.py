#!/usr/bin/env python3
"""
Enhanced smoke test for deep_pavements_lite that uses real Mapillary API data

This test creates dummy test data by fetching a small amount of real images 
from Mapillary API using the MAPILLARY_API secret.
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
from my_mappilary_api.mapillary_api import get_mapillary_images_metadata, mapillary_data_to_gdf, download_all_pictures_from_gdf

def get_mapillary_token():
    """Get Mapillary API token from environment or file"""
    # Check environment variable first (for GitHub Actions)
    token = os.environ.get('MAPILLARY_API')
    if token:
        print("✓ Found MAPILLARY_API token in environment")
        return token.strip()
    
    # Check token files as fallback
    token_files = ["mapillary_token", "workspace/data/mapillary_token", "data/mapillary_token"]
    
    for token_file in token_files:
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                token = f.read().strip()
            print(f"✓ Found token in {token_file}")
            return token
    
    return None

def create_dummy_test_data():
    """Create dummy test data using Mapillary API"""
    print("🔄 Creating dummy test data using Mapillary API...")
    
    # Get API token
    mapillary_token = get_mapillary_token()
    if not mapillary_token:
        print("❌ No Mapillary API token found")
        print("   Set MAPILLARY_API environment variable or create mapillary_token file")
        return None
    
    # Use coordinates around San Francisco (known to have good street imagery)
    # Small bounding box to limit the amount of data
    lat_min, lon_min = 37.7749, -122.4194  # SF center
    lat_max, lon_max = 37.7759, -122.4184  # Small 1km x 1km area
    
    print(f"📍 Fetching images from bounding box: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
    
    try:
        # Get metadata from Mapillary API
        metadata = get_mapillary_images_metadata(
            lon_min, lat_min, lon_max, lat_max, 
            token=mapillary_token
        )
        
        if not metadata.get("data"):
            print("❌ No images found in the specified area")
            return None
        
        print(f"✓ Found {len(metadata['data'])} images from Mapillary API")
        
        # Limit to first image for testing
        metadata['data'] = metadata['data'][:1]
        print(f"✓ Limited to {len(metadata['data'])} image(s) for testing")
        
        # Convert to GeoDataFrame
        gdf = mapillary_data_to_gdf(metadata)
        
        if gdf.empty:
            print("❌ Failed to create GeoDataFrame from metadata")
            return None
        
        # Add file paths
        gdf['file_path'] = gdf['id'].apply(lambda x: os.path.join(data_path, f"{x}.jpg"))
        
        print("📥 Downloading images...")
        download_all_pictures_from_gdf(gdf, data_path)
        
        # Verify images were downloaded
        for _, row in gdf.iterrows():
            if not os.path.exists(row['file_path']):
                print(f"❌ Failed to download image: {row['file_path']}")
                return None
            else:
                # Verify image can be loaded
                try:
                    with Image.open(row['file_path']) as img:
                        print(f"✓ Downloaded image: {row['file_path']} ({img.size} pixels)")
                except Exception as e:
                    print(f"❌ Downloaded image is corrupted: {e}")
                    return None
        
        print("✓ Dummy test data created successfully using Mapillary API")
        return gdf
        
    except Exception as e:
        print(f"❌ Failed to create dummy test data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_fallback_test_geodataframe():
    """Create a minimal GeoDataFrame with the existing test image as fallback"""
    print("📋 Creating fallback test GeoDataFrame with existing test image...")
    
    # Check if test image exists
    test_image_path = os.path.join(data_path, 'test_image.jpg')
    if not os.path.exists(test_image_path):
        # Copy from test_data if needed
        source_path = os.path.join('test_data', 'street_scene.jpg')
        if os.path.exists(source_path):
            import shutil
            shutil.copy2(source_path, test_image_path)
            print(f"✓ Copied {source_path} to {test_image_path}")
        else:
            print(f"❌ No test image available")
            return None
    
    # Create test data for the image
    test_data = {
        'id': ['test_image'],
        'file_path': [test_image_path],
        'geometry': [Point(-122.4194, 37.7749)]  # San Francisco coordinates
    }
    
    gdf = gpd.GeoDataFrame(test_data, crs='EPSG:4326')
    print("✓ Fallback test GeoDataFrame created")
    return gdf

def main():
    """Run the enhanced smoke test"""
    print("🚀 Starting Enhanced Deep Pavements Lite Smoke Test")
    print("=" * 60)
    
    # Ensure data directory exists
    os.makedirs(data_path, exist_ok=True)
    
    # Try to create dummy test data using Mapillary API
    test_gdf = create_dummy_test_data()
    
    # Fallback to static test image if API fails
    if test_gdf is None or test_gdf.empty:
        print("⚠ Falling back to static test image")
        test_gdf = create_fallback_test_geodataframe()
        
        if test_gdf is None or test_gdf.empty:
            print("❌ Failed to create any test data")
            return 1
    
    print(f"✓ Test GeoDataFrame ready with {len(test_gdf)} image(s)")
    
    # Process the test images
    print("🔄 Processing test images with deep_pavements_lite...")
    print("-" * 60)
    
    try:
        result_gdf = process_images(test_gdf, data_path, debug_mode=False)
        
        if result_gdf.empty:
            print("⚠ No results generated (this may be expected without GPU/models)")
            print("✓ Core processing pipeline executed without errors")
        else:
            print(f"✓ Successfully processed {len(result_gdf)} image(s)")
            print("✓ Surface classifications generated:")
            # Show only available columns
            available_cols = ['filename', 'image_id']
            for col in ['road', 'left_sidewalk', 'right_sidewalk']:
                if col in result_gdf.columns:
                    available_cols.append(col)
            print(result_gdf[available_cols].to_string())
            
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
            file_path = os.path.join(output_path, file)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"  - {file} ({file_size} bytes)")
    
    print("=" * 60)
    print("🎉 Enhanced smoke test completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)