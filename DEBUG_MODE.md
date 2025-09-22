# Debug Mode Documentation

## Overview

Deep Pavements Lite now supports a comprehensive debug mode that saves all intermediary results and generates detailed reports for analysis and troubleshooting.

## Usage

### CLI Usage

Add the `--debug` flag to any of the CLI tools:

```bash
# Main runner with debug mode
python runner.py --lat_min 37.7 --lon_min -122.5 --lat_max 37.8 --lon_max -122.4 --debug

# Test runner with debug mode  
python test_runner.py --lat_min 37.7 --lon_min -122.5 --lat_max 37.8 --lon_max -122.4 --debug

# Debug mode is optional - without the flag, only standard outputs are generated
python runner.py --lat_min 37.7 --lon_min -122.5 --lat_max 37.8 --lon_max -122.4
```

### API Usage

When using the library programmatically:

```python
from lib import process_images
import geopandas as gpd

# Enable debug mode
result_gdf = process_images(input_gdf, data_path, debug_mode=True)

# Standard mode (default)
result_gdf = process_images(input_gdf, data_path, debug_mode=False)
```

## Debug Output Structure

When debug mode is enabled, a `debug_outputs` folder is created in the data directory with the following structure:

```
data/
├── debug_outputs/
│   ├── images/                    # Original images
│   │   ├── image_id_1.jpg
│   │   └── image_id_2.jpg
│   ├── segmented_images/          # Segmented images with color overlays
│   │   ├── image_id_1_segmented.png
│   │   └── image_id_2_segmented.png
│   ├── metadata/                  # Metadata and configuration files
│   │   ├── image_id_1_metadata.json
│   │   ├── image_id_1_mask.npy
│   │   ├── image_id_1_color_encoding.json
│   │   └── ...
│   └── reports/                   # HTML reports
│       └── debug_report.html
└── output/                        # Standard outputs (always generated)
    ├── image_id_1.png
    ├── image_id_1_segments.json
    └── surface_classifications.geojson
```

## Debug Output Details

### 1. Original Images (`debug_outputs/images/`)
- Copies of all processed images
- Preserved in original format and quality
- Named using the image ID from Mapillary

### 2. Segmented Images (`debug_outputs/segmented_images/`)
- Visualizations of semantic segmentation results
- Color overlays showing detected roads, sidewalks, and cars:
  - **Red**: Roads
  - **Green**: Sidewalks  
  - **Blue**: Cars
- Saved as PNG files for easy viewing

### 3. Metadata Files (`debug_outputs/metadata/`)

#### Image Metadata (`*_metadata.json`)
```json
{
  "image_id": "123456789",
  "filename": "123456789.jpg",
  "original_size": [2048, 1536],
  "coordinates": "-122.4194, 37.7749",
  "file_path": "data/123456789.jpg"
}
```

#### Segmentation Masks (`*_mask.npy`)
- Raw segmentation masks as NumPy arrays
- Contains class IDs for each pixel
- Can be loaded with `np.load()` for further analysis

#### Color Encoding (`*_color_encoding.json`)
```json
{
  "classes": {
    "0": {
      "name": "road",
      "color": [128, 64, 128],
      "description": "Road surface including asphalt and concrete roads"
    },
    "1": {
      "name": "sidewalk",
      "color": [244, 35, 232],
      "description": "Sidewalk areas for pedestrian walking"
    },
    "13": {
      "name": "car",
      "color": [0, 0, 142],
      "description": "Cars and other vehicles"
    }
  },
  "legend": {
    "road": "Red overlay in segmentation images",
    "sidewalk": "Green overlay in segmentation images",
    "car": "Blue overlay in segmentation images"
  }
}
```

### 4. HTML Report (`debug_outputs/reports/debug_report.html`)

A comprehensive HTML report containing:

- **Processing Summary**: Total images processed, detection statistics
- **Per-Image Results**: 
  - Surface classification results (road, left sidewalk, right sidewalk)
  - GPS coordinates
  - Detected segments with confidence scores
  - CLIP classification probabilities
- **Metadata Tables**: Detailed information for each processed image
- **Segment Analysis**: List of all detected road/sidewalk segments

The report is styled with responsive CSS and can be opened in any web browser.

## CLIP Probabilities and Ratios

The debug mode captures and reports:

1. **CLIP Classification Confidence**: How confident the model is about each surface type
2. **Surface Type Ratios**: The relative proportions of different surface materials
3. **Segmentation Quality Metrics**: Area ratios between different detected segments

These are included in the HTML report and can be used to:
- Evaluate model performance
- Identify problematic classifications
- Tune confidence thresholds
- Analyze regional surface patterns

## Performance Considerations

- Debug mode increases processing time due to additional file I/O
- Disk space usage increases significantly (images, masks, and reports)
- Enable debug mode only when needed for analysis or troubleshooting
- Consider processing smaller batches when debug mode is enabled

## Troubleshooting

If debug outputs are not generated:
1. Check that the data directory is writable
2. Ensure sufficient disk space is available
3. Verify that the debug flag is properly passed through the call chain
4. Check console output for any error messages during debug file creation

## Integration with Existing Workflow

Debug mode is fully backward compatible:
- Existing scripts work unchanged when debug mode is not enabled
- Standard outputs are always generated regardless of debug mode
- Debug outputs are additional, not replacements for standard outputs