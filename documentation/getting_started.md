# Getting Started

Welcome to **Deep Pavements Lite**, an automated pipeline for pavement surface material classification from street-level imagery using CLIP and OneFormer.

---

## 📋 Requirements

- **Operating System:** Linux / macOS
- **Python:** version `3.10` or higher
- **Core Dependencies:** PyTorch, Torchvision, Transformers, GeoPandas, Shapely, OpenCV
- **Mapillary Access:** A Mapillary developer token (v4 API) is required to query metadata and download images.

---

## 🚀 Installation & Setup

Follow these steps to set up the environment and run your first analysis:

### 1. Initialize Submodules
Deep Pavements Lite uses the `my_mappilary_api` submodule for map data downloading. If you haven't done so, initialize it:
```bash
git submodule update --init --recursive
```

### 2. Configure Your API Token
Create a file named `mapillary_token` in the root of the project folder containing your Mapillary access token:
```bash
echo "YOUR_MAPILLARY_TOKEN" > mapillary_token
```
*Alternatively, you can export it to your environment:*
```bash
export MAPILLARY_API="YOUR_MAPILLARY_TOKEN"
```

### 3. Run the Automated Setup
Use the provided `setup.sh` script to configure the directories, download the fine-tuned CLIP model weights, and run the verification smoke test:
```bash
./setup.sh
```

---

## 🏃 Running the Curitiba Sample

To run the pipeline over the sample area in **Curitiba, Brazil** using the command line runner:

```bash
.venv/bin/python runner.py \
    --lat_min -25.452901 \
    --lon_min -49.270045 \
    --lat_max -25.448231 \
    --lon_max -49.256247 \
    --map \
    --debug
```

### Command Options:
- `--map`: Generates a global interactive Leaflet map of results inside `data/output/`.
- `--debug`: Saves intermediary step images (heuristic masks, crops) and generates a detailed step-by-step debug HTML report.
- `--half_res` / `--quarter_res`: Downscale images to save memory and CPU/GPU processing time.

---

## 📁 Output Structure

All outputs are saved to the `data/` directory by default:

```
data/
├── output/
│   ├── surface_classifications.geojson  # GeoJSON spatial feature dataset
│   ├── surface_classifications.gpkg     # GeoPackage database
│   ├── model_info.json                  # Metadata about model versions
│   └── global_map.html                  # Interactive Leaflet map (with --map)
└── debug_outputs/
    ├── reports/
    │   └── debug_report.html            # Standalone visual HTML report (with --debug)
    └── [image_id]_roads_0.png           # Intermediary cropped segments for CLIP
```
