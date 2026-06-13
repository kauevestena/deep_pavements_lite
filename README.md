[![Smoke Test](https://github.com/kauevestena/deep_pavements_lite/actions/workflows/smoke_test.yml/badge.svg)](https://github.com/kauevestena/deep_pavements_lite/actions/workflows/smoke_test.yml)

# deep_pavements_lite

Automated pavement surface classification from street-level imagery using deep learning.

This version of Deep Pavements is an all-in-one package ("batteries-included"), using a fine-tuned **CLIP** model for surface material classification and **OneFormer** for semantic segmentation of roads and sidewalks.

## Architecture

```
Input (bounding box) → Mapillary API → Download Images
    → OneFormer Segmentation (roads, sidewalks, cars)
    → Polygon Extraction (contour detection)
    → CLIP Classification (11 surface types)
    → GeoJSON / GeoPackage / Interactive Map
```

### Package Structure

```
deep_pavements/
├── __init__.py         # Public API
├── constants.py        # Configuration and labels
├── models.py           # CLIP + OneFormer model loading
├── segmentation.py     # Semantic segmentation
├── classification.py   # Surface material classification
├── geometry.py         # Polygon extraction, road axis, regions
├── debug.py            # Debug outputs and HTML reports
├── pipeline.py         # Main processing orchestration
└── visualization.py    # Interactive Leaflet map generation
```

### Surface Types

The model classifies 11 surface material types: `asphalt`, `concrete`, `concrete_plates`, `grass`, `ground`, `sett`, `paving_stones`, `cobblestone`, `gravel`, `sand`, `compacted`.

## Local Setup

1) Clone the repository:
```bash
git clone https://github.com/kauevestena/deep_pavements_lite
cd deep_pavements_lite
git submodule update --init --recursive
```

2) Install dependencies:
```bash
pip install -r requirements.txt
pip install -r my_mappilary_api/requirements.txt
```

For development (includes testing and linting tools):
```bash
pip install -e ".[dev]"
```

3) Supply your Mapillary token (choose one):
```bash
# Option 1: Pass directly via CLI
python runner.py ... --mapillary_token "YOUR_TOKEN"

# Option 2: Environment variable
export MAPILLARY_API="YOUR_TOKEN"

# Option 3: Token file
echo "YOUR_TOKEN" > mapillary_token
```

> ⚠️ **Security**: Never commit your Mapillary token to the repository. The `mapillary_token` file is gitignored by default.

4) Run the application:
```bash
python runner.py \
    --lat_min <min_latitude> \
    --lon_min <min_longitude> \
    --lat_max <max_latitude> \
    --lon_max <max_longitude> \
    [--mapillary_token "<token>"] \
    [--max_images <count>] \
    [--half_res | --quarter_res] \
    [--debug] \
    [--map] \
    [--workers <n>]
```

### CLI Flags

| Flag | Description |
|---|---|
| `--mapillary_token` | Mapillary access token (alternative to env var or file) |
| `--max_images` | Randomly sample up to N images from metadata |
| `--half_res` / `--quarter_res` | Downscale images to 50% or 25% resolution |
| `--debug` | Save intermediary results (segmentation masks, overlays, HTML report) |
| `--map` | Generate an interactive Leaflet map of results |
| `--workers` | Number of parallel workers for I/O operations |
| `-o` / `--output_dir` | Output directory (default: `data/`) |

### Outputs

- `surface_classifications.geojson` — GeoJSON with per-image surface types
- `surface_classifications.gpkg` — GeoPackage format (same data)
- `surface_map.html` — Interactive map (when `--map` is used)
- `model_info.json` — Model version metadata
- Per-image segment JSONs and processed PNGs

## Docker Setup

1) Clone and setup:
```bash
git clone https://github.com/kauevestena/deep_pavements_lite
cd deep_pavements_lite
git submodule update --init --recursive
```

2) Build the Docker image:
```bash
docker build --tag 'deep_pavements_lite' .
```

To skip precaching of weights, add `--build-arg TO_PRECACHE=false`.

3) Run the Docker container:
```bash
MOUNT_FOLDER="$HOME/data/deep_pavements_lite"
mkdir -p $MOUNT_FOLDER
echo "<YOUR MAPILLARY TOKEN>" > $MOUNT_FOLDER/mapillary_token

docker run --name running_d_p_l \
    -v $MOUNT_FOLDER:/workspace/data \
    --gpus all \
    -it 'deep_pavements_lite' \
    python runner.py \
        --lat_min <min_latitude> \
        --lon_min <min_longitude> \
        --lat_max <max_latitude> \
        --lon_max <max_longitude> \
        [--mapillary_token "<token>"] \
        [--max_images <count>] \
        [--half_res | --quarter_res] \
        [--debug] \
        [--map]
```

(Or use it inside VSCode as a dev container — see `.devcontainer/devcontainer.json`.)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=deep_pavements --cov-report=term-missing

# Lint
ruff check deep_pavements/ tests/

# Type check
mypy deep_pavements/ --ignore-missing-imports
```

## License

MIT — see [LICENSE](LICENSE) for details.
