# Contributing to Deep Pavements Lite

Thank you for your interest in contributing to Deep Pavements Lite! This guide will help you get started with development, understand the codebase, and submit your contributions.

## Development Environment Setup

### Prerequisites

- Python 3.10 or later
- Git
- CUDA-compatible GPU (recommended for model inference)

### Setting Up

1. **Clone the repository** (with submodules):

   ```bash
   git clone --recurse-submodules https://github.com/kauevestena/deep_pavements_lite.git
   cd deep_pavements_lite
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or: .venv\Scripts\activate  # Windows
   ```

3. **Install the package with dev dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

   This installs the project in editable mode along with development tools (pytest, ruff, mypy).

4. **Install submodule dependencies**:

   ```bash
   pip install -r my_mappilary_api/requirements.txt
   ```

5. **Download the fine-tuned model** (optional, for full pipeline testing):

   ```bash
   python download_finetuned_model.py
   ```

## Running Tests

We use [pytest](https://docs.pytest.org/) for testing. Tests are located in the `tests/` directory.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_models.py

# Run with coverage report
pytest --cov=deep_pavements --cov-report=term-missing

# Run with coverage HTML report
pytest --cov=deep_pavements --cov-report=html
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, configured with a line length of 120 characters targeting Python 3.10.

```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .

# Check formatting without modifying files
ruff format --check .
```

## Type Checking

We use [mypy](https://mypy.readthedocs.io/) for static type analysis.

```bash
# Run type checking
mypy deep_pavements/

# Run on a specific file
mypy deep_pavements/models.py
```

Configuration is in `pyproject.toml`. We have `ignore_missing_imports = true` set since some geospatial and ML libraries lack complete type stubs.

## Module Architecture

The `deep_pavements/` package is organized into focused modules:

```
deep_pavements/
├── __init__.py
├── constants.py        — Configuration and constants
├── models.py           — Model loading, downloading, caching
├── segmentation.py     — OneFormer + heuristic segmentation
├── classification.py   — CLIP surface classification
├── geometry.py         — Polygon extraction, road axis, sidewalk regions
├── debug.py            — Debug outputs and HTML reporting
├── pipeline.py         — Main processing orchestration
└── visualization.py    — Interactive map generation
```

### Module Descriptions

| Module | Responsibility |
|---|---|
| **`constants.py`** | Central location for configuration values, surface type labels, color maps, model paths, and other constants used throughout the pipeline. |
| **`models.py`** | Handles downloading, caching, and loading of both the CLIP (fine-tuned) model and the OneFormer segmentation model. Manages GPU/CPU device selection. |
| **`segmentation.py`** | Performs semantic segmentation using OneFormer to identify road and sidewalk regions in street-level images. Includes heuristic post-processing for segment refinement. |
| **`classification.py`** | Uses CLIP (or the fine-tuned variant) to classify pavement surface materials (asphalt, concrete, paving stones, etc.) from image crops. |
| **`geometry.py`** | Extracts geographic polygons from segmentation masks, computes road axes, and determines sidewalk regions using spatial operations with Shapely and GeoPandas. |
| **`debug.py`** | Generates debug visualizations, overlay images, and HTML reports for inspecting intermediate pipeline results. |
| **`pipeline.py`** | Orchestrates the full processing pipeline: image acquisition → segmentation → classification → geometry extraction → output generation. Contains the `main()` entry point. |
| **`visualization.py`** | Creates interactive maps (e.g., Folium/Leaflet-based) for visualizing classification results overlaid on geographic data. |

### Data Flow

```
Street-level Images
        │
        ▼
  ┌─────────────┐
  │ segmentation │  ← OneFormer model
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │classification │  ← CLIP model (fine-tuned)
  └──────┬───────┘
         │
         ▼
  ┌──────────┐
  │ geometry  │  ← Polygon extraction, spatial ops
  └─────┬────┘
        │
        ▼
  ┌───────────────┐
  │ visualization  │  ← Interactive map output
  └───────────────┘
```

## How to Add New Surface Types

Surface types (e.g., asphalt, concrete, paving stones) are defined in `constants.py`. To add a new surface type:

1. **Add the label** to the surface type list in `constants.py`:

   ```python
   SURFACE_TYPES = [
       "asphalt",
       "concrete",
       "paving_stones",
       "your_new_surface",  # Add here
   ]
   ```

2. **Add a color mapping** (if applicable) for visualization:

   ```python
   SURFACE_COLORS = {
       # ... existing colors ...
       "your_new_surface": "#RRGGBB",
   }
   ```

3. **Update CLIP prompts** in `classification.py` — the text prompts used for zero-shot classification should include descriptions of the new surface type:

   ```python
   SURFACE_PROMPTS = [
       # ... existing prompts ...
       "a photo of a your_new_surface road surface",
   ]
   ```

4. **Add tests** for the new surface type in the test suite.

5. **Update documentation** if the new surface type changes the output schema.

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/kauevestena/deep_pavements_lite/issues) to report bugs or request features.
- Include the Python version, OS, GPU info, and relevant logs.
- For bugs, include a minimal reproducible example if possible.

### Submitting Pull Requests

1. **Fork** the repository and create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**, following the code style guidelines above.

3. **Add or update tests** for any new functionality.

4. **Run the full checks** before submitting:

   ```bash
   ruff check .
   ruff format --check .
   mypy deep_pavements/
   pytest
   ```

5. **Commit** with clear, descriptive messages:

   ```bash
   git commit -m "Add support for cobblestone surface classification"
   ```

6. **Push** your branch and open a Pull Request against `main`.

7. In your PR description:
   - Explain **what** your change does and **why**.
   - Link any related issues (e.g., `Fixes #42`).
   - Include before/after screenshots for visual changes.

### PR Review Process

- A maintainer will review your PR, possibly requesting changes.
- All checks (lint, type checking, tests) must pass.
- Once approved, the PR will be squash-merged into `main`.

## Questions?

If you have questions about the codebase or need help getting started, feel free to open a [Discussion](https://github.com/kauevestena/deep_pavements_lite/discussions) or reach out via Issues.
