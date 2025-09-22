# Smoke Test for Deep Pavements Lite

This document describes the smoke test implementation for the Deep Pavements Lite project.

## Overview

The smoke test is designed to verify that the core functionality of Deep Pavements Lite works correctly in a CPU-only environment, such as GitHub Actions CI. Since the full processing involves heavy inference models that require significant computational resources, the smoke test processes just a single test image to validate the pipeline.

## Test Structure

### Files

- **`.github/workflows/smoke_test.yml`**: GitHub Actions workflow configuration
- **`smoke_test.py`**: Python script that runs the smoke test
- **`test_data/street_scene.jpg`**: Test image included in the repository

### What the Test Does

1. **Environment Setup**: Installs Python dependencies including CPU-only PyTorch
2. **Image Loading**: Loads a pre-created test image that simulates a street scene
3. **Pipeline Testing**: Runs the core image processing pipeline
4. **Output Validation**: Checks that output files are generated correctly

### Expected Behavior

- The test should complete successfully even without GPU acceleration
- When ML models can't be downloaded (due to network restrictions), the pipeline gracefully degrades to copying images only
- Output files are generated in the `data/output/` directory
- The test validates the core processing workflow without requiring external API access

## Running the Test

### Locally

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r my_mappilary_api/requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Create data directory and copy test image
mkdir -p data
cp test_data/street_scene.jpg data/test_image.jpg

# Run the smoke test
python smoke_test.py
```

### In GitHub Actions

The test runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

## Limitations

- **CPU-only processing**: Models may not load or perform optimally without GPU
- **Network restrictions**: Some models require internet access for download
- **Single image**: Only tests with one image to keep CI time reasonable
- **No Mapillary API**: Uses a mock token and doesn't test actual image downloading

## Success Criteria

The smoke test passes if:

1. ✅ Python dependencies install successfully
2. ✅ Test image loads correctly
3. ✅ Core processing pipeline executes without errors
4. ✅ Output directory is created
5. ✅ At least one output file is generated

The test is designed to be robust and should pass even when ML models are not available, making it suitable for CI environments with limited resources.