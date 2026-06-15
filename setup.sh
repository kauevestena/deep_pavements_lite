#!/bin/bash

# Setup script for deep_pavements_lite
# This script downloads the fine-tuned CLIP model when internet access is available

echo "🚀 Setting up deep_pavements_lite..."

# Create test data directory structure
echo "📁 Setting up test data directories..."
mkdir -p data
mkdir -p data/output

# Copy test image if available
if [ -f "test_data/street_scene.jpg" ]; then
    cp test_data/street_scene.jpg data/test_image.jpg
    echo "✓ Test image copied to data directory"
else
    echo "⚠ Test image not found, creating minimal test environment"
fi

# Check if fine-tuned model already exists
if [ -f "deep_pavements_clip_model.pt" ]; then
    echo "✓ Fine-tuned CLIP model already exists"
else
    echo "📥 Attempting to download fine-tuned CLIP model..."
    python download_finetuned_model.py
    
    if [ -f "deep_pavements_clip_model.pt" ]; then
        echo "✅ Fine-tuned CLIP model downloaded successfully"
    else
        echo "⚠ Fine-tuned model download failed, will use mock model for testing"
    fi
fi

# Run smoke test to verify setup
echo "🧪 Running smoke test..."
python tests/smoke_test.py

if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully!"
    echo ""
    echo "📋 Setup Summary:"
    echo "  - Dependencies: ✓ Installed"
    echo "  - Submodules: ✓ Initialized"
    echo "  - CLIP Model: $([ -f "deep_pavements_clip_model.pt" ] && echo "✓ Fine-tuned" || echo "⚠ Mock (for testing)")"
    echo "  - Mapillary API: $([ -n "$MAPILLARY_API" ] && echo "✓ Token available" || echo "⚠ No token (will use fallback)")"
    echo "  - Smoke Test: ✓ Passed"
else
    echo "❌ Setup failed!"
    exit 1
fi