# GitHub Actions Workflow Enhancement

To integrate the enhanced smoke test with proper MAPILLARY_API support and fine-tuned CLIP model, update your GitHub Actions workflow as follows:

## Add Environment Variable

Add the MAPILLARY_API secret to your workflow:

```yaml
env:
  MAPILLARY_API: ${{ secrets.MAPILLARY_API }}
```

## Updated Workflow Steps

Replace the current smoke test step with:

```yaml
- name: Setup and run enhanced smoke test
  env:
    MAPILLARY_API: ${{ secrets.MAPILLARY_API }}
  run: |
    # Initialize submodules
    git submodule update --init --recursive
    
    # Install dependencies
    pip install -r requirements.txt
    pip install git+https://github.com/openai/CLIP.git
    
    # Run setup script which attempts to download fine-tuned model and runs smoke test
    chmod +x setup.sh
    ./setup.sh
```

## Benefits

1. **Real API Testing**: Uses actual Mapillary API when MAPILLARY_API secret is available
2. **Fine-tuned Model**: Attempts to download the fine-tuned CLIP model for better surface classification
3. **Robust Fallbacks**: Works even when network access is limited (uses mock models and static test data)
4. **Better Reporting**: Provides detailed setup summary showing what worked and what fell back to mock/dummy data

## Secret Configuration

In your GitHub repository settings, add:
- **Secret Name**: `MAPILLARY_API`
- **Secret Value**: Your actual Mapillary API token

This ensures real API testing in CI while maintaining local development capability with dummy/mock data.