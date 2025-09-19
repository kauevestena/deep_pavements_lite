import sys
sys.path.append('.')
import clip
import torch
import os
from constants import DEVICE, clip_model_path

# Load CLIP model using the same pattern as lib.py
device = torch.device(DEVICE)

try:
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Try to load fine-tuned model if available
    # This is the model from: https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials
    if os.path.exists(clip_model_path):
        print(f"Loading fine-tuned model from {clip_model_path}")
        model.load_state_dict(torch.load(clip_model_path, map_location=device))
        print("✓ Fine-tuned CLIP model loaded successfully")
    else:
        print(f"Fine-tuned model {clip_model_path} not found, using default CLIP model")
        print("✓ Default CLIP model loaded successfully")
    
    model.eval()
    
except Exception as e:
    print(f"Error: Could not load CLIP model: {e}")
    raise

