#!/usr/bin/env python3
"""
Download the fine-tuned CLIP model from HuggingFace
"""

import os
import torch
from huggingface_hub import hf_hub_download
from constants import clip_model_path

def download_finetuned_clip_model():
    """
    Download the fine-tuned CLIP model from HuggingFace Hub
    """
    model_repo = "kauevestena/clip-vit-base-patch32-finetuned-surface-materials"
    
    print(f"Downloading fine-tuned CLIP model from {model_repo}...")
    
    # First try HuggingFace Hub download
    try:
        # Download the PyTorch model file
        downloaded_file = hf_hub_download(
            repo_id=model_repo,
            filename="pytorch_model.bin",
            cache_dir="./cache"
        )
        
        print(f"Downloaded model to: {downloaded_file}")
        
        # Copy to expected location
        import shutil
        shutil.copy2(downloaded_file, clip_model_path)
        
        print(f"✓ Fine-tuned CLIP model saved to: {clip_model_path}")
        
        # Verify the file
        if os.path.exists(clip_model_path):
            file_size = os.path.getsize(clip_model_path) / (1024*1024)
            print(f"✓ Model file size: {file_size:.1f} MB")
            return True
        else:
            print("❌ Model file not found after download")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download model via HuggingFace Hub: {e}")
        print("Trying direct download fallback...")
        
        # Fallback to direct download from HuggingFace
        try:
            import requests
            
            direct_url = "https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials/resolve/main/model.pt"
            print(f"Downloading from direct URL: {direct_url}")
            
            response = requests.get(direct_url, stream=True)
            response.raise_for_status()
            
            with open(clip_model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if os.path.exists(clip_model_path):
                file_size = os.path.getsize(clip_model_path) / (1024*1024)
                print(f"✓ Fine-tuned CLIP model saved via direct URL: {clip_model_path}")
                print(f"✓ Model file size: {file_size:.1f} MB")
                return True
            else:
                print("❌ Model file not found after direct download")
                return False
                
        except Exception as direct_e:
            print(f"❌ Direct download also failed: {direct_e}")
            print("This may be due to network restrictions.")
            print("Continuing with default CLIP model...")
            return False

if __name__ == "__main__":
    download_finetuned_clip_model()