import sys
sys.path.append('.')
import clip
import torch
import os
from constants import DEVICE, clip_model_path

def _try_download_finetuned_model():
    """
    Try to download the fine-tuned CLIP model from HuggingFace Hub.
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    # First try HuggingFace Hub download
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        
        print("Attempting to download fine-tuned CLIP model...")
        
        # Download the PyTorch model file
        downloaded_file = hf_hub_download(
            repo_id="kauevestena/clip-vit-base-patch32-finetuned-surface-materials",
            filename="pytorch_model.bin",
            cache_dir="./cache"
        )
        
        # Copy to expected location
        shutil.copy2(downloaded_file, clip_model_path)
        
        if os.path.exists(clip_model_path):
            file_size = os.path.getsize(clip_model_path) / (1024*1024)
            print(f"✓ Fine-tuned model downloaded successfully ({file_size:.1f} MB)")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Unable to download fine-tuned model via HuggingFace Hub: {e}")
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
                print(f"✓ Fine-tuned model downloaded successfully via direct URL ({file_size:.1f} MB)")
                return True
            else:
                return False
                
        except Exception as direct_e:
            print(f"Direct download also failed: {direct_e}")
            print("This may be due to network restrictions or missing dependencies.")
            return False

# Load CLIP model using the same pattern as lib.py
device = torch.device(DEVICE)

try:
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Try to load fine-tuned model if available
    # This is the model from: https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials
    if os.path.exists(clip_model_path):
        print(f"Loading fine-tuned model from {clip_model_path}")
        try:
            checkpoint = torch.load(clip_model_path, map_location=device)
            # Check if this is a checkpoint with metadata or just state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✓ Fine-tuned CLIP model loaded successfully")
        except Exception as load_e:
            print(f"Failed to load fine-tuned model: {load_e}")
            print("Using default CLIP model")
    else:
        print(f"Fine-tuned model {clip_model_path} not found")
        # Try to download the fine-tuned model
        if _try_download_finetuned_model():
            print(f"Loading downloaded fine-tuned model from {clip_model_path}")
            try:
                checkpoint = torch.load(clip_model_path, map_location=device)
                # Check if this is a checkpoint with metadata or just state dict
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("✓ Fine-tuned CLIP model downloaded and loaded successfully")
            except Exception as load_e:
                print(f"Failed to load downloaded model: {load_e}")
                print("Using default CLIP model")
        else:
            print("Using default CLIP model")
            print("✓ Default CLIP model loaded successfully")
    
    model.eval()
    
except Exception as e:
    print(f"Error: Could not load CLIP model: {e}")
    raise

