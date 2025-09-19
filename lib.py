from constants import *
from my_mappilary_api.mapillary_api import *
import torch
import clip
from PIL import Image
import os
from tqdm import tqdm

def process_images(data_path):
    # Load CLIP model
    device = torch.device(DEVICE)
    
    try:
        print("Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Try to load fine-tuned model if available
        if os.path.exists(clip_model_path):
            print(f"Loading fine-tuned model from {clip_model_path}")
            model.load_state_dict(torch.load(clip_model_path, map_location=device))
        else:
            print(f"Fine-tuned model {clip_model_path} not found, using default CLIP model")
        
        model.eval()
        model_available = True
        print("✓ CLIP model loaded successfully")
        
    except Exception as e:
        print(f"Warning: Could not load CLIP model: {e}")
        print("Continuing without model loading (will copy images only)")
        model_available = False
        model = None
        preprocess = None

    # Create output directory
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)
    print(f"Created output directory: {output_path}")

    # Process images
    image_files = [f for f in os.listdir(data_path) if f.endswith(ext_in)]
    
    if not image_files:
        print(f"No {ext_in} files found in {data_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for filename in tqdm(image_files, desc="Processing images"):
        # Open image
        image_path = os.path.join(data_path, filename)
        image = Image.open(image_path)

        if model_available:
            # Preprocess image
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Get image features
            with torch.no_grad():
                image_features = model.encode_image(image_input)

            # TODO: Add logic to classify and segment the image
            # For now, we just extract features but don't use them

        # Save output
        output_filename = filename.replace(ext_in, ext_out)
        output_filepath = os.path.join(output_path, output_filename)
        # For now, just save the original image
        image.save(output_filepath)

    print("Image processing complete.")
