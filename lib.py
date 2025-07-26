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
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.load_state_dict(torch.load(clip_model_path, map_location=device))
    model.eval()

    # Create output directory
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)

    # Process images
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith(ext_in):
            # Open image
            image_path = os.path.join(data_path, filename)
            image = Image.open(image_path)

            # Preprocess image
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Get image features
            with torch.no_grad():
                image_features = model.encode_image(image_input)

            # TODO: Add logic to classify and segment the image

            # Save output
            output_filename = filename.replace(ext_in, ext_out)
            output_filepath = os.path.join(output_path, output_filename)
            # For now, just save the original image
            image.save(output_filepath)

    print("Image processing complete.")
