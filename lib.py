from constants import *
from my_mappilary_api.mapillary_api import *
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from skimage import measure

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

            # Segment the image using OneFormer and analyze surface types
            segmentation_result = segment_image_and_classify_surfaces(
                image, model, preprocess, device, filename
            )

        # Save output
        output_filename = filename.replace(ext_in, ext_out)
        output_filepath = os.path.join(output_path, output_filename)
        
        # Save the original image (or processed version)
        image.save(output_filepath)

    print("Image processing complete.")


def segment_image_and_classify_surfaces(image, clip_model, clip_preprocess, device, filename):
    """
    Segment image using OneFormer and classify surface types using CLIP.
    Returns a dictionary with encoded polygons and surface classifications.
    """
    try:
        # Initialize OneFormer (load lazily to avoid startup overhead)
        if not hasattr(segment_image_and_classify_surfaces, 'oneformer_processor'):
            print("Loading OneFormer model...")
            segment_image_and_classify_surfaces.oneformer_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large"
            )
            segment_image_and_classify_surfaces.oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large"
            ).to(device)
            segment_image_and_classify_surfaces.oneformer_model.eval()
            print("✓ OneFormer model loaded successfully")
        
        processor = segment_image_and_classify_surfaces.oneformer_processor
        oneformer_model = segment_image_and_classify_surfaces.oneformer_model
        
        # Cityscapes classes mapping to our pathway categories
        # Based on cityscapes dataset: road=0, sidewalk=1 (approximate indices)
        pathway_class_mapping = {
            'roads': [0],  # 'road' class in cityscapes
            'sidewalks': [1]  # 'sidewalk' class in cityscapes
        }
        
        # Prepare image for OneFormer
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get segmentation
        with torch.no_grad():
            outputs = oneformer_model(**inputs)
        
        # Get predicted segmentation mask
        predicted_semantic_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        
        # Convert to numpy for processing
        segmentation_mask = predicted_semantic_map.cpu().numpy()
        
        results = {
            'filename': filename,
            'image_size': image.size,
            'pathway_segments': []
        }
        
        # Process each pathway category
        for category_name in pathway_categories:
            if category_name in pathway_class_mapping:
                class_ids = pathway_class_mapping[category_name]
                
                # Create mask for this pathway category
                category_mask = np.zeros_like(segmentation_mask, dtype=bool)
                for class_id in class_ids:
                    category_mask |= (segmentation_mask == class_id)
                
                if np.any(category_mask):
                    # Find contours/polygons for this category
                    polygons = extract_polygons_from_mask(category_mask)
                    
                    for i, polygon in enumerate(polygons):
                        if len(polygon) > 6:  # At least 3 points (6 coordinates)
                            # Extract region from image for surface classification
                            surface_type = classify_surface_type(
                                image, polygon, clip_model, clip_preprocess, device
                            )
                            
                            segment_info = {
                                'category': category_name,
                                'polygon': polygon.tolist(),
                                'surface_type': surface_type,
                                'segment_id': f"{category_name}_{i}"
                            }
                            results['pathway_segments'].append(segment_info)
        
        # Save results as JSON
        output_json_filename = os.path.splitext(filename)[0] + '_segments.json'
        output_json_path = os.path.join(output_path, output_json_filename)
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved segmentation results to {output_json_path}")
        return results
        
    except Exception as e:
        print(f"Warning: Could not perform segmentation and classification: {e}")
        return {
            'filename': filename,
            'image_size': image.size,
            'pathway_segments': [],
            'error': str(e)
        }


def extract_polygons_from_mask(mask):
    """Extract polygon contours from a binary mask."""
    polygons = []
    
    # Find contours using scikit-image
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    
    for contour in contours:
        # Simplify contour to reduce points while preserving shape
        simplified = measure.approximate_polygon(contour, tolerance=2.0)
        
        # Convert to flat coordinate list [x1, y1, x2, y2, ...]
        if len(simplified) >= 3:  # At least 3 points for a valid polygon
            # Swap coordinates (row, col) -> (x, y) and flatten
            polygon = simplified[:, [1, 0]].flatten()
            polygons.append(polygon)
    
    return polygons


def classify_surface_type(image, polygon, clip_model, clip_preprocess, device):
    """
    Classify surface type of a polygonal region using CLIP.
    """
    try:
        # Convert polygon to mask
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create mask from polygon
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Convert polygon coords to list of tuples
        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, fill=255)
        
        # Apply mask to image
        masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)
        
        # Crop to bounding box to focus on the region
        bbox = mask.getbbox()
        if bbox:
            cropped_image = masked_image.crop(bbox)
        else:
            cropped_image = masked_image
        
        # Resize if too small for effective classification
        if cropped_image.size[0] < 32 or cropped_image.size[1] < 32:
            cropped_image = cropped_image.resize((224, 224))
        
        # Prepare prompts for surface classification
        surface_prompts = [f"a photo of {surface} surface" for surface in default_surfaces]
        
        # Encode texts and image
        text_tokens = clip.tokenize(surface_prompts).to(device)
        image_input = clip_preprocess(cropped_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            
            # Compute similarities
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            
            # Get best match
            best_match_idx = similarities.argmax().item()
            confidence = similarities[0, best_match_idx].item()
            
            return {
                'surface': default_surfaces[best_match_idx],
                'confidence': float(confidence)
            }
    
    except Exception as e:
        print(f"Warning: Could not classify surface type: {e}")
        return {
            'surface': 'unknown',
            'confidence': 0.0
        }
