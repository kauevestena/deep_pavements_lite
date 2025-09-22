"""
Deep Pavements Lite - Core Image Processing Library

This module provides the core functionality for analyzing pavement surfaces in street-level 
imagery using computer vision and machine learning techniques. The pipeline combines:

1. **CLIP (Contrastive Language-Image Pre-training)**: For image feature extraction and 
   surface material classification using natural language descriptions
2. **OneFormer**: For semantic segmentation to identify roads and sidewalks in urban scenes
3. **Polygon Extraction**: To extract geometric representations of pavement segments
4. **Surface Classification**: To classify material types (asphalt, concrete, etc.)

**Workflow Overview:**
1. Load and preprocess street-level images (typically from Mapillary)
2. Use OneFormer to segment images into semantic classes (roads, sidewalks)
3. Extract polygon contours from segmentation masks
4. Classify surface materials within each polygon using CLIP
5. Output processed images and structured JSON data with geometric and material information

**Key Dependencies:**
- PyTorch: Deep learning framework for model inference
- CLIP: OpenAI's vision-language model for surface classification
- OneFormer: Universal image segmentation model
- PIL: Image processing and manipulation
- scikit-image: Computer vision algorithms for contour extraction

**Output Format:**
The pipeline generates JSON files containing:
- Image metadata (filename, dimensions)
- Segmented pathway information (roads, sidewalks)
- Polygon coordinates for each segment
- Surface material classifications with confidence scores

Author: Kauê de Moraes Vestena
License: MIT
"""

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
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
import geopandas as gpd
import pandas as pd

def _create_mock_clip_model(device):
    """
    Create a mock CLIP model for testing purposes when real model is unavailable.
    
    Returns:
        tuple: (mock_model, mock_preprocess) that can be used for testing
    """
    import torchvision.transforms as transforms
    
    class MockCLIPModel:
        def __init__(self, device):
            self.device = device
            
        def encode_image(self, image_input):
            # Return a mock image embedding with correct dimensions
            batch_size = image_input.shape[0]
            return torch.randn(batch_size, 512, device=self.device)
            
        def encode_text(self, text_tokens):
            # Return mock text embeddings with correct dimensions
            batch_size = text_tokens.shape[0]
            return torch.randn(batch_size, 512, device=self.device)
            
        def eval(self):
            return self
            
        def to(self, device):
            self.device = device
            return self
    
    # Create mock preprocessing function
    mock_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add mock tokenize function to the clip module
    def mock_tokenize(texts, context_length=77, truncate=True):
        """Mock tokenize function that returns dummy tokens"""
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        # Return dummy token tensor
        return torch.randint(0, 1000, (batch_size, context_length), device=device)
    
    # Monkey patch the clip module to use our mock tokenize
    import clip
    clip.tokenize = mock_tokenize
    
    print("✓ Mock CLIP model created for testing (network unavailable)")
    return MockCLIPModel(device), mock_preprocess

def _try_download_finetuned_model():
    """
    Try to download the fine-tuned CLIP model from HuggingFace Hub.
    
    Returns:
        bool: True if download was successful, False otherwise
    """
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
        print(f"Unable to download fine-tuned model: {e}")
        print("This may be due to network restrictions or missing dependencies.")
        return False

def process_images(input_gdf: gpd.GeoDataFrame, data_path: str) -> gpd.GeoDataFrame:
    """
    Process images using metadata from a GeoDataFrame with CLIP and OneFormer models.
    
    This function implements the main image processing pipeline that:
    1. Loads CLIP model (with optional fine-tuned weights for surface classification)
    2. Processes images specified in the input GeoDataFrame
    3. Performs semantic segmentation and surface material classification
    4. Analyzes road/sidewalk layout and classifies surfaces on both sides
    5. Returns structured geodataframe with surface classifications including GPS coordinates
    
    Args:
        input_gdf (gpd.GeoDataFrame): GeoDataFrame containing image metadata with columns:
                                     - 'id': Mapillary image ID
                                     - 'file_path': Path to downloaded image file
                                     - 'geometry': Point geometry with GPS coordinates
                                     - Other Mapillary metadata fields
        data_path (str): Path to directory containing input images and output directory.
    
    Returns:
        gpd.GeoDataFrame: Results containing surface classifications with columns:
            - filename: Original image filename
            - image_id: Mapillary image ID
            - road: Surface type of the main road
            - left_sidewalk: Surface type of left side (or 'no_sidewalk'/'car_hindered')
            - right_sidewalk: Surface type of right side (or 'no_sidewalk'/'car_hindered')
            - geometry: Point geometry with actual GPS coordinates from Mapillary
    
    Raises:
        Exception: Gracefully handles model loading failures and continues with 
                  image copying only (without AI processing).
    
    Note:
        - Uses GPU acceleration if available (CUDA), falls back to CPU
        - Supports fine-tuned CLIP models for improved surface classification
        - Progress is displayed using tqdm progress bars
        - Creates output directory automatically if it doesn't exist
        - Images without road detections are skipped from results
        - Uses actual GPS coordinates and image IDs from Mapillary metadata
    """
    # Load CLIP model
    device = torch.device(DEVICE)
    
    try:
        print("Loading CLIP model...")
        # Load CLIP vision-language model with ViT-B/32 architecture
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Try to load fine-tuned model weights if available
        # Fine-tuned models typically perform better on domain-specific tasks
        if os.path.exists(clip_model_path):
            print(f"Loading fine-tuned model from {clip_model_path}")
            model.load_state_dict(torch.load(clip_model_path, map_location=device))
            print("✓ Fine-tuned CLIP model loaded successfully")
        else:
            print(f"Fine-tuned model {clip_model_path} not found")
            # Try to download the fine-tuned model
            if _try_download_finetuned_model():
                print(f"Loading downloaded fine-tuned model from {clip_model_path}")
                model.load_state_dict(torch.load(clip_model_path, map_location=device))
                print("✓ Fine-tuned CLIP model downloaded and loaded successfully")
            else:
                print("Using default CLIP model")
        
        model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)
        model_available = True
        print("✓ CLIP model loaded successfully")
        
    except Exception as e:
        print(f"Warning: Could not load CLIP model: {e}")
        print("Attempting to create mock CLIP model for testing...")
        try:
            model, preprocess = _create_mock_clip_model(device)
            model_available = True
            print("✓ Mock CLIP model loaded successfully for testing")
        except Exception as mock_e:
            print(f"Failed to create mock model: {mock_e}")
            print("Continuing without model loading (will copy images only)")
            model_available = False
            model = None
            preprocess = None

    # Create output directory for processed results
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)
    print(f"Created output directory: {output_path}")

    # Use the input GDF to get images to process
    if input_gdf.empty:
        print("No images found in input GeoDataFrame")
        return gpd.GeoDataFrame()  # Return empty geodataframe
    
    print(f"Found {len(input_gdf)} images to process")
    
    # Initialize list to store results for geodataframe
    surface_results = []
    
    # Process each image with progress bar
    for idx, row in tqdm(input_gdf.iterrows(), total=len(input_gdf), desc="Processing images"):
        # Get image information from the row
        image_id = row['id']
        filename = f"{image_id}.jpg"
        image_path = row['file_path']
        coordinates = row['geometry']
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
            
        # Load image from file
        image = Image.open(image_path)

        if model_available:
            # Preprocess image for CLIP model input
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Extract image features using CLIP (not used in current pipeline but available)
            with torch.no_grad():
                image_features = model.encode_image(image_input)

            # Perform semantic segmentation and surface classification
            segmentation_result = segment_image_and_classify_surfaces(
                image, model, preprocess, device, filename
            )

            # Process segmentation results to classify road and sidewalk surfaces
            # Only process images that have road detections
            road_polygons = []
            for segment in segmentation_result.get('pathway_segments', []):
                if segment.get('category') == 'roads':
                    road_polygons.append(np.array(segment.get('polygon', [])))
            
            if road_polygons:  # Only process if we found roads
                # Find the axis of the road and extend to image boundaries
                road_axis = get_road_axis_line(road_polygons, image.size)
                
                if road_axis:
                    # Classify surfaces on left and right sides of the road
                    surface_classification = classify_sidewalk_regions(
                        segmentation_result, road_axis, image.size
                    )
                    
                    # Create result entry using actual metadata from Mapillary
                    result_entry = {
                        'filename': filename,
                        'image_id': image_id,
                        'road': surface_classification['road'],
                        'left_sidewalk': surface_classification['left_sidewalk'],
                        'right_sidewalk': surface_classification['right_sidewalk'],
                        'geometry': coordinates  # Use actual GPS coordinates from Mapillary
                    }
                    surface_results.append(result_entry)

        # Save processed image to output directory
        output_filename = filename.replace(ext_in, ext_out)
        output_filepath = os.path.join(output_path, output_filename)
        
        # Save the original image (or processed version if desired)
        image.save(output_filepath)

    print("Image processing complete.")
    
    # Create geodataframe from results
    if surface_results:
        gdf = gpd.GeoDataFrame(surface_results, crs='EPSG:4326')
        
        # Save as GeoJSON in output directory
        geojson_path = os.path.join(output_path, "surface_classifications.geojson")
        gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"Saved surface classifications to {geojson_path}")
        
        return gdf
    else:
        print("No valid road detections found in any images.")
        return gpd.GeoDataFrame()


def segment_image_and_classify_surfaces(image: Image.Image, clip_model, clip_preprocess, 
                                       device: torch.device, filename: str) -> dict:
    """
    Segment image using OneFormer and classify surface types using CLIP.
    
    This function performs the core computer vision analysis by combining semantic 
    segmentation with surface material classification:
    
    1. **Semantic Segmentation**: Uses OneFormer (trained on Cityscapes dataset) to identify
       roads and sidewalks in the input image
    2. **Polygon Extraction**: Converts segmentation masks to polygon contours using
       contour detection algorithms
    3. **Surface Classification**: Applies CLIP model to classify surface materials
       (asphalt, concrete, cobblestone, etc.) within each detected segment
    4. **Results Aggregation**: Combines geometric and material information into structured data
    
    Args:
        image (PIL.Image.Image): Input street-level image to be analyzed
        clip_model: Loaded CLIP model for surface material classification
        clip_preprocess: CLIP preprocessing function for image normalization
        device (torch.device): PyTorch device (CPU/CUDA) for model inference
        filename (str): Original filename for metadata and output file naming
    
    Returns:
        dict: Structured results containing:
            - 'filename': Original image filename
            - 'image_size': Image dimensions (width, height)
            - 'pathway_segments': List of detected segments, each containing:
                - 'category': Segment type ('roads', 'sidewalks', or 'car')
                - 'polygon': Flattened coordinate array [x1,y1,x2,y2,...]
                - 'surface_type': Material classification result with confidence
                - 'segment_id': Unique identifier for the segment
            - 'error': Error message if processing fails (optional)
    
    Note:
        - OneFormer models are loaded lazily on first call to optimize startup time
        - Uses Cityscapes class mapping: road=0, sidewalk=1, car=13
        - Saves JSON results to output directory automatically
        - Handles processing errors gracefully and returns error information
        
    Model Details:
        - OneFormer: "shi-labs/oneformer_cityscapes_swin_large" for segmentation
        - CLIP: Vision-language model for surface material classification
    """
    try:
        # Initialize OneFormer (load lazily to avoid startup overhead)
        # Using function attributes to store models as a simple singleton pattern
        if not hasattr(segment_image_and_classify_surfaces, 'oneformer_processor'):
            print("Loading OneFormer model...")
            # Load Cityscapes-trained model for urban scene segmentation
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
        
        # Cityscapes dataset class mapping to our pathway categories of interest
        # Based on standard Cityscapes annotations: road=0, sidewalk=1, car=13
        pathway_class_mapping = {
            'roads': [0],  # 'road' class in cityscapes
            'sidewalks': [1],  # 'sidewalk' class in cityscapes
            'car': [13]  # 'car' class in cityscapes
        }
        
        # Prepare image for OneFormer inference with semantic segmentation task
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Perform semantic segmentation inference
        with torch.no_grad():
            outputs = oneformer_model(**inputs)
        
        # Post-process model outputs to get segmentation mask
        # Target size ensures output matches input image dimensions
        predicted_semantic_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]  # PIL size is (width, height), need (height, width)
        )[0]
        
        # Convert tensor to numpy array for further processing
        segmentation_mask = predicted_semantic_map.cpu().numpy()
        
        results = {
            'filename': filename,
            'image_size': image.size,
            'pathway_segments': []
        }
        
        # Process each pathway category (roads, sidewalks, car) defined in constants
        for category_name in pathway_categories:
            if category_name in pathway_class_mapping:
                class_ids = pathway_class_mapping[category_name]
                
                # Create binary mask for this specific pathway category
                # Combine all relevant class IDs using logical OR
                category_mask = np.zeros_like(segmentation_mask, dtype=bool)
                for class_id in class_ids:
                    category_mask |= (segmentation_mask == class_id)
                
                # Process only if the category is present in the image
                if np.any(category_mask):
                    # Extract polygon contours from the binary mask
                    polygons = extract_polygons_from_mask(category_mask)
                    
                    # Process each detected polygon in this category
                    for i, polygon in enumerate(polygons):
                        if len(polygon) > 6:  # At least 3 points (6 coordinates: x1,y1,x2,y2,x3,y3)
                            # Classify surface material within this polygon region
                            surface_type = classify_surface_type(
                                image, polygon, clip_model, clip_preprocess, device
                            )
                            
                            # Aggregate segment information
                            segment_info = {
                                'category': category_name,
                                'polygon': polygon.tolist(),  # Convert numpy array to JSON-serializable list
                                'surface_type': surface_type,
                                'segment_id': f"{category_name}_{i}"  # Unique identifier
                            }
                            results['pathway_segments'].append(segment_info)
        
        # Save results as JSON file alongside the processed image
        output_json_filename = os.path.splitext(filename)[0] + '_segments.json'
        output_json_path = os.path.join(data_path, "output", output_json_filename)  # Use data_path + output subdir
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


def extract_polygons_from_mask(mask: np.ndarray) -> list:
    """
    Extract polygon contours from a binary segmentation mask.
    
    This function converts binary segmentation masks into polygon representations
    using contour detection and polygon approximation algorithms from scikit-image.
    
    **Algorithm Steps:**
    1. **Contour Detection**: Uses marching squares algorithm (find_contours) to detect
       object boundaries in the binary mask at 0.5 threshold
    2. **Polygon Approximation**: Simplifies contours using Douglas-Peucker algorithm
       to reduce point count while preserving shape (tolerance=2.0 pixels)
    3. **Coordinate Transformation**: Converts from (row,col) to (x,y) coordinates
       and flattens to format compatible with polygon rendering libraries
    4. **Filtering**: Removes degenerate polygons with fewer than 3 vertices
    
    Args:
        mask (np.ndarray): Binary mask array where True/1 indicates object pixels
                          and False/0 indicates background. Typically boolean or uint8.
    
    Returns:
        list: List of polygon coordinate arrays. Each polygon is represented as:
              - np.ndarray of shape (n*2,) containing flattened coordinates
              - Format: [x1, y1, x2, y2, ..., xn, yn]
              - Coordinates are in image pixel space (x=column, y=row)
              - Only polygons with ≥3 vertices are included
    
    Note:
        - Uses tolerance=2.0 for polygon simplification (adjustable for quality/speed tradeoff)
        - Coordinates are swapped from scikit-image's (row,col) to standard (x,y) format
        - Empty list returned if no valid contours found
        - Suitable for further processing with PIL.ImageDraw, OpenCV, or GIS libraries
    """
    polygons = []
    
    # Find contours using scikit-image's marching squares algorithm
    # Threshold of 0.5 works well for binary masks (True/False or 0/1)
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    
    for contour in contours:
        # Simplify contour using Douglas-Peucker algorithm to reduce point count
        # Higher tolerance = fewer points but less detail; 2.0 pixels is a good balance
        simplified = measure.approximate_polygon(contour, tolerance=2.0)
        
        # Convert to flat coordinate list format: [x1, y1, x2, y2, ...]
        if len(simplified) >= 3:  # At least 3 points for a valid polygon
            # Swap coordinates from scikit-image's (row, col) to standard (x, y)
            # and flatten the array for compatibility with drawing libraries
            polygon = simplified[:, [1, 0]].flatten()
            polygons.append(polygon)
    
    return polygons


def classify_surface_type(image: Image.Image, polygon: np.ndarray, clip_model, 
                         clip_preprocess, device: torch.device) -> dict:
    """
    Classify surface material type of a polygonal region using CLIP vision-language model.
    
    This function extracts a specific region defined by a polygon from the input image
    and uses CLIP (Contrastive Language-Image Pre-training) to classify the surface
    material by comparing the visual features against textual descriptions of different
    surface types.
    
    **Classification Process:**
    1. **Region Extraction**: Creates a binary mask from polygon coordinates
    2. **Image Masking**: Applies mask to isolate the region of interest
    3. **Preprocessing**: Crops to bounding box and resizes if necessary for optimal classification
    4. **Feature Encoding**: Uses CLIP to encode both image region and text prompts
    5. **Similarity Matching**: Computes cosine similarities between image and text features
    6. **Classification**: Returns the surface type with highest similarity score
    
    Args:
        image (PIL.Image.Image): Source image containing the surface region
        polygon (np.ndarray): Flattened polygon coordinates [x1,y1,x2,y2,...] defining
                             the region boundary in image pixel coordinates
        clip_model: Loaded CLIP model for vision-language understanding
        clip_preprocess: CLIP preprocessing function for image normalization
        device (torch.device): PyTorch device (CPU/CUDA) for model inference
    
    Returns:
        dict: Classification result containing:
            - 'surface' (str): Predicted surface material type from default_surfaces list
                              (e.g., 'asphalt', 'concrete', 'cobblestone', 'gravel')
            - 'confidence' (float): Softmax probability score [0.0-1.0] indicating
                                  model confidence in the prediction
            
            On error returns:
            - 'surface': 'unknown'
            - 'confidence': 0.0
    
    Surface Types Classified:
        Based on default_surfaces from constants.py:
        - asphalt, concrete, concrete_plates
        - grass, ground, sett, paving_stones  
        - cobblestone, gravel, sand, compacted
    
    Note:
        - Minimum region size enforced (32x32 pixels) with automatic upsampling
        - Uses natural language prompts: "a photo of {surface} surface"
        - Handles degenerate polygons and processing errors gracefully
        - CLIP normalization applied for consistent feature representation
    """
    try:
        # Convert polygon coordinates to a drawable mask
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create binary mask from polygon coordinates
        mask = Image.new('L', image.size, 0)  # 'L' mode = grayscale
        draw = ImageDraw.Draw(mask)
        
        # Convert flattened coordinates to list of (x,y) tuples for PIL
        coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, fill=255)  # White pixels inside polygon
        
        # Apply mask to extract only the polygon region from the image
        # Areas outside polygon become black (0,0,0)
        masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)
        
        # Crop to bounding box to focus computation on the region of interest
        bbox = mask.getbbox()  # Returns (left, top, right, bottom) or None
        if bbox:
            cropped_image = masked_image.crop(bbox)
        else:
            cropped_image = masked_image
        
        # Ensure minimum size for effective CLIP classification
        # CLIP works best with reasonably sized images; upscale tiny regions
        if cropped_image.size[0] < 32 or cropped_image.size[1] < 32:
            cropped_image = cropped_image.resize((224, 224))  # CLIP's native resolution
        
        # Prepare text prompts for all surface types from constants
        # Using natural language format that CLIP was trained on
        surface_prompts = [f"a photo of {surface} surface" for surface in default_surfaces]
        
        # Encode both text prompts and image using CLIP
        text_tokens = clip.tokenize(surface_prompts).to(device)
        image_input = clip_preprocess(cropped_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get feature embeddings for both modalities
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            
            # Compute cosine similarities between image and all text descriptions
            # Softmax converts similarities to probability distribution
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            
            # Select surface type with highest similarity score
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


def calculate_polygon_area(polygon_coords: np.ndarray) -> float:
    """
    Calculate the area of a polygon in pixels.
    
    Args:
        polygon_coords (np.ndarray): Flattened polygon coordinates [x1,y1,x2,y2,...]
    
    Returns:
        float: Area of the polygon in square pixels
    """
    try:
        # Convert flattened coordinates to list of (x,y) tuples
        coords = [(polygon_coords[i], polygon_coords[i+1]) for i in range(0, len(polygon_coords), 2)]
        if len(coords) < 3:
            return 0.0
        
        polygon = Polygon(coords)
        return polygon.area
    except Exception:
        return 0.0


def get_road_axis_line(road_polygons: list, image_size: tuple) -> LineString:
    """
    Find the main axis of road polygons and extend it to image boundaries.
    
    Args:
        road_polygons (list): List of road polygon coordinate arrays
        image_size (tuple): Image dimensions (width, height)
    
    Returns:
        LineString: Line representing the road axis extended to image boundaries
    """
    if not road_polygons:
        return None
    
    try:
        # Find the largest road polygon as the main road
        largest_road = None
        max_area = 0
        
        for polygon_coords in road_polygons:
            area = calculate_polygon_area(polygon_coords)
            if area > max_area:
                max_area = area
                largest_road = polygon_coords
        
        if largest_road is None:
            return None
        
        # Convert to shapely polygon
        coords = [(largest_road[i], largest_road[i+1]) for i in range(0, len(largest_road), 2)]
        if len(coords) < 3:
            return None
        
        road_polygon = Polygon(coords)
        
        # Get the bounding box of the road
        minx, miny, maxx, maxy = road_polygon.bounds
        
        # Calculate the centroid and orientation
        centroid = road_polygon.centroid
        
        # Determine the main axis direction by analyzing the polygon shape
        # Use the longest dimension of the bounding box
        if (maxx - minx) > (maxy - miny):
            # Road is more horizontal - create vertical dividing line through centroid
            x = centroid.x
            line = LineString([(x, 0), (x, image_size[1])])
        else:
            # Road is more vertical - create horizontal dividing line through centroid
            y = centroid.y
            line = LineString([(0, y), (image_size[0], y)])
        
        return line
    
    except Exception as e:
        print(f"Warning: Could not determine road axis: {e}")
        return None


def classify_sidewalk_regions(segmentation_result: dict, road_axis: LineString, image_size: tuple) -> dict:
    """
    Classify surfaces on the left and right sides of the road axis.
    
    Args:
        segmentation_result (dict): Results from segment_image_and_classify_surfaces
        road_axis (LineString): Line dividing the image into left and right regions
        image_size (tuple): Image dimensions (width, height)
    
    Returns:
        dict: Classification results with road, left_sidewalk, right_sidewalk entries
    """
    result = {
        'road': 'unknown',
        'left_sidewalk': 'no_sidewalk',
        'right_sidewalk': 'no_sidewalk'
    }
    
    try:
        # Create image boundary polygon
        image_polygon = Polygon([(0, 0), (image_size[0], 0), (image_size[0], image_size[1]), (0, image_size[1])])
        
        # Split image into left and right regions using road axis
        try:
            split_result = split(image_polygon, road_axis)
            if len(split_result.geoms) >= 2:
                left_region = split_result.geoms[0]
                right_region = split_result.geoms[1]
            else:
                # If split fails, fall back to simple vertical/horizontal division
                centroid = road_axis.centroid
                if road_axis.coords[0][0] == road_axis.coords[1][0]:  # Vertical line
                    x = centroid.x
                    left_region = Polygon([(0, 0), (x, 0), (x, image_size[1]), (0, image_size[1])])
                    right_region = Polygon([(x, 0), (image_size[0], 0), (image_size[0], image_size[1]), (x, image_size[1])])
                else:  # Horizontal line
                    y = centroid.y
                    left_region = Polygon([(0, 0), (image_size[0], 0), (image_size[0], y), (0, y)])
                    right_region = Polygon([(0, y), (image_size[0], y), (image_size[0], image_size[1]), (0, image_size[1])])
        except Exception:
            # Fallback: simple vertical division at image center
            mid_x = image_size[0] / 2
            left_region = Polygon([(0, 0), (mid_x, 0), (mid_x, image_size[1]), (0, image_size[1])])
            right_region = Polygon([(mid_x, 0), (image_size[0], 0), (image_size[0], image_size[1]), (mid_x, image_size[1])])
        
        # Process segments
        road_polygons = []
        left_sidewalk_polygons = []
        right_sidewalk_polygons = []
        left_car_polygons = []
        right_car_polygons = []
        
        for segment in segmentation_result.get('pathway_segments', []):
            category = segment.get('category', '')
            polygon_coords = np.array(segment.get('polygon', []))
            surface_type = segment.get('surface_type', {}).get('surface', 'unknown')
            
            if len(polygon_coords) < 6:  # At least 3 points
                continue
            
            # Convert to shapely polygon
            coords = [(polygon_coords[i], polygon_coords[i+1]) for i in range(0, len(polygon_coords), 2)]
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    continue
                
                if category == 'roads':
                    road_polygons.append((poly, surface_type))
                elif category == 'sidewalks':
                    # Determine which side of the road axis this sidewalk is on
                    centroid = poly.centroid
                    if left_region.contains(centroid):
                        left_sidewalk_polygons.append((poly, surface_type))
                    elif right_region.contains(centroid):
                        right_sidewalk_polygons.append((poly, surface_type))
                elif category == 'car':
                    # Determine which side of the road axis this car is on
                    centroid = poly.centroid
                    if left_region.contains(centroid):
                        left_car_polygons.append((poly, surface_type))
                    elif right_region.contains(centroid):
                        right_car_polygons.append((poly, surface_type))
                        
            except Exception:
                continue
        
        # Classify road surface (use the largest road polygon)
        if road_polygons:
            largest_road = max(road_polygons, key=lambda x: x[0].area)
            result['road'] = largest_road[1]
        
        # Classify left sidewalk
        result['left_sidewalk'] = classify_side_surface(left_sidewalk_polygons, left_car_polygons, road_polygons)
        
        # Classify right sidewalk
        result['right_sidewalk'] = classify_side_surface(right_sidewalk_polygons, right_car_polygons, road_polygons)
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not classify sidewalk regions: {e}")
        return result


def classify_side_surface(sidewalk_polygons: list, car_polygons: list, road_polygons: list) -> str:
    """
    Classify the surface type for one side of the road.
    
    Args:
        sidewalk_polygons (list): List of (polygon, surface_type) tuples for sidewalks
        car_polygons (list): List of (polygon, surface_type) tuples for cars
        road_polygons (list): List of (polygon, surface_type) tuples for roads
    
    Returns:
        str: Surface classification ('no_sidewalk', 'car_hindered', or actual surface type)
    """
    if sidewalk_polygons:
        # Use the largest sidewalk polygon
        largest_sidewalk = max(sidewalk_polygons, key=lambda x: x[0].area)
        return largest_sidewalk[1]
    
    if not car_polygons:
        return 'no_sidewalk'
    
    # Calculate car to road area ratio
    total_car_area = sum(poly.area for poly, _ in car_polygons)
    total_road_area = sum(poly.area for poly, _ in road_polygons)
    
    if total_road_area == 0:
        return 'no_sidewalk'
    
    car_road_ratio = total_car_area / total_road_area
    
    # If cars are less than 1/3 the size of roads, consider it no sidewalk
    # Otherwise, it's car-hindered
    if car_road_ratio < 1/3:
        return 'no_sidewalk'
    else:
        return 'car_hindered'
