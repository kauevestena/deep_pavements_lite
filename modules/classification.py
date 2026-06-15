"""
Deep Pavements Lite — Surface material classification using CLIP.

Provides functions for:
- Classifying surface materials within polygon regions using CLIP
- Orchestrating segmentation + classification for a single image
"""

from __future__ import annotations

import json
import os
from typing import Any

import clip
import numpy as np
import torch
from PIL import Image, ImageDraw

from modules.constants import (
    CLIP_INPUT_SIZE,
    MIN_REGION_SIZE,
    PATHWAY_CLASS_MAPPING,
    data_path,
    default_surfaces,
    ext_out,
    pathway_categories,
)
from modules.geometry import extract_polygons_from_mask
from modules.segmentation import (
    create_segmentation_overlay,
    get_cityscapes_color_encoding,
    segment_image,
)


def classify_surface_type(
    image: Image.Image,
    polygon: np.ndarray,
    clip_model: Any,
    clip_preprocess: Any,
    device: torch.device,
) -> dict[str, Any]:
    """
    Classify surface material type of a polygonal region using CLIP.

    Extracts the region defined by the polygon, uses CLIP to compute
    similarity scores against text descriptions of surface types,
    and returns the best match with confidence.

    Args:
        image: Source image containing the surface region.
        polygon: Flattened polygon coordinates [x1, y1, x2, y2, ...].
        clip_model: Loaded CLIP model.
        clip_preprocess: CLIP preprocessing transform.
        device: PyTorch device for inference.

    Returns:
        Dict with 'surface' (str) and 'confidence' (float).
        Returns surface='unknown', confidence=0.0 on error.
    """
    try:
        # Create binary mask from polygon coordinates
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, fill=255)

        # Apply mask to extract only the polygon region
        masked_image = Image.composite(
            image, Image.new("RGB", image.size, (0, 0, 0)), mask
        )

        # Crop to bounding box
        bbox = mask.getbbox()
        cropped = masked_image.crop(bbox) if bbox else masked_image

        # Ensure minimum size for effective classification
        if cropped.size[0] < MIN_REGION_SIZE or cropped.size[1] < MIN_REGION_SIZE:
            cropped = cropped.resize((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))

        # Prepare text prompts for all surface types
        surface_prompts = [f"a photo of {s} surface" for s in default_surfaces]

        # Encode text and image with CLIP
        text_tokens = clip.tokenize(surface_prompts).to(device)
        image_input = clip_preprocess(cropped).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)

            # Cosine similarity → softmax probabilities
            similarities = (image_features @ text_features.T).softmax(dim=-1)

            best_idx = similarities.argmax().item()
            confidence = similarities[0, best_idx].item()

            return {
                "surface": default_surfaces[best_idx],
                "confidence": float(confidence),
            }

    except Exception as e:
        print(f"Warning: Could not classify surface type: {e}")
        return {"surface": "unknown", "confidence": 0.0}


def segment_and_classify(
    image: Image.Image,
    clip_model: Any,
    clip_preprocess: Any,
    device: torch.device,
    filename: str,
    debug_info: dict | None = None,
) -> dict:
    """
    Segment image using OneFormer and classify surface types using CLIP.

    Orchestrates the full per-image analysis:
    1. Semantic segmentation (OneFormer or heuristic fallback)
    2. Polygon extraction from segmentation masks
    3. Surface classification within each polygon
    4. Results aggregation and optional debug output

    Args:
        image: Input street-level image.
        clip_model: Loaded CLIP model.
        clip_preprocess: CLIP preprocessing transform.
        device: PyTorch device for inference.
        filename: Original filename for metadata.
        debug_info: If provided, saves debug outputs (segmented images, masks, etc.).

    Returns:
        Dict with 'filename', 'image_size', 'pathway_segments',
        'segmentation_method', and optionally 'error'.
    """
    try:
        # Perform semantic segmentation
        segmentation_mask, segmentation_method = segment_image(image, device)

        results: dict = {
            "filename": filename,
            "image_size": image.size,
            "pathway_segments": [],
            "segmentation_method": segmentation_method,
        }

        # Process each pathway category
        for category_name in pathway_categories:
            if category_name not in PATHWAY_CLASS_MAPPING:
                continue

            class_ids = PATHWAY_CLASS_MAPPING[category_name]

            # Create binary mask for this category
            category_mask = np.zeros_like(segmentation_mask, dtype=bool)
            for class_id in class_ids:
                category_mask |= segmentation_mask == class_id

            if not np.any(category_mask):
                continue

            # Extract polygon contours
            polygons = extract_polygons_from_mask(category_mask)

            for i, polygon in enumerate(polygons):
                if len(polygon) > 6:  # At least 3 points
                    surface_type = classify_surface_type(
                        image, polygon, clip_model, clip_preprocess, device
                    )

                    segment_info = {
                        "category": category_name,
                        "polygon": polygon.tolist(),
                        "surface_type": surface_type,
                        "segment_id": f"{category_name}_{i}",
                    }
                    results["pathway_segments"].append(segment_info)

        # Save results as JSON
        output_json_filename = os.path.splitext(filename)[0] + "_segments.json"
        output_json_path = os.path.join(data_path, "output", output_json_filename)
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save debug outputs if requested
        if debug_info:
            _save_debug_segmentation(
                image, segmentation_mask, debug_info, PATHWAY_CLASS_MAPPING
            )

        print(f"Saved segmentation results to {output_json_path}")
        return results

    except Exception as e:
        print(f"Warning: Could not perform segmentation and classification: {e}")
        return {
            "filename": filename,
            "image_size": image.size,
            "pathway_segments": [],
            "error": str(e),
        }


def _save_debug_segmentation(
    image: Image.Image,
    segmentation_mask: np.ndarray,
    debug_info: dict,
    pathway_class_mapping: dict[str, list[int]],
) -> None:
    """Save debug outputs for segmentation (overlay, mask, color encoding)."""
    # Save segmented image with overlay
    overlay = create_segmentation_overlay(image, segmentation_mask, pathway_class_mapping)
    segmented_filename = f"{debug_info['image_id']}_segmented.png"
    segmented_path = os.path.join(
        debug_info["debug_segmented_path"], segmented_filename
    )
    overlay.save(segmented_path)

    # Save raw segmentation mask
    mask_filename = f"{debug_info['image_id']}_mask.npy"
    mask_path = os.path.join(debug_info["debug_metadata_path"], mask_filename)
    np.save(mask_path, segmentation_mask)

    # Save color encoding info
    color_encoding = get_cityscapes_color_encoding()
    encoding_filename = f"{debug_info['image_id']}_color_encoding.json"
    encoding_path = os.path.join(debug_info["debug_metadata_path"], encoding_filename)
    with open(encoding_path, "w") as f:
        json.dump(color_encoding, f, indent=2)
