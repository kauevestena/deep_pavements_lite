"""
Deep Pavements Lite — Semantic segmentation.

Provides functions for:
- OneFormer-based semantic segmentation of urban street scenes
- Heuristic fallback segmentation when ML models are unavailable
- Segmentation mask overlay visualization
- Cityscapes color encoding utilities
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage import filters, measure, morphology


def segment_image(
    image: Image.Image,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    """
    Perform semantic segmentation on a street scene image.

    Tries OneFormer at full resolution first, then half resolution as a
    memory fallback, and finally heuristic segmentation if OneFormer fails.

    Args:
        image: Input street-level PIL image.
        device: PyTorch device for model inference.

    Returns:
        Tuple of (segmentation_mask, method_name):
        - segmentation_mask: np.ndarray with Cityscapes class IDs
        - method_name: String describing which method was used
    """
    try:
        return _segment_with_oneformer(image, device)
    except Exception as oneformer_error:
        print(f"OneFormer segmentation failed: {oneformer_error}")
        print("Using heuristic segmentation fallback...")
        mask = create_heuristic_segmentation(image)
        print("✓ Heuristic segmentation completed")
        return mask, "Heuristic fallback"


def _segment_with_oneformer(
    image: Image.Image,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    """Segment using OneFormer, trying full then half resolution."""
    from deep_pavements.models import OneFormerModelCache

    processor, model = OneFormerModelCache.load(device)

    # Try full resolution first
    try:
        inputs = processor(
            images=image, task_inputs=["semantic"], return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        mask = predicted_map.cpu().numpy()
        print("✓ OneFormer segmentation completed at full resolution")
        return mask, "OneFormer (full resolution)"

    except RuntimeError as memory_error:
        if "out of memory" not in str(memory_error).lower() and "cuda" not in str(
            memory_error
        ).lower():
            raise

        print(f"Memory error at full resolution: {memory_error}")
        print("Trying OneFormer with half resolution...")

        # Half resolution fallback
        half_size = (image.size[0] // 2, image.size[1] // 2)
        resized = image.resize(half_size, Image.Resampling.LANCZOS)

        inputs = processor(
            images=resized, task_inputs=["semantic"], return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[resized.size[::-1]]
        )[0]

        small_mask = predicted_map.cpu().numpy()
        # Resize mask back to original dimensions
        mask = np.array(
            Image.fromarray(small_mask.astype(np.uint8)).resize(
                image.size, Image.Resampling.NEAREST
            )
        )

        print("✓ OneFormer segmentation completed at half resolution")
        return mask, "OneFormer (half resolution)"


def create_heuristic_segmentation(image: Image.Image) -> np.ndarray:
    """
    Create a heuristic-based segmentation for street scenes.

    Uses simple computer vision techniques (Otsu thresholding, morphology,
    connected components) to create a reasonable segmentation when
    OneFormer is unavailable.

    Args:
        image: Input street scene image.

    Returns:
        Segmentation mask with Cityscapes-compatible class IDs:
        0=road, 1=sidewalk, 13=car, 255=background.
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Initialize with background class
    mask = np.full((height, width), 255, dtype=np.uint8)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array

    # ── Road detection ────────────────────────────────────────────────
    # Assume road is in the lower portion and darker
    lower_third = int(height * 0.6)
    lower_region = gray[lower_third:, :]

    if lower_region.size > 0:
        threshold = filters.threshold_otsu(lower_region)
        road_mask = np.zeros((height, width), dtype=bool)
        road_mask[lower_third:, :] = lower_region < (threshold * 1.05)

        # Morphological cleanup
        road_mask = morphology.binary_closing(road_mask, morphology.disk(5))
        road_mask = morphology.binary_opening(road_mask, morphology.disk(3))

        # Keep largest connected component (main road)
        labels = measure.label(road_mask)
        if labels.max() > 0:
            largest_region = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            road_mask = largest_region

        mask[road_mask] = 0  # Cityscapes road class

    # ── Sidewalk detection ────────────────────────────────────────────
    if np.any(mask == 0):
        road_pixels = mask == 0
        dilated_road = morphology.binary_dilation(road_pixels, morphology.disk(8))

        adjacent_to_road = dilated_road & ~road_pixels
        lighter_threshold = np.percentile(gray, 60)
        lighter_areas = gray > lighter_threshold

        sidewalk_mask = adjacent_to_road & lighter_areas
        sidewalk_mask = morphology.binary_closing(sidewalk_mask, morphology.disk(3))

        mask[sidewalk_mask] = 1  # Cityscapes sidewalk class

    # ── Car detection ─────────────────────────────────────────────────
    upper_region = gray[: int(height * 0.6), :]
    if upper_region.size > 0:
        car_threshold = np.percentile(upper_region, 25)
        dark_objects = upper_region < car_threshold

        car_labels = measure.label(dark_objects)
        for region in measure.regionprops(car_labels):
            if 200 < region.area < 5000:
                bbox = region.bbox
                height_obj = bbox[2] - bbox[0]
                width_obj = bbox[3] - bbox[1]

                if width_obj > 0 and height_obj > 0:
                    aspect_ratio = max(width_obj, height_obj) / min(
                        width_obj, height_obj
                    )
                    if 1.2 < aspect_ratio < 3.5:
                        coords = region.coords
                        mask[coords[:, 0], coords[:, 1]] = 13  # Cityscapes car class

    return mask


def create_segmentation_overlay(
    image: Image.Image,
    segmentation_mask: np.ndarray,
    pathway_class_mapping: dict[str, list[int]],
) -> Image.Image:
    """
    Create a visual overlay of the segmentation mask on the original image.

    Args:
        image: Original PIL image.
        segmentation_mask: Numpy array with Cityscapes class IDs.
        pathway_class_mapping: Mapping of categories to class IDs.

    Returns:
        PIL Image with colored segmentation overlay (RGB).
    """
    overlay_image = image.copy().convert("RGBA")

    # Color map: class_id → (R, G, B, A)
    colors: dict[int, tuple[int, int, int, int]] = {
        0: (255, 0, 0, 100),  # roads — red
        1: (0, 255, 0, 100),  # sidewalks — green
        13: (0, 0, 255, 100),  # car — blue
    }

    for _category_name, class_ids in pathway_class_mapping.items():
        for class_id in class_ids:
            if class_id in colors:
                class_mask = segmentation_mask == class_id
                if np.any(class_mask):
                    color = colors[class_id]
                    mask_array = np.zeros(
                        (*segmentation_mask.shape, 4), dtype=np.uint8
                    )
                    mask_array[class_mask] = color

                    mask_image = Image.fromarray(mask_array, "RGBA")
                    mask_image = mask_image.resize(
                        image.size, Image.Resampling.NEAREST
                    )
                    overlay_image = Image.alpha_composite(overlay_image, mask_image)

    return overlay_image.convert("RGB")


def get_cityscapes_color_encoding() -> dict:
    """
    Get the color encoding for Cityscapes classes used in segmentation.

    Returns:
        Dictionary with class information and colors.
    """
    return {
        "classes": {
            "0": {
                "name": "road",
                "color": [128, 64, 128],
                "description": "Road surface including asphalt and concrete roads",
            },
            "1": {
                "name": "sidewalk",
                "color": [244, 35, 232],
                "description": "Sidewalk areas for pedestrian walking",
            },
            "13": {
                "name": "car",
                "color": [0, 0, 142],
                "description": "Cars and other vehicles",
            },
        },
        "legend": {
            "road": "Red overlay in segmentation images",
            "sidewalk": "Green overlay in segmentation images",
            "car": "Blue overlay in segmentation images",
        },
    }
