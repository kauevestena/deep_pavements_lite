"""
Backward compatibility shim.

This module re-exports all public symbols from the deep_pavements package.
New code should import from deep_pavements directly:

    from deep_pavements import process_images
    from deep_pavements.geometry import extract_polygons_from_mask
"""

from deep_pavements.pipeline import process_images
from deep_pavements.classification import classify_surface_type, segment_and_classify
from deep_pavements.segmentation import (
    create_heuristic_segmentation,
    create_segmentation_overlay,
    get_cityscapes_color_encoding,
    segment_image,
)
from deep_pavements.geometry import (
    calculate_polygon_area,
    classify_side_surface,
    classify_sidewalk_regions,
    extract_polygons_from_mask,
    get_road_axis_line,
)
from deep_pavements.debug import generate_debug_html_report
from deep_pavements.models import (
    create_mock_clip_model,
    download_finetuned_model,
    load_clip_model,
    MockCLIPModel,
)
from deep_pavements.constants import *  # noqa: F401, F403

# Alias for backward compatibility
segment_image_and_classify_surfaces = segment_and_classify
_create_mock_clip_model = create_mock_clip_model
_try_download_finetuned_model = download_finetuned_model
_create_heuristic_segmentation = create_heuristic_segmentation
