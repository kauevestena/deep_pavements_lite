# API Reference

This page describes the primary public APIs and modules in the Deep Pavements Lite package.

---

## 📌 Pipeline Module (`modules.pipeline`)

The main interface for running the end-to-end processing pipeline.

### `process_images(images_gdf, output_path, debug_mode=False, workers=1)`
Processes all street images defined in the metadata GeoDataFrame.

- **Parameters:**
  - `images_gdf` *(geopandas.GeoDataFrame)*: Input table containing image IDs, file paths, and geospatial geometry (points).
  - `output_path` *(str)*: Target directory to save result GeoJSON, GeoPackage, and JSON metadata.
  - `debug_mode` *(bool, optional)*: If `True`, enables step-by-step intermediate image export and builds an HTML visual debug report. Default is `False`.
  - `workers` *(int, optional)*: Number of parallel workers for I/O operations. Default is `1`.
- **Returns:**
  - `geopandas.GeoDataFrame`: The processed GDF enriched with `road`, `left_sidewalk`, and `right_sidewalk` classification columns.

---

## 📌 Classification Module (`modules.classification`)

Interfaces with CLIP to identify material surface classes.

### `classify_surface_type(image, polygon, model, preprocess, device="cpu")`
Classifies the surface material of a specific segmented polygon.

- **Parameters:**
  - `image` *(PIL.Image)*: Original high-resolution street photo.
  - `polygon` *(np.ndarray)*: Flattened list of polygon boundary coordinates `[x1, y1, x2, y2, ...]`.
  - `model` *(CLIP)*: CLIP model instance.
  - `preprocess` *(callable)*: Image transform function.
  - `device` *(str, optional)*: Execution device (`"cuda"` or `"cpu"`).
- **Returns:**
  - `dict`: A dictionary containing predicted `surface` string and `confidence` float.

---

## 📌 Geometry Module (`modules.geometry`)

Performs spatial checks and shapes classification.

### `get_road_axis_line(road_polygons, image_size)`
Calculates the central dividing line of the road using PCA or centroid orientations.

- **Parameters:**
  - `road_polygons` *(list[np.ndarray])*: Polygons representing the road surface.
  - `image_size` *(tuple[int, int])*: Width and height of the image.
- **Returns:**
  - `shapely.geometry.LineString | None`: Dividing line representing the road axis.

### `classify_sidewalk_regions(segmentation_result, road_axis, image_size)`
Divides detected sidewalks into left and right sides relative to the road axis.

- **Parameters:**
  - `segmentation_result` *(dict)*: Dictionary of detected polygons and their surface types.
  - `road_axis` *(LineString)*: Calculated dividing axis line.
  - `image_size` *(tuple[int, int])*: Width and height of the image.
- **Returns:**
  - `dict`: Dictionary mapping `"left_sidewalk"` and `"right_sidewalk"` to their surface types.

---

## 📌 Visualization Module (`modules.visualization`)

Renders GIS overlays and interactive maps.

### `generate_map(gdf, output_path)`
Creates a standalone, interactive Leaflet HTML map showing points color-coded by pavement surface materials.

- **Parameters:**
  - `gdf` *(geopandas.GeoDataFrame)*: The final processed GDF containing pavement classification results.
  - `output_path` *(str)*: Target folder where `global_map.html` and assets will be saved.
