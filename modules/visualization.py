"""
Deep Pavements Lite — Interactive map visualization.

Generates a standalone HTML file with a Leaflet.js interactive map
showing classified pavement surfaces. No npm or build step required —
uses CDN-hosted Leaflet.
"""

from __future__ import annotations

import json
import os
from typing import Any

import geopandas as gpd


# Surface type → marker color mapping
SURFACE_COLORS: dict[str, str] = {
    "asphalt": "#333333",
    "concrete": "#999999",
    "concrete_plates": "#AAAAAA",
    "grass": "#2ecc71",
    "ground": "#8B4513",
    "sett": "#C0392B",
    "paving_stones": "#E67E22",
    "cobblestone": "#95A5A6",
    "gravel": "#BDC3C7",
    "sand": "#F1C40F",
    "compacted": "#7F8C8D",
    "unknown": "#E74C3C",
    "no_sidewalk": "#FFFFFF",
    "car_hindered": "#3498DB",
}


def generate_map(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    filename: str = "surface_map.html",
) -> str:
    """
    Generate a standalone interactive Leaflet map from classification results.

    Args:
        gdf: GeoDataFrame with surface classification results.
             Expected columns: road, left_sidewalk, right_sidewalk,
             image_id, filename, geometry (Point).
        output_path: Directory to save the HTML file.
        filename: Output filename (default: surface_map.html).

    Returns:
        Absolute path to the generated HTML file.
    """
    # Convert GeoDataFrame to feature list for JavaScript
    features = _gdf_to_features(gdf)
    features_json = json.dumps(features, indent=2)

    # Build unique surface types for legend
    all_surfaces: set[str] = set()
    for f in features:
        for key in ("road", "left_sidewalk", "right_sidewalk"):
            val = f["properties"].get(key, "unknown")
            all_surfaces.add(val)

    legend_items = ""
    for surface in sorted(all_surfaces):
        color = SURFACE_COLORS.get(surface, "#999")
        legend_items += f'<div class="legend-item"><span class="legend-color" style="background:{color}"></span>{surface}</div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Pavements Lite — Surface Classification Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
        .legend {{
            position: absolute; bottom: 30px; right: 10px; z-index: 1000;
            background: rgba(255,255,255,0.95); padding: 14px 18px;
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            font-size: 13px; max-height: 60vh; overflow-y: auto;
        }}
        .legend h4 {{ margin-bottom: 8px; color: #333; }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
        .legend-color {{
            display: inline-block; width: 14px; height: 14px;
            border-radius: 50%; margin-right: 8px; border: 1px solid #ccc;
        }}
        .title-bar {{
            position: absolute; top: 10px; left: 50px; z-index: 1000;
            background: rgba(255,255,255,0.95); padding: 10px 20px;
            border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            font-size: 16px; font-weight: bold; color: #333;
        }}
        .leaflet-popup-content {{ min-width: 200px; }}
        .popup-table {{ width: 100%; border-collapse: collapse; }}
        .popup-table td {{ padding: 4px 8px; border-bottom: 1px solid #eee; }}
        .popup-table td:first-child {{ font-weight: bold; color: #555; }}
        .surface-badge {{
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            color: white; font-size: 12px; font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="title-bar">🛣️ Deep Pavements Lite — Surface Classification Map</div>
    <div id="map"></div>
    <div class="legend">
        <h4>Surface Types</h4>
        {legend_items}
    </div>

    <script>
        const features = {features_json};

        const surfaceColors = {json.dumps(SURFACE_COLORS)};

        // Initialize map
        const map = L.map('map').setView([0, 0], 2);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);

        // Add markers
        const markers = [];
        features.forEach(function(f) {{
            const lat = f.geometry.coordinates[1];
            const lng = f.geometry.coordinates[0];
            const p = f.properties;
            const roadColor = surfaceColors[p.road] || '#999';

            const marker = L.circleMarker([lat, lng], {{
                radius: 8,
                fillColor: roadColor,
                color: '#333',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.85
            }});

            function badge(surface) {{
                const c = surfaceColors[surface] || '#999';
                return '<span class="surface-badge" style="background:' + c + '">' + surface + '</span>';
            }}

            const conf = (v) => v !== undefined ? (v * 100).toFixed(0) + '%' : 'N/A';

            marker.bindPopup(
                '<table class="popup-table">' +
                '<tr><td>Image ID</td><td>' + p.image_id + '</td></tr>' +
                '<tr><td>Road</td><td>' + badge(p.road) + ' ' + conf(p.road_confidence) + '</td></tr>' +
                '<tr><td>Left Sidewalk</td><td>' + badge(p.left_sidewalk) + ' ' + conf(p.left_confidence) + '</td></tr>' +
                '<tr><td>Right Sidewalk</td><td>' + badge(p.right_sidewalk) + ' ' + conf(p.right_confidence) + '</td></tr>' +
                '<tr><td>Coordinates</td><td>' + lat.toFixed(6) + ', ' + lng.toFixed(6) + '</td></tr>' +
                '</table>'
            );

            marker.addTo(map);
            markers.push(marker);
        }});

        // Fit map to markers
        if (markers.length > 0) {{
            const group = L.featureGroup(markers);
            map.fitBounds(group.getBounds().pad(0.1));
        }}
    </script>
</body>
</html>"""

    output_file = os.path.join(output_path, filename)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated interactive map: {output_file}")
    return output_file


def _gdf_to_features(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    """Convert GeoDataFrame rows to GeoJSON-like feature dicts for JavaScript."""
    features = []
    for _, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row.geometry.x, row.geometry.y],
            },
            "properties": {
                "image_id": str(row.get("image_id", "")),
                "filename": str(row.get("filename", "")),
                "road": str(row.get("road", "unknown")),
                "road_confidence": float(row.get("road_confidence", 0)),
                "left_sidewalk": str(row.get("left_sidewalk", "unknown")),
                "left_confidence": float(row.get("left_confidence", 0)),
                "right_sidewalk": str(row.get("right_sidewalk", "unknown")),
                "right_confidence": float(row.get("right_confidence", 0)),
            },
        }
        features.append(feature)
    return features
