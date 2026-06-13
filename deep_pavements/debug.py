"""
Deep Pavements Lite — Debug output and HTML report generation.

Provides functions for saving intermediary results and generating
comprehensive HTML reports for analysis and troubleshooting.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


def generate_debug_html_report(
    debug_data: list[dict[str, Any]],
    reports_path: str,
) -> None:
    """
    Generate a modern, interactive HTML report with all debug information.

    Creates a styled, self-contained HTML dashboard with a summary,
    image filters, search, side-by-side comparison tabs, categorized segment list,
    and lazy-loaded Leaflet GPS mapping.

    Args:
        debug_data: List of debug information dicts for each processed image.
        reports_path: Directory path to save the HTML report.
    """
    # Calculate statistics
    total_images = len(debug_data)
    images_with_road = sum(
        1 for item in debug_data
        if item.get("surface_classification", {}).get("road", "none") not in ("none", "unknown", "no_road")
    )
    images_with_sidewalk = sum(
        1 for item in debug_data
        if item.get("surface_classification", {}).get("left_sidewalk", "none") not in ("none", "unknown", "no_sidewalk")
        or item.get("surface_classification", {}).get("right_sidewalk", "none") not in ("none", "unknown", "no_sidewalk")
    )

    road_rate = (images_with_road / total_images * 100) if total_images > 0 else 0
    sidewalk_rate = (images_with_sidewalk / total_images * 100) if total_images > 0 else 0

    all_confidences = []
    surface_counts: dict[str, int] = {}
    unique_surfaces = set()

    for item in debug_data:
        # Segment confidences
        segments = item.get("segmentation_result", {}).get("pathway_segments", [])
        for seg in segments:
            st = seg.get("surface_type", {})
            if isinstance(st, dict) and "confidence" in st:
                all_confidences.append(st["confidence"])

        # Surface classifications for stats and filtering
        classification = item.get("surface_classification", {})
        for key in ["road", "left_sidewalk", "right_sidewalk"]:
            surface = classification.get(key)
            if surface and surface not in ("none", "unknown", "no_sidewalk", "no_road", "car_hindered"):
                surface_counts[surface] = surface_counts.get(surface, 0) + 1
                unique_surfaces.add(surface)

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    # Sort surfaces for distribution bar chart
    sorted_surfaces = sorted(surface_counts.items(), key=lambda x: x[1], reverse=True)
    total_detections = sum(surface_counts.values())

    # Generate distribution HTML
    distribution_html = ""
    for surface, count in sorted_surfaces:
        pct = (count / total_detections * 100) if total_detections > 0 else 0
        distribution_html += f"""
        <div class="dist-item">
            <div class="dist-header">
                <span class="dist-name">{surface.title()}</span>
                <span class="dist-count">{count} ({pct:.1f}%)</span>
            </div>
            <div class="dist-bar-bg">
                <div class="dist-bar-fill" style="width: {pct:.1f}%"></div>
            </div>
        </div>
        """

    # Generate dynamic filter buttons for surfaces
    surface_filter_buttons = ""
    for surface in sorted(unique_surfaces):
        surface_filter_buttons += f"""
        <button class="filter-btn" data-filter="{surface}">{surface.title()}</button>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Pavements Lite Debug Report</title>
    
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {{
            --bg-color: #0b0f19;
            --container-bg: #111827;
            --card-bg: #1f2937;
            --border-color: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --font-sans: 'Inter', sans-serif;
            --font-title: 'Plus Jakarta Sans', sans-serif;
        }}

        [data-theme="light"] {{
            --bg-color: #f3f4f6;
            --container-bg: #ffffff;
            --card-bg: #f9fafb;
            --border-color: #e5e7eb;
            --text-primary: #111827;
            --text-secondary: #4b5563;
            --accent-color: #4f46e5;
            --accent-hover: #4338ca;
            --accent-gradient: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.05);
            --shadow-lg: 0 10px 25px rgba(0,0,0,0.05);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.2s, border-color 0.2s;
        }}

        body {{
            font-family: var(--font-sans);
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 40px 20px;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: var(--container-bg);
            padding: 30px;
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--border-color);
        }}

        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 24px;
            margin-bottom: 30px;
        }}

        .header-title h1 {{
            font-family: var(--font-title);
            font-size: 28px;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}

        .header-title p {{
            color: var(--text-secondary);
            font-size: 14px;
        }}

        .theme-btn {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 16px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: var(--shadow-sm);
        }}

        .theme-btn:hover {{
            background-color: var(--border-color);
        }}

        /* Dashboard Overview */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
            margin-bottom: 40px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}

        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: var(--shadow-sm);
            position: relative;
            overflow: hidden;
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-gradient);
        }}

        .stat-label {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}

        .stat-value {{
            font-family: var(--font-title);
            font-size: 32px;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .stat-subtext {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 8px;
        }}

        .dist-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            box-shadow: var(--shadow-sm);
        }}

        .dist-card h3 {{
            font-family: var(--font-title);
            font-size: 16px;
            margin-bottom: 16px;
            color: var(--text-secondary);
        }}

        .dist-item {{
            margin-bottom: 14px;
        }}

        .dist-item:last-child {{
            margin-bottom: 0;
        }}

        .dist-header {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 4px;
        }}

        .dist-name {{
            color: var(--text-primary);
        }}

        .dist-count {{
            color: var(--text-secondary);
        }}

        .dist-bar-bg {{
            background-color: var(--bg-color);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            width: 100%;
        }}

        .dist-bar-fill {{
            background: var(--accent-gradient);
            height: 100%;
            border-radius: 4px;
        }}

        /* Control Panel */
        .control-panel {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: var(--shadow-sm);
        }}

        .search-row {{
            display: flex;
            gap: 16px;
        }}

        .search-box {{
            flex: 1;
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 12px 16px 12px 40px;
            border-radius: 8px;
            font-family: var(--font-sans);
            font-size: 14px;
        }}

        .search-box input:focus {{
            border-color: var(--accent-color);
            outline: none;
        }}

        .search-icon {{
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
            pointer-events: none;
        }}

        .sort-box select {{
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 12px 16px;
            border-radius: 8px;
            font-family: var(--font-sans);
            font-size: 14px;
            cursor: pointer;
            min-width: 200px;
        }}

        .sort-box select:focus {{
            border-color: var(--accent-color);
            outline: none;
        }}

        .filter-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }}

        .filter-label {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-right: 8px;
        }}

        .filter-btn {{
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
        }}

        .filter-btn:hover {{
            border-color: var(--text-secondary);
            color: var(--text-primary);
        }}

        .filter-btn.active {{
            background-color: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }}

        /* Image Cards */
        .cards-list {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}

        .image-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
            display: flex;
            flex-direction: column;
        }}

        .image-card.hidden {{
            display: none !important;
        }}

        .card-header {{
            background-color: rgba(0, 0, 0, 0.15);
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-header-left h2 {{
            font-family: var(--font-title);
            font-size: 18px;
            font-weight: 700;
        }}

        .card-header-left p {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        .card-header-actions {{
            display: flex;
            gap: 12px;
        }}

        .action-btn {{
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .action-btn:hover {{
            background-color: var(--border-color);
        }}

        .card-body {{
            padding: 24px;
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 24px;
        }}

        /* Left Column: Visualizer */
        .visualizer-column {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .viewer-tabs {{
            display: flex;
            background-color: var(--bg-color);
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .tab-btn {{
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 8px 12px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 6px;
            text-align: center;
        }}

        .tab-btn.active {{
            background-color: var(--card-bg);
            color: var(--text-primary);
            box-shadow: var(--shadow-sm);
        }}

        .viewer-content {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            aspect-ratio: 4 / 3;
            background-color: #000;
            position: relative;
        }}

        .view-panel {{
            display: none;
            width: 100%;
            height: 100%;
        }}

        .view-panel.active {{
            display: block;
        }}

        .view-panel img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}

        .split-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            height: 100%;
            width: 100%;
            background-color: var(--border-color);
        }}

        .split-half {{
            position: relative;
            height: 100%;
            background-color: #000;
            overflow: hidden;
        }}

        .split-half img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .split-label {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: #fff;
            padding: 4px 8px;
            font-size: 11px;
            font-weight: 700;
            border-radius: 4px;
            backdrop-filter: blur(4px);
            z-index: 5;
        }}

        /* Map Container */
        .map-container {{
            display: none;
            width: 100%;
            height: 250px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            margin-top: 16px;
        }}

        .map-container.active {{
            display: block;
        }}

        .map-div {{
            width: 100%;
            height: 100%;
        }}

        .map-error {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            background-color: var(--bg-color);
            color: var(--text-secondary);
            font-size: 13px;
            text-align: center;
            padding: 20px;
        }}

        .map-fallback-link {{
            color: var(--accent-color);
            text-decoration: none;
            margin-top: 10px;
            font-weight: 600;
        }}

        .map-fallback-link:hover {{
            text-decoration: underline;
        }}

        /* Right Column: Metadata & Results */
        .data-column {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        /* Classification Grid */
        .class-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}

        .class-card {{
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .class-label {{
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .class-value {{
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
        }}

        /* Metadata Table */
        .meta-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .meta-table tr {{
            border-bottom: 1px solid var(--border-color);
        }}

        .meta-table tr:last-child {{
            border-bottom: none;
        }}

        .meta-table td {{
            padding: 8px 0;
        }}

        .meta-label {{
            font-weight: 600;
            color: var(--text-secondary);
            width: 150px;
        }}

        .meta-val {{
            color: var(--text-primary);
        }}

        /* Segments Accordion */
        .accordion {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}

        .accordion-trigger {{
            width: 100%;
            background-color: rgba(0, 0, 0, 0.1);
            border: none;
            color: var(--text-primary);
            padding: 12px 16px;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            text-align: left;
        }}

        .accordion-trigger:hover {{
            background-color: rgba(0, 0, 0, 0.15);
        }}

        .accordion-trigger .chevron {{
            font-size: 10px;
            transition: transform 0.2s;
        }}

        .accordion.active .accordion-trigger .chevron {{
            transform: rotate(180deg);
        }}

        .accordion-content {{
            display: none;
            padding: 16px;
            background-color: var(--bg-color);
            border-top: 1px solid var(--border-color);
        }}

        .accordion.active .accordion-content {{
            display: block;
        }}

        /* Segment sub-tabs */
        .seg-tabs {{
            display: flex;
            gap: 6px;
            margin-bottom: 12px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}

        .seg-tab-btn {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
        }}

        .seg-tab-btn.active {{
            background-color: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }}

        .segments-scroll {{
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 6px;
        }}

        .segments-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }}

        .segments-table th {{
            background-color: var(--card-bg);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 6px 10px;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .segments-table td {{
            padding: 6px 10px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        .segments-table tr:last-child td {{
            border-bottom: none;
        }}

        /* Badge Pills */
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-align: center;
        }}

        .badge-high {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-medium {{
            background-color: rgba(245, 158, 11, 0.15);
            color: var(--warning-color);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .badge-low {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        .badge-category {{
            background-color: rgba(99, 102, 241, 0.15);
            color: var(--accent-color);
            border: 1px solid rgba(99, 102, 241, 0.3);
            text-transform: capitalize;
        }}

        /* Responsive Layout */
        @media (max-width: 1024px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}

            .card-body {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-title">
                <h1>🛣️ Deep Pavements Lite Dashboard</h1>
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <button id="theme-toggle" class="theme-btn" aria-label="Toggle theme">
                ☀️ Light Mode
            </button>
        </header>

        <!-- Dashboard Grid -->
        <section class="dashboard-grid">
            <!-- Stats -->
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-label">Processed Images</span>
                    <span class="stat-value">{total_images}</span>
                    <span class="stat-subtext">Total street-level images analyzed</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Road Detection Rate</span>
                    <span class="stat-value">{road_rate:.1f}%</span>
                    <span class="stat-subtext">{images_with_road} / {total_images} images with road segments</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Sidewalk Detection Rate</span>
                    <span class="stat-value">{sidewalk_rate:.1f}%</span>
                    <span class="stat-subtext">{images_with_sidewalk} / {total_images} images with sidewalks</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Average Confidence</span>
                    <span class="stat-value">{avg_confidence:.2f}</span>
                    <span class="stat-subtext">Average surface classification probability</span>
                </div>
            </div>

            <!-- Distribution chart -->
            <div class="dist-card">
                <h3>Surface Material Distribution</h3>
                {distribution_html if distribution_html else '<p style="font-size: 13px; color: var(--text-secondary);">No surface materials detected yet.</p>'}
            </div>
        </section>

        <!-- Control Panel -->
        <section class="control-panel">
            <div class="search-row">
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="search-input" placeholder="Search by Image ID or filename..." aria-label="Search images">
                </div>
                <div class="sort-box">
                    <select id="sort-select" aria-label="Sort images">
                        <option value="id-asc">Sort by: Image ID (A-Z)</option>
                        <option value="id-desc">Sort by: Image ID (Z-A)</option>
                        <option value="segments-desc">Sort by: Segments (High to Low)</option>
                        <option value="segments-asc">Sort by: Segments (Low to High)</option>
                        <option value="confidence-desc">Sort by: Confidence (High to Low)</option>
                        <option value="confidence-asc">Sort by: Confidence (Low to High)</option>
                    </select>
                </div>
            </div>

            <div class="filter-row">
                <span class="filter-label">Filter:</span>
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="road">Road Detected</button>
                <button class="filter-btn" data-filter="sidewalk">Sidewalk Detected</button>
                <button class="filter-btn" data-filter="car-hindered">Car Hindered</button>
                {surface_filter_buttons}
            </div>
        </section>

        <!-- Cards Container -->
        <main id="cards-container" class="cards-list">
"""

    # Add each image section
    for idx, item in enumerate(debug_data, 1):
        filename = item.get("filename", "unknown")
        image_id = item.get("image_id", "unknown")
        coordinates = item.get("coordinates", "unknown")

        segmentation_result = item.get("segmentation_result", {})
        segments = segmentation_result.get("pathway_segments", [])

        surface_classification = item.get("surface_classification", {})
        road_surface = surface_classification.get("road", "none")
        left_sidewalk = surface_classification.get("left_sidewalk", "none")
        right_sidewalk = surface_classification.get("right_sidewalk", "none")

        road_detected = str(road_surface not in ("none", "unknown", "no_road")).lower()
        sidewalk_detected = str(
            left_sidewalk not in ("none", "unknown", "no_sidewalk")
            or right_sidewalk not in ("none", "unknown", "no_sidewalk")
        ).lower()
        car_hindered = str(left_sidewalk == "car_hindered" or right_sidewalk == "car_hindered").lower()

        # Compute average confidence for this card
        card_confidences = []
        for s in segments:
            st = s.get("surface_type", {})
            if isinstance(st, dict) and "confidence" in st:
                card_confidences.append(st["confidence"])
        card_avg_confidence = sum(card_confidences) / len(card_confidences) if card_confidences else 0.0

        # Construct segments rows
        segment_rows_html = ""
        for seg in segments:
            category = seg.get("category", "unknown")
            surface_type = seg.get("surface_type", {})
            if isinstance(surface_type, dict):
                surface_name = surface_type.get("surface", "unknown")
                confidence = surface_type.get("confidence", 0.0)
                if confidence >= 0.8:
                    badge_class = "badge-high"
                elif confidence >= 0.5:
                    badge_class = "badge-medium"
                else:
                    badge_class = "badge-low"
                confidence_val = f'<span class="badge {badge_class}">{confidence:.2f}</span>'
            else:
                surface_name = str(surface_type)
                confidence_val = '<span class="badge badge-low">N/A</span>'

            segment_rows_html += f"""
            <tr data-category="{category}">
                <td><span class="badge badge-category">{category}</span></td>
                <td><strong>{surface_name}</strong></td>
                <td>{confidence_val}</td>
            </tr>
            """

        html_content += f"""
            <article class="image-card" 
                 data-id="{image_id}" 
                 data-filename="{filename}" 
                 data-segments="{len(segments)}" 
                 data-confidence="{card_avg_confidence:.4f}"
                 data-has-road="{road_detected}" 
                 data-has-sidewalk="{sidewalk_detected}" 
                 data-car-hindered="{car_hindered}"
                 data-road-surface="{road_surface}"
                 data-left-sidewalk="{left_sidewalk}"
                 data-right-sidewalk="{right_sidewalk}"
                 data-coordinates="{coordinates}">
                 
                <!-- Card Header -->
                <div class="card-header">
                    <div class="card-header-left">
                        <h2>Image {idx}: {filename}</h2>
                        <p>ID: {image_id} | GPS: {coordinates}</p>
                    </div>
                    <div class="card-header-actions">
                        <button class="action-btn" onclick="toggleMap(this, '{image_id}', '{coordinates}')" id="map-btn-{image_id}">
                            🗺️ Show Map
                        </button>
                    </div>
                </div>
                
                <!-- Card Body -->
                <div class="card-body">
                    <!-- Left: Tabbed Image Viewer -->
                    <div class="visualizer-column">
                        <div class="viewer-tabs" id="tabs-{image_id}">
                            <button class="tab-btn active" onclick="switchTab(this, 'original')">Original</button>
                            <button class="tab-btn" onclick="switchTab(this, 'segmented')">Segmented Overlay</button>
                            <button class="tab-btn" onclick="switchTab(this, 'side-by-side')">Split Side-by-Side</button>
                        </div>
                        <div class="viewer-content">
                            <!-- Original Image Panel -->
                            <div class="view-panel original active" id="panel-original-{image_id}">
                                <img src="../images/{filename}" alt="Original image of road" loading="lazy">
                            </div>
                            <!-- Segmented Overlay Image Panel -->
                            <div class="view-panel segmented" id="panel-segmented-{image_id}">
                                <img src="../segmented_images/{image_id}_segmented.png" alt="Segmented road classification overlay" loading="lazy">
                            </div>
                            <!-- Side-by-Side Image Panel -->
                            <div class="view-panel side-by-side" id="panel-side-by-side-{image_id}">
                                <div class="split-container">
                                    <div class="split-half">
                                        <div class="split-label">Original</div>
                                        <img src="../images/{filename}" alt="Original image" loading="lazy">
                                    </div>
                                    <div class="split-half">
                                        <div class="split-label">Segmented</div>
                                        <img src="../segmented_images/{image_id}_segmented.png" alt="Segmented image" loading="lazy">
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Map Dropdown Panel -->
                        <div class="map-container" id="map-container-{image_id}">
                            <div class="map-div" id="map-{image_id}"></div>
                        </div>
                    </div>
                    
                    <!-- Right: Metadata & Classification -->
                    <div class="data-column">
                        <!-- Classifications -->
                        <div class="class-grid">
                            <div class="class-card">
                                <span class="class-label">🛣️ Road</span>
                                <span class="class-value">{road_surface}</span>
                            </div>
                            <div class="class-card">
                                <span class="class-label">👈 Left Sidewalk</span>
                                <span class="class-value">{left_sidewalk}</span>
                            </div>
                            <div class="class-card">
                                <span class="class-label">👉 Right Sidewalk</span>
                                <span class="class-value">{right_sidewalk}</span>
                            </div>
                        </div>
                        
                        <!-- Metadata Properties -->
                        <table class="meta-table">
                            <tr>
                                <td class="meta-label">Coordinates</td>
                                <td class="meta-val">{coordinates}</td>
                            </tr>
                            <tr>
                                <td class="meta-label">Segmentation Mode</td>
                                <td class="meta-val">{segmentation_result.get('segmentation_method', 'Unknown')}</td>
                            </tr>
                            <tr>
                                <td class="meta-label">Road Axis Line</td>
                                <td class="meta-val">{'✓ Available' if item.get('road_axis') else '✗ Not Detected'}</td>
                            </tr>
                            <tr>
                                <td class="meta-label">Total Segments</td>
                                <td class="meta-val">{len(segments)}</td>
                            </tr>
                        </table>
                        
                        <!-- Segments Accordion -->
                        <div class="accordion" id="accordion-{image_id}">
                            <button class="accordion-trigger" onclick="toggleAccordion('{image_id}')">
                                <span>🔍 Detailed Segment Analysis</span>
                                <span class="chevron">▼</span>
                            </button>
                            <div class="accordion-content">
                                <div class="seg-tabs">
                                    <button class="seg-tab-btn active" onclick="filterSegCategory(this, 'all')">All</button>
                                    <button class="seg-tab-btn" onclick="filterSegCategory(this, 'roads')">Roads</button>
                                    <button class="seg-tab-btn" onclick="filterSegCategory(this, 'sidewalks')">Sidewalks</button>
                                    <button class="seg-tab-btn" onclick="filterSegCategory(this, 'car')">Cars</button>
                                </div>
                                <div class="segments-scroll">
                                    <table class="segments-table">
                                        <thead>
                                            <tr>
                                                <th>Category</th>
                                                <th>Surface</th>
                                                <th>Confidence</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {segment_rows_html if segment_rows_html else '<tr><td colspan="3" style="text-align:center;">No segments detected</td></tr>'}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </article>
        """

    html_content += """
        </main>
    </div>

    <!-- Scripts -->
    <script>
        // Tab switching logic for visualizer
        function switchTab(btn, mode) {
            const tabsContainer = btn.parentElement;
            const tabButtons = tabsContainer.querySelectorAll('.tab-btn');
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Get card container
            const card = btn.closest('.image-card');
            const imageId = card.getAttribute('data-id');
            
            // Panels
            const panelOriginal = document.getElementById(`panel-original-${imageId}`);
            const panelSegmented = document.getElementById(`panel-segmented-${imageId}`);
            const panelSideBySide = document.getElementById(`panel-side-by-side-${imageId}`);
            
            panelOriginal.classList.remove('active');
            panelSegmented.classList.remove('active');
            panelSideBySide.classList.remove('active');
            
            if (mode === 'original') {
                panelOriginal.classList.add('active');
            } else if (mode === 'segmented') {
                panelSegmented.classList.add('active');
            } else if (mode === 'side-by-side') {
                panelSideBySide.classList.add('active');
            }
        }

        // Accordion Toggle
        function toggleAccordion(imageId) {
            const accordion = document.getElementById(`accordion-${imageId}`);
            accordion.classList.toggle('active');
        }

        // Accordion segments category filtering
        function filterSegCategory(btn, category) {
            const segTabsContainer = btn.parentElement;
            const buttons = segTabsContainer.querySelectorAll('.seg-tab-btn');
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const accordionContent = btn.closest('.accordion-content');
            const rows = accordionContent.querySelectorAll('.segments-table tbody tr');
            
            rows.forEach(row => {
                const rowCategory = row.getAttribute('data-category');
                if (category === 'all' || rowCategory === category) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }

        // Leaflet dynamic map toggler with offline support
        function toggleMap(btn, cardId, coordinatesStr) {
            const mapContainer = document.getElementById(`map-container-${cardId}`);
            if (mapContainer.classList.contains('active')) {
                mapContainer.classList.remove('active');
                btn.textContent = '🗺️ Show Map';
                return;
            }
            
            mapContainer.classList.add('active');
            btn.textContent = '🗺️ Hide Map';
            
            const mapDiv = document.getElementById(`map-${cardId}`);
            if (mapDiv.dataset.initialized === 'true') {
                return;
            }
            
            const parts = coordinatesStr.split(',').map(s => parseFloat(s.trim()));
            if (parts.length !== 2 || isNaN(parts[0]) || isNaN(parts[1])) {
                mapDiv.innerHTML = `<div class="map-error">Invalid GPS coordinates: ${coordinatesStr}</div>`;
                return;
            }
            
            // Mapillary coordinate format: x (Lon), y (Lat)
            const lon = parts[0];
            const lat = parts[1];
            
            if (typeof L === 'undefined') {
                mapDiv.innerHTML = `
                    <div class="map-error">
                        <p>Map tiles cannot be loaded (Offline or CDN unavailable)</p>
                        <p>Coordinates: <strong>${lat}, ${lon}</strong></p>
                        <a href="https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=17/${lat}/${lon}" target="_blank" class="map-fallback-link">
                            Open in OpenStreetMap ↗
                        </a>
                    </div>`;
                return;
            }
            
            try {
                const map = L.map(`map-${cardId}`).setView([lat, lon], 17);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 19,
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
                
                L.marker([lat, lon]).addTo(map)
                    .bindPopup(`<b>Image ID:</b> ${cardId}<br><b>GPS:</b> ${lat}, ${lon}`)
                    .openPopup();
                
                mapDiv.dataset.initialized = 'true';
                
                // Force size invalidation for proper leaflet redraw in toggled tab
                setTimeout(() => {
                    map.invalidateSize();
                }, 150);
            } catch (err) {
                console.error("Leaflet initialization failed", err);
                mapDiv.innerHTML = `<div class="map-error">Failed to initialize map framework.</div>`;
            }
        }

        // Filtering and sorting control panel
        const searchInput = document.getElementById('search-input');
        const sortSelect = document.getElementById('sort-select');
        const filterBtns = document.querySelectorAll('.filter-btn');
        const cardsContainer = document.getElementById('cards-container');
        const cards = Array.from(document.querySelectorAll('.image-card'));

        let currentFilter = 'all';
        let searchQuery = '';

        function updateFilterView() {
            cards.forEach(card => {
                const filename = card.getAttribute('data-filename').toLowerCase();
                const id = card.getAttribute('data-id').toLowerCase();
                const matchesSearch = filename.includes(searchQuery) || id.includes(searchQuery);
                
                let matchesFilter = false;
                if (currentFilter === 'all') {
                    matchesFilter = true;
                } else if (currentFilter === 'road') {
                    matchesFilter = card.getAttribute('data-has-road') === 'true';
                } else if (currentFilter === 'sidewalk') {
                    matchesFilter = card.getAttribute('data-has-sidewalk') === 'true';
                } else if (currentFilter === 'car-hindered') {
                    matchesFilter = card.getAttribute('data-car-hindered') === 'true';
                } else {
                    // Surface category filter
                    const roadSurface = card.getAttribute('data-road-surface');
                    const leftSidewalk = card.getAttribute('data-left-sidewalk');
                    const rightSidewalk = card.getAttribute('data-right-sidewalk');
                    matchesFilter = (roadSurface === currentFilter) || (leftSidewalk === currentFilter) || (rightSidewalk === currentFilter);
                }
                
                if (matchesSearch && matchesFilter) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });
        }

        searchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value.toLowerCase();
            updateFilterView();
        });

        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentFilter = btn.getAttribute('data-filter');
                updateFilterView();
            });
        });

        sortSelect.addEventListener('change', (e) => {
            const sortBy = e.target.value;
            const sorted = [...cards];
            
            if (sortBy === 'id-asc') {
                sorted.sort((a, b) => a.getAttribute('data-id').localeCompare(b.getAttribute('data-id')));
            } else if (sortBy === 'id-desc') {
                sorted.sort((a, b) => b.getAttribute('data-id').localeCompare(a.getAttribute('data-id')));
            } else if (sortBy === 'segments-desc') {
                sorted.sort((a, b) => parseInt(b.getAttribute('data-segments')) - parseInt(a.getAttribute('data-segments')));
            } else if (sortBy === 'segments-asc') {
                sorted.sort((a, b) => parseInt(a.getAttribute('data-segments')) - parseInt(b.getAttribute('data-segments')));
            } else if (sortBy === 'confidence-desc') {
                sorted.sort((a, b) => parseFloat(b.getAttribute('data-confidence')) - parseFloat(a.getAttribute('data-confidence')));
            } else if (sortBy === 'confidence-asc') {
                sorted.sort((a, b) => parseFloat(a.getAttribute('data-confidence')) - parseFloat(b.getAttribute('data-confidence')));
            }
            
            // Re-append sorted cards
            sorted.forEach(card => cardsContainer.appendChild(card));
        });

        // Theme Toggle Logic
        const themeToggle = document.getElementById('theme-toggle');
        
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('deep-pavements-theme', newTheme);
            
            themeToggle.textContent = newTheme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
        });

        // Initialize saved theme
        const savedTheme = localStorage.getItem('deep-pavements-theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        themeToggle.textContent = savedTheme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
    </script>
</body>
</html>"""

    report_path = os.path.join(reports_path, "debug_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated debug HTML report: {report_path}")


def save_debug_image_metadata(
    image_id: str,
    filename: str,
    image_size: tuple[int, int],
    coordinates: Any,
    file_path: str,
    debug_metadata_path: str,
) -> None:
    """
    Save image metadata as JSON for debug inspection.

    Args:
        image_id: Mapillary image ID.
        filename: Original image filename.
        image_size: Image dimensions (width, height).
        coordinates: GPS coordinates (Point geometry or string).
        file_path: Path to the original image file.
        debug_metadata_path: Directory to save the metadata JSON.
    """
    metadata = {
        "image_id": image_id,
        "filename": filename,
        "original_size": image_size,
        "coordinates": (
            f"{coordinates.x}, {coordinates.y}"
            if hasattr(coordinates, "x")
            else str(coordinates)
        ),
        "file_path": file_path,
    }
    metadata_file = os.path.join(debug_metadata_path, f"{image_id}_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
