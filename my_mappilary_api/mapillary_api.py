import requests
import os
import json
import numpy as np
from shapely.geometry import Point, box
from math import cos, pi
import urllib
import wget
from time import sleep
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def get_coordinates_as_point(inputdict):
    """Convert geometry dict to Point object."""
    return Point(inputdict['coordinates'])


def check_type_by_first_valid(series):
    """Check the type of the first valid value in a series."""
    for value in series:
        if value is not None:
            return type(value)
    return None


def selected_columns_to_str(df, desired_type=list):
    """Convert columns of specified type to strings."""
    for column in df.columns:
        c_type = check_type_by_first_valid(df[column])
        
        if c_type == desired_type:
            df[column] = df[column].apply(lambda x: str(x))


def get_mapillary_images_metadata(minLon, minLat, maxLon, maxLat, token, outpath=None, params_dict=None):
    """
    Request images from Mapillary API given a bbox

    Parameters:
        minLon (float): The minimum longitude.
        minLat (float): The minimum latitude.
        maxLon (float): The maximum longitude.
        maxLat (float): The maximum latitude.
        token (str): The Mapillary API token.
        outpath (str, optional): Path to save the response.
        params_dict (dict, optional): Custom parameters for the API request.

    Returns:
        dict: A dictionary containing the response from the API.
    """
    url = "https://graph.mapillary.com/images"

    if not params_dict:
        params = {
            "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
            'limit': 5000,
            "access_token": token,
            "fields": ",".join([
                "altitude", 
                "atomic_scale", 
                "camera_parameters", 
                "camera_type", 
                "captured_at",
                "compass_angle", 
                "computed_altitude", 
                "computed_compass_angle", 
                "computed_geometry",
                "computed_rotation", 
                "creator", 
                "exif_orientation", 
                "geometry", 
                "height", 
                "make", 
                "model", 
                "thumb_original_url", 
                "sequence", 
                "sfm_cluster", 
                "width",
                "detections",
            ])
        }
    else:
        params = params_dict
        
    response = requests.get(url, params=params)
    as_dict = response.json()

    if outpath:
        with open(outpath, 'w') as f:
            json.dump(as_dict, f, indent=4, ensure_ascii=False)

    return as_dict


def download_mapillary_image(url, outfilepath, cooldown=1):
    """Download an image from a URL."""
    try:
        wget.download(url, out=outfilepath)
        if cooldown:
            sleep(cooldown)
    except Exception as e:
        print(f'Error downloading {url}: {e}')


def mapillary_data_to_gdf(data, outpath=None, filtering_polygon=None):
    """
    Convert Mapillary API response to GeoDataFrame.
    
    Parameters:
        data (dict or str): Mapillary API response or path to JSON file.
        outpath (str, optional): Path to save the GeoDataFrame.
        filtering_polygon (shapely.geometry, optional): Polygon to filter data.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Mapillary data.
    """
    if isinstance(data, str):
        with open(data, 'r') as f:
            data = json.load(f)

    if data.get('data'):
        as_df = pd.DataFrame.from_records(data['data'])

        if 'geometry' in as_df.columns:
            as_df.geometry = as_df.geometry.apply(get_coordinates_as_point)
            as_gdf = gpd.GeoDataFrame(as_df, crs='EPSG:4326', geometry='geometry')
            
            selected_columns_to_str(as_gdf)

            if filtering_polygon:
                as_gdf = as_gdf[as_gdf.intersects(filtering_polygon)]

            if outpath:
                as_gdf.to_file(outpath)

            return as_gdf
        else:
            return gpd.GeoDataFrame()
    else:
        return gpd.GeoDataFrame()


def download_all_pictures_from_gdf(gdf, output_dir, cooldown=1):
    """
    Download all images from a GeoDataFrame containing Mapillary data.
    
    Parameters:
        gdf (gpd.GeoDataFrame): GeoDataFrame with Mapillary image metadata.
        output_dir (str): Directory to save downloaded images.
        cooldown (int): Delay between downloads in seconds.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if 'thumb_original_url' not in gdf.columns:
        print("No 'thumb_original_url' column found in GeoDataFrame")
        return
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading images"):
        if pd.isna(row['thumb_original_url']):
            continue
            
        url = row['thumb_original_url']
        # Use the image ID as filename if available, otherwise use index
        if 'id' in row:
            filename = f"{row['id']}.jpg"
        else:
            filename = f"image_{idx}.jpg"
            
        filepath = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if not os.path.exists(filepath):
            download_mapillary_image(url, filepath, cooldown)
        else:
            print(f"File {filename} already exists, skipping...")