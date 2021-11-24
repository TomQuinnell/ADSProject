from .config import *

from .access import *

"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""
import osmnx as ox
import matplotlib.pyplot as plt
import mlai.plot as plot

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers 
encoded? What do columns represent, makes sure they are correctly labeled. How is the data indexed. Crete visualisation 
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""
default_pois = ("aerialway", "amenity", "building.", "emergency", "healthcare", "highway", "historic", "leisure",
                "office", "public transport", "railway")


def get_bounding_box(lat, lon, width=0.02, height=0.02):
    """ North, South, East, West of bounding box """
    return lat + height / 2, lat - height / 2, lon + width / 2, lon - width / 2


def get_lat_lon_for(place_name):
    # Gets the lat, lon pair for the geocode place name
    return ox.geocoder.geocode(place_name)


def get_tags(points_of_interest, poi_to_tag=None):
    tags = {}
    if poi_to_tag is None:
        poi_to_tag = {"aerialway": True, "amenity": True, "building.": True, "emergency": True,
                      "healthcare": True, "highway": ["motorway", "primary"], "historic": True,
                      "leisure": True, "office": True, "public transport": True, "railway": True}
    for poi in points_of_interest:
        tags_for_poi = poi_to_tag[poi]
        for tag in tags_for_poi:
            tags[tag] = True
    return tags


def get_points_of_interest(north, south, east, west, tags):
    return ox.geometries_from_bbox(north, south, east, west, tags)


def get_gdf_data_for(place_name, bb_width=0.02, bb_height=0.02):
    # Get lat, lon for place_name
    place_name = clean_place_name(place_name)
    lat, lon = get_lat_lon_for(place_name)

    # Create bounding box for area
    north, south, east, west = get_bounding_box(lat, lon, width=bb_width, height=bb_height)

    # Get graph data
    graph = ox.graph_from_bbox(north, south, east, west)
    nodes, edges = ox.graph_to_gdfs(graph)
    area = ox.geocode_to_gdf(place_name)
    return nodes, edges, area, [north, south, east, west]


def plot_on_map(nodes, edges, area, bbs):
    north, south, east, west = bbs
    fig, ax = plt.subplots(figsize=plot.big_figsize)

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot all POIs
    nodes.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.tight_layout()


def visualise_pois_near(place_name, poi_names=default_pois, bb_width=0.02, bb_height=0.02):
    # get gdf data
    nodes, edges, area, bbs = get_gdf_data_for(place_name, bb_width, bb_height)

    # get points of interest
    points_of_interest = get_points_of_interest(*bbs, get_tags(poi_names))

    plot_on_map(points_of_interest, edges, area, bbs)

    return nodes, edges, area, points_of_interest, bbs
