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
import pandas as pd

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers 
encoded? What do columns represent, makes sure they are correctly labeled. How is the data indexed. Crete visualisation 
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""
default_pois = {"aerialway": True, "amenity": ["school", "education"], "building": ["religious", "sports"],
                "emergency": True, "healthcare": True, "highway": ["motorway", "primary"], "historic": True,
                "leisure": True, "office": True, "public transport": True, "railway": True}


def get_bounding_box(lat, lon, width=0.02, height=0.02):
    """ North, South, East, West of bounding box """
    return lat + height / 2, lat - height / 2, lon + width / 2, lon - width / 2


def get_lat_lon_for(place_name):
    # Gets the lat, lon pair for the geocode place name
    return ox.geocoder.geocode(place_name)


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


def visualise_pois_near(place_name, poi_tags=default_pois, bb_width=0.02, bb_height=0.02):
    # get gdf data
    nodes, edges, area, bbs = get_gdf_data_for(place_name, bb_width, bb_height)

    # get points of interest
    points_of_interest = get_points_of_interest(*bbs, poi_tags)

    plot_on_map(points_of_interest, edges, area, bbs)

    return nodes, edges, area, points_of_interest, bbs


def filter_pois(df, column_selector, column_value=True):
    try:
        non_null_df = df[df[column_selector].notnull()]
        if not column_value:
            return non_null_df[non_null_df[column_selector] == column_value]
        return non_null_df
    except KeyError:
        return pd.DataFrame([], columns=df.columns)


def plot_price_hist(df):
    plt.hist(df['price'], 50, density=True)


def plot_house_prices(house_df, by_type=True):
    if by_type:
        plot_price_hist(house_df.loc[house_df["property_type"] == "F"])
        plot_price_hist(house_df.loc[house_df["property_type"] == "S"])
        plot_price_hist(house_df.loc[house_df["property_type"] == "D"])
        plot_price_hist(house_df.loc[house_df["property_type"] == "T"])
        plot_price_hist(house_df.loc[house_df["property_type"] == "O"])
    else:
        plot_price_hist(house_df)
