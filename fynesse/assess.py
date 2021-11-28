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
from shapely.geometry import Point

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers 
encoded? What do columns represent, makes sure they are correctly labeled. How is the data indexed. Crete visualisation 
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


""" Default point of interest tags """
default_pois = {"aerialway": True, "amenity": ["school", "education"], "building": ["religious", "sports"],
                "emergency": True, "healthcare": True, "highway": ["motorway", "primary"], "historic": True,
                "leisure": True, "office": True, "public transport": True, "railway": True}


def plot_on_map(nodes, edges, area, bbs):
    """
    Plot Map data
    :param nodes: nodes to plot
    :param edges: edges to plot
    :param area: region area to plot
    :param bbs: bounding box
    """
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
    if not nodes.empty:
        nodes.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.tight_layout()


def visualise_pois_near(place_name, poi_tags=default_pois, bb_width=0.02, bb_height=0.02):
    """
    Visualise points of interest near place_name with tags
    :param place_name: place name
    :param poi_tags: Point of Interest tags
    :param bb_width: bounding box width
    :param bb_height: bounding box height
    :return: nodes, edges, area for region, points of interest in this region, and bounding box
    """
    # get gdf data
    nodes, edges, area, bbs = get_gdf_data_for(place_name, bb_width, bb_height)

    # get points of interest
    points_of_interest = get_points_of_interest(*bbs, poi_tags)

    plot_on_map(points_of_interest, edges, area, bbs)

    return nodes, edges, area, points_of_interest, bbs


def filter_pois(df, column_selector, column_value=True):
    """
    Filter the points of interest from the query
    :param df: dataframe to filter
    :param column_selector: column to select
    :param column_value: values in column to select
    :return: the Points of interest for the column and value
    """
    try:
        non_null_df = df[df[column_selector].notnull()]
        if not column_value:
            return non_null_df[non_null_df[column_selector] == column_value]
        return non_null_df
    except KeyError:
        return pd.DataFrame([], columns=df.columns)


def plot_price_hist(df, house_type):
    """
    Plot a histogram of prices of a certain type
    :param df: dataframe of price data
    :param house_type: type of house to plot, if All then don't filter
    """
    if house_type != "All":
        df = df.loc[df['property_type'] == house_type]
    fig, ax = plt.subplots()
    ax.set_title("Price histogram for " + house_type)
    ax.hist(df['price'], 50, density=True)


def plot_house_prices(house_df, by_type=True):
    """
    Plot house prices
    :param house_df: house price data
    :param by_type: boolean indicating whether to plot by each type, or all at once
    """
    if by_type:
        plot_price_hist(house_df, "F")
        plot_price_hist(house_df, "S")
        plot_price_hist(house_df, "D")
        plot_price_hist(house_df, "T")
        plot_price_hist(house_df, "O")
    else:
        plot_price_hist(house_df, "All")


def closest_point(point, points):
    min_dist = 100000000
    closest = None
    for p in points:
        dist = point.distance(Point(points))
        if dist < min_dist:
            min_dist = dist
            closest = p
    return closest, min_dist


def get_poi_names(tags):
    poi_names = []
    for k in tags.keys():
        if tags[k] == True:
            poi_names.append(k)
        else:
            for v in tags[k]:
                poi_names.append(k + ";" + v)
    return poi_names


def get_points_of_interest_wthin_dist(pois, radius, point):
    pois['dist_to_point'] = list(map(lambda pt: point.distance(pt), list(pois['geometry'])))
    return pois.loc[pois['dist_to_point'] < radius]


def get_features_for_houses(houses, bbox, feature_radius=0.01):
    # get features for each house, storing in lookup table of bboxes
    poi_names = get_poi_names(default_pois)
    poi_lookup = {}
    feature_data = [[] for _ in range(len(poi_names))]
    region_pois = get_points_of_interest(*bbox, default_pois)
    for row in houses.itertuples():
        house_pos = (row.longitude, row.lattitude)
        closest_house, dist_closest = closest_point(Point(*house_pos), list(poi_lookup.keys()))
        if dist_closest < feature_radius:
            row_features = poi_lookup[closest_house]
        else:
            house_pois = get_points_of_interest_wthin_dist(region_pois, feature_radius, Point(*house_pos))
            row_features = [len(filter_pois(house_pois, poi_name.split(";")[0], poi_name.split(";")[-1]))
                            for poi_name in poi_names]
            poi_lookup[house_pos] = row_features

        for i, feature_col in enumerate(feature_data):
            feature_col.append(row_features[i])

    for i, feature_name in enumerate(poi_names):
        houses[feature_name] = feature_data[i]
