# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""
import numpy as np
import statsmodels.api as sm
from fynesse import assess, access
from shapely.geometry import Point
import pandas as pd

"""Address a particular question that arises from the data"""
property_types = ["F", "S", "D", "T", "O"]
price_scale = 1000
onehot_true = 1
onehot_false = 0.0001


def type_onehot(df, property_type):
    return np.where(df['property_type'] == property_type, onehot_true, onehot_false).reshape(-1, 1)


def append_type_onehots_from_df(features, df):
    for property_type in property_types:
        features.append(type_onehot(df, property_type))


def get_param_names(poi_names, added_features=False):
    param_names = [*poi_names, "lattitude", "longitude", "year - 1995", *["OnehotType" + property_type for property_type in property_types]]
    if added_features:
        param_names.append("Nearest")
    return param_names


def design_matrix(houses, poi_names, added_features=False):
    vectors = [np.array(houses[poi_name]).reshape(-1, 1) + np.random.random() / 1000 for poi_name in poi_names]
    vectors.append(np.array(houses['lattitude']).reshape(-1, 1))
    vectors.append(np.array(houses['longitude']).reshape(-1, 1))
    vectors.append(np.array([date.year - 1995 for date in list(houses['date_of_transfer'])]).reshape(-1, 1))
    append_type_onehots_from_df(vectors, houses)
    if added_features:
        print("Adding Nearest Price feature...")
        vectors.append(nearest_price_feature(houses, list(houses['lattitude']), list(houses['longitude'])).reshape(-1, 1))
    return np.concatenate(vectors, axis=1)


def append_type_onehots(features, n, chosen_property_types=None):
    if chosen_property_types is None:
        chosen_property_types = np.random.choice(property_types, size=n)
    chosen_property_types = np.array(chosen_property_types)
    for property_type in property_types:
        features.append(np.where(chosen_property_types == property_type, onehot_true, onehot_false).reshape(-1, 1))


def sample_normals(houses, poi_names, n):
    samples = []
    for poi_name in poi_names:
        poi_col = houses[poi_name]
        sampled = np.random.normal(poi_col.mean(), poi_col.std(), size=n)
        samples.append(np.where(sampled < 0, 0, sampled).reshape(-1, 1) + 0.0001)
    return samples


def nearest_price_feature(houses, lats, lons, add_noise=True):
    feature = []
    for i in range(len(lats)):
        min_dist = 1000000
        price = 0
        lat = lats[i]
        lon = lons[i]
        for row in houses.itertuples():
            house_lat = row.lattitude
            house_lon = row.longitude
            dist_to_house = Point(lat, lon).distance(Point(house_lat, house_lon))
            if dist_to_house < min_dist:
                min_dist = dist_to_house
                price = row.price
        if add_noise:
            price -= np.random.random() * 100
        # stop overflow for exp
        price = max(price, 0)
        feature.append(np.log(price))
    return np.array([feature])


def get_pois_many(lats, lons, poi_tags, bbox_size):
    poi_names = assess.get_poi_names(poi_tags)
    features = [[] for poi_name in poi_names]
    for i in range(len(lats)):
        lat = lats[i]
        lon = lons[i]
        pois = access.get_points_of_interest(*access.get_bounding_box(lat, lon, bbox_size, bbox_size), poi_tags)
        for j, poi_name in enumerate(poi_names):
            features[j].append(len(assess.filter_pois(pois, poi_name.split(";")[0], poi_name.split(";")[-1])))
    return [np.array([feature]).reshape(-1, 1) for feature in features]


def get_features(lats, lons, dates, property_type_list, houses, poi_tags, sample_norm_pois, bbox_size, n=1, added_features=False):
    if sample_norm_pois:
        features = sample_normals(houses, assess.get_poi_names(poi_tags), n)
    else:
        features = get_pois_many(lats, lons, poi_tags, bbox_size)
    features.append(np.array(lats).reshape(-1, 1))
    features.append(np.array(lons).reshape(-1, 1))
    features.append((np.array(dates, dtype=float) - 1995).reshape(-1, 1))
    append_type_onehots(features, n, property_type_list)
    if added_features:
        features.append(np.array(nearest_price_feature(houses, lats, lons)).reshape(-1, 1))
    return np.concatenate(features, axis=1)


def train_linear_model(houses, poi_names, added_features=False):
    x = design_matrix(houses, poi_names, added_features=added_features)
    y = np.array(houses['price']) / price_scale

    m_linear = sm.OLS(y, x)
    m_linear_fitted = m_linear.fit()
    return m_linear_fitted


def train_positive_linear_model(houses, poi_names, added_features=False):
    x = design_matrix(houses, poi_names, added_features=added_features)
    y = np.array(houses['price']) / price_scale

    m_pos_linear = sm.GLM(y, x, family=sm.families.Poisson(link=sm.families.links.log()))
    m_pos_linear_fitted = m_pos_linear.fit()
    return m_pos_linear_fitted


def predict_many(lats, lons, dates, property_type_list, houses, poi_tags, model, sample_norm_pois, bbox_size, added_features):
    features = get_features(lats, lons, dates, property_type_list, houses, poi_tags, sample_norm_pois, bbox_size, n=len(lats), added_features=added_features)
    return model.get_prediction(np.array(features, dtype=float))


def predict_once(lat, lon, date, property_type, houses, poi_tags, model, sample_norm_pois=True, bbox_size=0.02, added_features=False):
    return predict_many([lat], [lon], [date], [property_type], houses, poi_tags, model, sample_norm_pois, bbox_size, added_features)


def summarise_model_pred(model, pred, poi_names=assess.get_poi_names(assess.default_pois)):
    print(pred.head())
    print(model.summary())
    print(get_param_names(poi_names))


def validate_latlon(lat, lon):
    if 50.10319 <= lat and lat <= 60.15456 and -7.64133 <= lon and lon <= 1.75159:
        return
    raise Exception("Latitude or Longitude does not lie in range for UK.")


def validate_date(date):
    if isinstance(date, str) and len(date) == 4 and "1995" <= date and date <= "2021":
        return
    else:
        raise Exception("date is not a string of year 1995-2021")


def validate_property_type(property_type):
    if property_type not in property_types:
        raise Exception("property type not valid. Should be an element in", property_types)


def warn_bad_prediction(pred, trained_model, houses):
    """
    Ideas:
        High std
        Differ from data range
        Few data samples
        Low probabliity of params?
    """
    prices = np.array(houses['price'])
    n = len(prices)
    pred_val = list(pred['mean'])[0] * price_scale

    if n < 30:
        print("WARNING. Prediction may be inaccurate. Low number of training samples:", n)

    mean, std = prices.mean(), prices.std()
    min_val, max_val = prices.min(), prices.max()
    mean_ci_lower, mean_ci_higher = list(pred['mean_ci_lower'])[0] * price_scale, list(pred['mean_ci_upper'])[0] * price_scale
    print()
    if pred_val >= mean + 2 * std or pred_val <= mean - 2 * std:
        print("WARNING. Prediction may be inaccurate. Prediction lies outside 2 std of mean of prices in region.")
    elif pred_val <= min_val or pred_val >= max_val:
        print("WARNING. Prediction may be inaccurate. Prediction lies outside of prices in region.")
    elif mean_ci_lower <= mean - 2 * std or mean_ci_higher >= mean + 2 * std:
        print("WARNING. Prediction may be inaccurate. Prediction CI lies outside 2 std of mean of prices in region")
    elif mean_ci_lower <= min_val or mean_ci_higher >= max_val:
        print("WARNING. Prediction may be inaccurate. Prediction CI lies outside of prices in region.")


def predict_price(latitude, longitude, date, property_type, conn):
    # Data validation
    validate_latlon(latitude, longitude)
    validate_date(date)
    validate_property_type(property_type)

    # Bounding box around lat and lon
    region_bbox_size = 0.05  # could have parameter to scale denseness of area?
    region_bbox = access.get_bounding_box(latitude, longitude, region_bbox_size, region_bbox_size)

    # Get date range, +- n years if still in range
    date_range = access.get_date_range(date, n=1)

    # For each postcode, for each year, add data
    access.join_data_within_bbox_times(conn, region_bbox, date_range)
    houses = access.sql_to_df(conn, "prices_coordinates_data")

    # If no postcode data found, print a warning and randomly guess
    if houses.shape[0] == 0:
        print("No samples found within bounding box. Returning a random guess")
        return np.random.random() * 400000 + 100000

    # Add POIs
    assess.get_features_for_houses(houses, region_bbox)

    # Train linear model
    # if too many samples, don't add nearest as it takes too long
    added_features = houses.shape[0] < 1250
    trained_model = train_positive_linear_model(houses, assess.get_poi_names(assess.default_pois),
                                                added_features=added_features)

    # Get prediction
    pred = predict_once(latitude, longitude, date, property_type, houses, assess.default_pois, trained_model,
                        added_features=added_features, sample_norm_pois=False).summary_frame(alpha=0.05)
    summarise_model_pred(trained_model, pred)

    # Warm if low quality prediction
    warn_bad_prediction(pred, trained_model, houses)

    # Return prediction - demux mean from prediction object
    return list(pred['mean'])[0] * price_scale
