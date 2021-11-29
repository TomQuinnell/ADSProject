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
    param_names = [*poi_names, "lattitude", "longitude", *["OnehotType" + property_type for property_type in property_types]]
    if added_features:
        param_names.append("Nearest")
    return param_names


def design_matrix(houses, poi_names, added_features=False):
    vectors = [np.array(houses[poi_name]).reshape(-1, 1) + np.random.random() / 1000 for poi_name in poi_names]
    vectors.append(np.array(houses['lattitude']).reshape(-1, 1))
    vectors.append(np.array(houses['longitude']).reshape(-1, 1))
    vectors.append(np.array([date[:4] for date in list(houses['date_of_transfer'])))
    print(vectors)
    append_type_onehots_from_df(vectors, houses)
    if added_features:
        print("adding features...")
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
    features.append(np.array(dates).reshape(-1, 1))
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
    return model.get_prediction(features)


def predict_once(lat, lon, date, property_type, houses, poi_tags, model, sample_norm_pois=True, bbox_size=0.02, added_features=False):
    return predict_many([lat], [lon], [date], [property_type], houses, poi_tags, model, sample_norm_pois, bbox_size, added_features)


def summarise_model_pred(model, pred, poi_names=assess.get_poi_names(assess.default_pois)):
    print(pred.head())
    print(model.summary())
    print(get_param_names(poi_names))