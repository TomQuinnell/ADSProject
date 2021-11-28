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

"""Address a particular question that arises from the data"""
property_types = ["F", "S", "D", "T", "O"]

def type_onehot(df, property_type):
    return np.where(df['property_type'] == property_type, 1, 0).reshape(-1, 1)


def append_type_onehots_from_df(features, df):
    for property_type in property_types:
        features.append(type_onehot(df, property_type))


def design_matrix(houses, poi_names):
    vectors = [houses[poi_name].reshape(-1, 1) for poi_name in poi_names]
    vectors.append(houses['lattitude'].reshape(-1, 1))
    vectors.append(houses['longitude'].reshape(-1, 1))
    append_type_onehots_from_df(vectors, houses)

    return np.concatenate(vectors, axis=1)


def append_type_onehots(features, n):
    chosen_propety_types = np.random.choice(property_types, size=n)
    for propety_type in property_types:
        features.append(np.where(chosen_propety_types == propety_type, 1, 0).reshape(-1, 1))


def sample_normals(houses, poi_names, n):
    samples = []
    for poi_name in poi_names:
        poi_col = houses[poi_name]
        samples.append(np.random.normal(poi_col.mean(), poi_col.std(), size=n).reshape(-1, 1))
    return samples


def get_features(lats, lons, houses, poi_names, n=1):
    features = sample_normals(houses, poi_names, n)
    features.append(np.array(lats).reshape(-1, 1))
    features.append(np.array(lons).reshape(-1, 1))
    append_type_onehots(features, n)
    return np.concatenate(features, axis=1)


def train_linear_model(houses, poi_names):
    x = design_matrix(houses, poi_names)
    y = houses['prices']

    m_linear = sm.OLS(y, x)
    m_linear_fitted = m_linear.fit()
    return m_linear_fitted


def predict_many(lats, lons, houses, poi_names, model):
    features = get_features(lats, lons, houses, poi_names, n=len(lats))
    return model.get_prediction(features)


def predict_once(lat, lon, houses, poi_names, model):
    return predict_many([lat], [lon], houses, poi_names, model)
