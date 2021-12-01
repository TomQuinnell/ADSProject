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
# constants used in the notebook
property_types = ["F", "S", "D", "T", "O"]
price_scale = 1000
onehot_true = 1
onehot_false = 0.0001


def type_onehot(df, property_type):
    """
    One hot vector of df.property_type == property_type
    :param df: Data Frame of house data
    :param property_type: type of property
    :return: This one hot vector
    """
    return np.where(df['property_type'] == property_type, onehot_true, onehot_false).reshape(-1, 1)


def append_type_onehots_from_df(features, df):
    """
    Append one hot vectors of property_type from df to features
    :param features: list to append one hot vectors to
    :param df: Data Frame of house data
    """
    for property_type in property_types:
        features.append(type_onehot(df, property_type))


def get_param_names(poi_names, added_features=False):
    """
    Get list of feature names
    :param poi_names: names of Points Of Interest
    :param added_features: boolean indicating whether to display added features (for later in notebook)
    :return: These names
    """
    param_names = [*poi_names, "lattitude", "longitude", "year - 1995", *["OnehotType" + property_type for property_type in property_types]]
    if added_features:
        param_names.append("Nearest")
    return param_names


def design_matrix(houses, poi_names, added_features=False):
    """
    Design matrix of features from houses Data Frame
    :param houses: Data Frame of house data
    :param poi_names: names of Points Of Interest
    :param added_features: boolean indicating whether to display added features (for later in notebook)
    :return: This design matrix - feature vectors stacked together
    """
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
    """
    For Prediction stage, append one hot vector to list of featyres, representing property_type choice
    :param features: List of features for Predictions
    :param n: Number of choices to make if none passed in
    :param chosen_property_types: Triggers random choice of property_type is None, else is the choice of property_types
    """
    if chosen_property_types is None:
        chosen_property_types = np.random.choice(property_types, size=n)
    chosen_property_types = np.array(chosen_property_types)
    for property_type in property_types:
        features.append(np.where(chosen_property_types == property_type, onehot_true, onehot_false).reshape(-1, 1))


def sample_normals(houses, poi_names, n):
    """
    Sample n normal distributions for each Point Of Interest type
    :param houses: Data Frame of house data
    :param poi_names: names of Points Of Interest
    :param n: number of samples
    :return: n samples from normal distributions for each Point Of Interest name
    """
    samples = []
    for poi_name in poi_names:
        poi_col = houses[poi_name]
        sampled = np.random.normal(poi_col.mean(), poi_col.std(), size=n)
        samples.append(np.where(sampled < 0, 0, sampled).reshape(-1, 1) + 0.0001)
    return samples


def nearest_price_feature(houses, lats, lons, add_noise=True):
    """
    Feature of nearest house price
    :param houses: Data Frame of house data
    :param lats: list of latitudes of house positions
    :param lons: list of longitudes of house positions
    :param add_noise: boolean indicating whether to add noise to price - not so empirical prediction
    :return: np array of Nearest Price feature (with an extra dimension to join up into Design Matrix)
    """
    feature = []
    # for each position
    for i in range(len(lats)):
        min_dist = 1000000
        price = 0
        lat = lats[i]
        lon = lons[i]
        # for each row in dataframe
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
    """
    Get Points Of Interest for many points
    :param lats: list of latitudes of positions
    :param lons: list of longitudes of  positions
    :param poi_tags: tags of Point Of Interest (see OSMNX documentation or assess for format)
    :param bbox_size: size of bounding box for Points Of Interest
    :return: Feature vectors for each Point Of Interest
    """
    poi_names = assess.get_poi_names(poi_tags)
    features = [[] for poi_name in poi_names]
    for i in range(len(lats)):
        lat = lats[i]
        lon = lons[i]
        pois = access.get_points_of_interest(*access.get_bounding_box(lat, lon, bbox_size, bbox_size), poi_tags)
        for j, poi_name in enumerate(poi_names):
            features[j].append(len(assess.filter_pois(pois, poi_name.split(";")[0], poi_name.split(";")[-1])))
    return [np.array([feature]).reshape(-1, 1) for feature in features]


def get_features(lats, lons, dates, property_type_list, houses, poi_tags, sample_norm_pois,
                 bbox_size, n=1, added_features=False):
    """
    Get feature vectors for Prediction stage
    :param lats: list of latitudes of house positions
    :param lons: list of longitudes of house positions
    :param dates: list of years of sales
    :param property_type_list: list of proprty_types
    :param houses: Data Frame of house sale data
    :param poi_tags: tags of Point Of Interest (see OSMNX documentation or assess for format)
    :param sample_norm_pois: boolean indicating whether to sample normals for Points Of Interest
            (Deprecated later in notebook)
    :param bbox_size: size of bounding box for Points Of Interest around Prediction positions
    :param n: number of samples for Points Of Interest
    :param added_features: boolean indicating whether to add extra features (See Notebook story)
    :return: These features concatenated together
    """
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
    """
    Train a linear model
    :param houses: Data Frame of house data
    :param poi_names: names of Points Of Interest to add
    :param added_features: boolean indicating whether to add extra features (See Notebook story)
    :return: fitted model
    """
    # create data
    x = design_matrix(houses, poi_names, added_features=added_features)
    y = np.array(houses['price']) / price_scale

    # create and fit model
    m_linear = sm.OLS(y, x)
    m_linear_fitted = m_linear.fit()
    return m_linear_fitted


def train_positive_linear_model(houses, poi_names, added_features=False):
    """
    Train a Poisson GLM model, with the log link function
    :param houses: Data Frame of house data
    :param poi_names: names of Points Of Interest to add
    :param added_features: boolean indicating whether to add extra features (See Notebook story)
    :return: fitted model
    """
    # create data
    x = design_matrix(houses, poi_names, added_features=added_features)
    y = np.array(houses['price']) / price_scale

    # create and fit model
    m_pos_linear = sm.GLM(y, x, family=sm.families.Poisson(link=sm.families.links.log()))
    m_pos_linear_fitted = m_pos_linear.fit()
    return m_pos_linear_fitted


def predict_many(lats, lons, dates, property_type_list, houses, poi_tags, model, sample_norm_pois, bbox_size, added_features):
    """
    Predict for many hypothetical house sales using a trained model
    :param lats: list of latitudes of house positions
    :param lons: list of longitudes of house positions
    :param dates: list of years of sales
    :param property_type_list: list of proprty_types
    :param houses: Data Frame of house sale data
    :param poi_tags: tags of Point Of Interest (see OSMNX documentation or assess for format)
    :param model: trained model with input shape matching feature shape
    :param sample_norm_pois: boolean indicating whether to sample normals for Points Of Interest
            (Deprecated later in notebook)
    :param bbox_size: size of bounding box for Points Of Interest around Prediction positions
    :param added_features: boolean indicating whether to add extra features (See Notebook story)
    :return: Prediction Object for features from this dataset
    """
    # get features for each point
    features = get_features(lats, lons, dates, property_type_list, houses, poi_tags, sample_norm_pois, bbox_size,
                            n=len(lats), added_features=added_features)

    # run model to get prediction for each feature set
    return model.get_prediction(np.array(features, dtype=float))


def predict_once(lat, lon, date, property_type, houses, poi_tags, model, sample_norm_pois=True, bbox_size=0.02, added_features=False):
    """
    Predict for a single hypothetical house sales using a trained model
    :param lat: latitudes of house position
    :param lon: of longitudes of house position
    :param date: years of sale
    :param property_type: type of property
    :param houses: Data Frame of house sale data
    :param poi_tags: tags of Point Of Interest (see OSMNX documentation or assess for format)
    :param model: trained model with input shape matching feature shape
    :param sample_norm_pois: boolean indicating whether to sample normals for Points Of Interest
            (Deprecated later in notebook)
    :param bbox_size: size of bounding box for Points Of Interest around Prediction positions
    :param added_features: boolean indicating whether to add extra features (See Notebook story)
    :return: Prediction Object for features from this dataset
    """
    return predict_many([lat], [lon], [date], [property_type], houses, poi_tags, model, sample_norm_pois, bbox_size, added_features)


def summarise_model_pred(model, pred, poi_names=assess.get_poi_names(assess.default_pois)):
    """
    Print summary of prediction, model parameters and parameter names for reference
    :param model: trained model with input shape matching feature shape
    :param pred: Prediction Object
    :param poi_names: name of Points Of Interest
    """
    print(pred.head())
    print(model.summary())
    print(get_param_names(poi_names))


def validate_latlon(lat, lon):
    """
    Validate latitude and longitude position to check if in UK range
    :param lat: latitude of position
    :param lon: longitude of position
    :raise: Exception if not within range. Prediction will not gather any data and thus not work.
    """
    if 50.10319 <= lat and lat <= 60.15456 and -7.64133 <= lon and lon <= 1.75159:
        return
    raise Exception("Latitude or Longitude does not lie in range for UK.")


def validate_date(date):
    """
    Validate date of sale to check is a string of year within training range (could extend in future)
    :param date: date to validate
    :raise: Exception if not within range. Prediction will not gather any data and thus not work.
    """
    if isinstance(date, str) and len(date) == 4 and "1995" <= date and date <= "2021":
        return
    else:
        raise Exception("date is not a string of year 1995-2021")


def validate_property_type(property_type):
    """
    Validate property type
    :param property_type: type of property to validate
    :raise: Exception if not a valida property type.
    """
    if property_type not in property_types:
        raise Exception("property type not valid. Should be an element in", property_types)


def warn_bad_prediction(pred, trained_model, houses):
    """
    Warn user if the Prediction is bad
    :param pred: Prediction Object
    :param trained_model: Model Trained on data
    :param houses: Data Frame of house sale data
    """
    # Get prices and prediction, scaling price if needed
    prices = np.array(houses['price'])
    n = len(prices)
    pred_val = list(pred['mean'])[0] * price_scale

    # few samples are statistically unreliable
    if n < 30:
        print("WARNING. Prediction may be inaccurate. Low number of training samples:", n)

    # check if Prediction mean and CI are within 2 +- STD of mean of data, and not outside data range respectively
    mean, std = prices.mean(), prices.std()
    min_val, max_val = prices.min(), prices.max()
    mean_ci_lower = list(pred['mean_ci_lower'])[0] * price_scale
    mean_ci_higher = list(pred['mean_ci_upper'])[0] * price_scale
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
    """
    Predict the price for a house sale, connecting to SQL data to gather features and train model for region to predict
    :param latitude: latitude of house sale to predict
    :param longitude: longitude of house sale to predict
    :param date: date of house sale to predict
    :param: property_type: type of house to predict
    :param: SQL Connection
    """
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
        # Expand bounding box and date range
        new_bounding_box_size = 0.2
        new_bounding_box = access.get_bounding_box(latitude, longitude, new_bounding_box_size, new_bounding_box_size)
        new_date_range_size = 2
        access.join_data_within_bbox_times(conn, new_bounding_box, access.get_date_range(date, new_date_range_size))
        houses = access.sql_to_df(conn, "prices_coordinates_data")

        # Try to choose a random sample from the property_type
        houses_with_type = houses.loc[houses['property_type'] == property_type]
        if houses_with_type.shape[0] != 0:
            print("No samples found with initial bounding box."
                  "Returning a guess from same property types in a larger bounding box", new_bounding_box)
            prices = np.array(houses_with_type['price'])
            return np.random.choice(prices)

        # Try to choose a random sample if none of the property_type
        if houses.shape[0] != 0:
            print("No samples found within initial bounding box."
                  "Returning a guess from all properties in a larger bounding box of", new_bounding_box)
            prices = np.array(houses_with_type['price'])
            return np.random.choice(prices)

        # Randomly guess between 100k and 500k
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
