import os
import pytest
import numpy as np
import pandas as pd
import skfda
import sklearn
import shap

import sys
cwd = os.getcwd()
sys.path.append(cwd + '/src/')
from subway_usa_preprocess_data import *
from subway_usa_build_model import *
import subway_usa_model_interpret as sb_inter

'''
Test subway_usa_preprocess_data.py 
'''
@pytest.fixture
def read_merged_data():
    poi_df = pd.read_csv("data/Subway USA/subway_usa_poi_variables.csv")
    trade_area_df = pd.read_csv("data/Subway USA/subway_usa_trade_area_variables.csv")
    stores_df = pd.read_csv("data/Subway USA/subway_usa_stores.csv", encoding='latin-1')
    merged_df = stores_df.merge(poi_df, on="store").merge(trade_area_df, on="store")
    return merged_df

def test_agg_inrix(read_merged_data):
    df = read_merged_data
    inrix_columns = set()
    for col in df.columns.tolist():
        if 'inrix_' in col:
            inrix_columns.add(col)
    processed_df = agg_inrix(df)
    assert "inrix_total_ta" in processed_df.columns
    assert not inrix_columns.issubset(processed_df.columns)

def test_agg_veh(read_merged_data):
    df = read_merged_data
    processed_df = agg_veh(df)
    vehicle_cols = {
        'hh_0vehicle_p_ta', 'hh_1vehicle_p_ta', 'hh_2vehicle_p_ta',
        'hh_3vehicle_p_ta', 'hh_4vehicle_p_ta', 'hh_5vehicle_p_ta'
    }
    assert "hh_expected_vehicle_ta" in processed_df.columns
    assert not vehicle_cols.issubset(processed_df.columns)

def test_agg_hh_pers(read_merged_data):
    df = read_merged_data
    processed_df = agg_hh_pers(df)
    hh_per_cols = {
        'hh_1pers_p_ta', 'hh_2pers_p_ta', 'hh_3pers_p_ta', 'hh_4pers_p_ta', 
        'hh_5pers_p_ta', 'hh_6pers_p_ta', 'hh_7pers_p_ta'
    }
    assert "hh_expected_pers_ta" in processed_df.columns
    assert not hh_per_cols.issubset(processed_df.columns)

def test_drop_specific_columns(read_merged_data):
    df = read_merged_data
    processed_df = drop_specific_columns(df)
    removed_cols = {
        "military", "sports_venues", "centerxy_nearest_dist", "emp_management_p"
    }
    exist_cols = {
        "emp_p_ta", "hh_expected_vehicle_ta", "hh_expected_pers_ta"
    }
    assert not removed_cols.issubset(processed_df.columns)
    assert exist_cols.issubset(processed_df.columns)

def test_merge_split_data():
    poi_df = pd.read_csv("data/Subway USA/subway_usa_poi_variables.csv")
    trade_area_df = pd.read_csv("data/Subway USA/subway_usa_trade_area_variables.csv")
    stores_df = pd.read_csv("data/Subway USA/subway_usa_stores.csv", encoding='latin-1')
    assert len(merge_split_data(stores_df, poi_df, trade_area_df)) == 2

def test_data_transform_pipeline():
    poi_df = pd.read_csv("data/Subway USA/subway_usa_poi_variables.csv")
    trade_area_df = pd.read_csv("data/Subway USA/subway_usa_trade_area_variables.csv")
    stores_df = pd.read_csv("data/Subway USA/subway_usa_stores.csv", encoding='latin-1')
    train_df, test_df = merge_split_data(stores_df, poi_df, trade_area_df)
    drop_features = ["store", "longitude", "latitude", "cbsa_name", "dma_name", "censusdivision", "censusregion"]
    ordinal_features_oth = [
        "market_size",
        "store_density",
    ]
    ordering_ordinal_oth = [
        ["Very Large Metro (1)", "Large Metro (2)", "Large City (3)", "Medium City (4)", "Small City (5)",
         "Small Town (6)", "Small Community (7)"],
        ["Rural", "Exurban", "Suburban", "Light Suburban", "Light Urban", "Urban", "Super Urban"],
    ]
    numeric_features = list(set(train_df.select_dtypes(include=np.number).columns.tolist()) - set(drop_features))
    assert len(data_transform_pipeline(
        train_df, test_df, drop_features, ordinal_features_oth, ordering_ordinal_oth, [], numeric_features
    )) == 2
    processed_train, processed_test = data_transform_pipeline(
        train_df, test_df, drop_features, ordinal_features_oth, ordering_ordinal_oth, [], numeric_features
    )
    assert not set(drop_features).issubset(processed_train.columns)

def test_process_subway_usa():
    process_subway_usa()
    out_dir = "data/Subway_USA_Preprocessed/"
    assert os.path.isfile(out_dir + "subway_usa_processed_train.csv")
    assert os.path.isfile(out_dir + "subway_usa_processed_test.csv")

'''
Test subway_usa_build_model.py
'''
@pytest.fixture
def read_processed_data():
    train_df, test_df, stores = read_data()
    train_df, test_df = select_features(train_df, test_df)
    return train_df, test_df, stores

def test_read_data():
    assert len(read_data()) == 3

def test_select_features():
    train_df, test_df, stores = read_data()
    original_cols = train_df.columns
    reduced_train, reduced_test = select_features(train_df, test_df)
    reduced_cols = reduced_train.columns
    assert len(reduced_cols) < len(original_cols)

def test_build_fcm_model(read_processed_data):
    train_df, test_df = read_processed_data[0], read_processed_data[1]
    fcm = build_fcm_model(train_df)
    assert isinstance(fcm, skfda.ml.clustering.FuzzyCMeans)

def test_add_labels(read_processed_data):
    train_df, test_df = read_processed_data[0], read_processed_data[1]
    fcm = build_fcm_model(train_df)
    labelled_train, labelled_test = add_labels(train_df, test_df, fcm)
    assert "labels" in labelled_train.columns
    assert "labels" in labelled_test.columns

def test_get_random_sample(read_processed_data):
    train_df, test_df, stores = read_processed_data[0], read_processed_data[1], read_processed_data[2]
    fcm = build_fcm_model(train_df)
    train_random_samples, test_random_samples = get_random_sample(5, train_df, test_df, stores, fcm)
    assert len(train_random_samples.keys()) == 5
    assert len(train_random_samples[0]) == 30
    assert len(test_random_samples.keys()) == 5
    assert len(test_random_samples[0]) == 30

'''
Test subway_usa_model_interpret.py
'''
@pytest.fixture
def read_labelled_data():
    train_df, test_df = sb_inter.read_data()
    X_train = train_df.drop(columns=["labels"])
    y_train = train_df["labels"]
    X_test = test_df.drop(columns=["labels"])
    y_test = test_df["labels"]
    return X_train, y_train, X_test, y_test

def test_model_interpret_read_data():
    assert(len(sb_inter.read_data()) == 2)

def test_fit_random_forest_classifier():
    train_df, test_df = sb_inter.read_data()
    X_train = train_df.drop(columns=["labels"])
    y_train = train_df["labels"]
    rf = sb_inter.fit_random_forest_classifier(X_train, y_train)
    assert isinstance(rf, sklearn.ensemble.RandomForestClassifier)

def test_plot_shap_feature_importance(read_labelled_data):
    X_train, y_train, X_test, y_test = read_labelled_data[0], read_labelled_data[1], read_labelled_data[2], read_labelled_data[3]
    rf = sb_inter.fit_random_forest_classifier(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    sb_inter.plot_shap_feature_importance(shap_values[0], X_test, 0)
    assert os.path.isfile("img/subway_usa/cluster_0_summary_plot.png")

def test_plot_shap_force_plot(read_labelled_data):
    X_train, y_train, X_test, y_test = read_labelled_data[0], read_labelled_data[1], read_labelled_data[2], read_labelled_data[3]
    rf = sb_inter.fit_random_forest_classifier(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    y_test_index_reset = y_test.reset_index(drop=True)
    indices = y_test_index_reset[y_test_index_reset == 0].index.tolist()
    pred_probs = rf.predict_proba(X_test.iloc[indices])
    most_confident_pred_idx = np.argmax(pred_probs[:, 0])
    sb_inter.plot_shap_force_plot(explainer, shap_values[0], X_test, 0, indices[most_confident_pred_idx])
    assert os.path.isfile("img/subway_usa/cluster_0_force_plot.png")
