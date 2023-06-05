import os
import pytest
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from src.smoothie_king_preprocess_data import *
from src.smoothie_king_build_model import *

'''
Test smoothie_king_preprocess_data.py
'''
def test_process_percent_non_percent_df():
    in_file = "data/Smoothie King/smoothie_king_trade_area_variables.csv"
    trade_area_df = pd.read_csv(in_file)
    process_percent_non_percent_df(trade_area_df, "Smoothie King/processed_trade_area.csv")
    out_file = "data/Smoothie King/processed_trade_area.csv"
    assert os.path.isfile(out_file)

def test_process_poi_df():
    in_file = "data/Smoothie King/smoothie_king_poi_variables.csv"
    poi_df = pd.read_csv(in_file)
    process_poi_df(poi_df, "Smoothie King/processed_poi.csv")
    out_file = "data/Smoothie King/processed_poi.csv"
    assert os.path.isfile(out_file)

'''
Test smoothie_king_build_model.py
'''
@pytest.fixture
def get_train_test_dfs():
    sk_df = read_data()
    sk_df = drop_unused_cols(sk_df)
    le = LabelEncoder()
    sk_df["category"] = le.fit_transform(sk_df["category"])
    train_df, test_df = train_test_split(sk_df, test_size=0.1, random_state=42)
    X_train = train_df.drop(columns=["category"])
    y_train = train_df["category"]
    X_test = test_df.drop(columns=["category"])
    y_test = test_df["category"]
    return X_train, y_train, X_test, y_test 

def test_read_data():
    merged_df = read_data()
    assert isinstance(merged_df, pd.DataFrame)
    assert "centerxy_full_0p5_intersect_gla" in merged_df.columns
    assert "emp_p_ta" in merged_df.columns

def test_drop_unused_cols():
    merged_df = read_data()
    modified_df = drop_unused_cols(merged_df)
    dropped_cols = {"store_num", "country_code", "longitude", "latitude", "state_name", "cbsa_name", "dma_name"}
    assert not modified_df.isnull().values.any() 
    assert not dropped_cols.issubset(modified_df.columns)

def test_define_preprocessor(get_train_test_dfs):
    X_train = get_train_test_dfs[0]
    preprocessor = define_preprocessor(X_train)
    assert isinstance(preprocessor, sklearn.compose.ColumnTransformer)

def test_build_random_forest_model(get_train_test_dfs):
    X_train = get_train_test_dfs[0]
    y_train = get_train_test_dfs[1]
    preprocessor = define_preprocessor(X_train)
    class_weight = None
    pipe_rf = build_random_forest_model(X_train, y_train, preprocessor, class_weight)
    assert isinstance(pipe_rf, sklearn.pipeline.Pipeline)
    assert "columntransformer" in pipe_rf.named_steps
    assert "randomforestclassifier" in pipe_rf.named_steps

def test_build_l1_reg_random_forest_model(get_train_test_dfs):
    X_train = get_train_test_dfs[0]
    y_train = get_train_test_dfs[1]
    preprocessor = define_preprocessor(X_train)
    class_weight = None
    pipe_lr_rf = build_l1_reg_random_forest_model(X_train, y_train, preprocessor, class_weight)
    assert isinstance(pipe_lr_rf, sklearn.pipeline.Pipeline)
    assert "columntransformer" in pipe_lr_rf.named_steps
    assert "selectfrommodel" in pipe_lr_rf.named_steps
    assert "randomforestclassifier" in pipe_lr_rf.named_steps

def test_build_l1_reg_random_forest_ovr_model(get_train_test_dfs):
    X_train = get_train_test_dfs[0]
    y_train = get_train_test_dfs[1]
    preprocessor = define_preprocessor(X_train)
    pipe_lr_rf_ovr = build_l1_reg_random_forest_ovr_model(X_train, y_train, preprocessor)
    assert isinstance(pipe_lr_rf_ovr, sklearn.pipeline.Pipeline)
    assert "columntransformer" in pipe_lr_rf_ovr.named_steps
    assert "selectfrommodel" in pipe_lr_rf_ovr.named_steps
    assert "onevsrestclassifier" in pipe_lr_rf_ovr.named_steps

def test_build_ensemble_model(get_train_test_dfs):
    X_train = get_train_test_dfs[0]
    y_train = get_train_test_dfs[1]
    preprocessor = define_preprocessor(X_train)
    class_weight = None
    pipe_rf = build_random_forest_model(X_train, y_train, preprocessor, class_weight)
    pipe_lr_rf = build_l1_reg_random_forest_model(X_train, y_train, preprocessor, class_weight)
    classifiers = {
        "classifier_1": pipe_rf,
        "classifier_2": pipe_lr_rf
    }
    hard_voting_model = build_ensemble_model(classifiers, X_train, y_train)
    assert isinstance(hard_voting_model, sklearn.ensemble.VotingClassifier)
    assert len(hard_voting_model.estimators_) == 2
