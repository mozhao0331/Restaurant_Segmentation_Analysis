import os
import pytest
import pandas as pd
import numpy as np
import sklearn
import shap
from joblib import load
from sklearn.preprocessing import LabelEncoder
from src.smoothie_king_preprocess_data import *
from src.smoothie_king_build_model import *
import src.smoothie_king_model_interpret as sk_inter

'''
Test smoothie_king_preprocess_data.py
'''
def test_process_percent_non_percent_df():
    in_file = "data/Smoothie King/smoothie_king_trade_area_variables.csv"
    trade_area_df = pd.read_csv(in_file)
    process_percent_non_percent_df(trade_area_df, "Smoothie_King_Preprocessed/processed_trade_area.csv")
    out_file = "data/Smoothie_King_Preprocessed/processed_trade_area.csv"
    assert os.path.isfile(out_file)

def test_process_poi_df():
    in_file = "data/Smoothie King/smoothie_king_poi_variables.csv"
    poi_df = pd.read_csv(in_file)
    process_poi_df(poi_df, "Smoothie_King_Preprocessed/processed_poi.csv")
    out_file = "data/Smoothie_King_Preprocessed/processed_poi.csv"
    assert os.path.isfile(out_file)

'''
Test smoothie_king_build_model.py
'''
@pytest.fixture
def get_train_test_dfs_for_build():
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

def test_define_preprocessor(get_train_test_dfs_for_build):
    X_train = get_train_test_dfs_for_build[0]
    preprocessor = define_preprocessor(X_train)
    assert isinstance(preprocessor, sklearn.compose.ColumnTransformer)

def test_build_random_forest_model(get_train_test_dfs_for_build):
    X_train = get_train_test_dfs_for_build[0]
    y_train = get_train_test_dfs_for_build[1]
    preprocessor = define_preprocessor(X_train)
    class_weight = None
    pipe_rf = build_random_forest_model(X_train, y_train, preprocessor, class_weight)
    assert isinstance(pipe_rf, sklearn.pipeline.Pipeline)
    assert "columntransformer" in pipe_rf.named_steps
    assert "randomforestclassifier" in pipe_rf.named_steps

def test_build_l1_reg_random_forest_model(get_train_test_dfs_for_build):
    X_train = get_train_test_dfs_for_build[0]
    y_train = get_train_test_dfs_for_build[1]
    preprocessor = define_preprocessor(X_train)
    class_weight = None
    pipe_lr_rf = build_l1_reg_random_forest_model(X_train, y_train, preprocessor, class_weight)
    assert isinstance(pipe_lr_rf, sklearn.pipeline.Pipeline)
    assert "columntransformer" in pipe_lr_rf.named_steps
    assert "selectfrommodel" in pipe_lr_rf.named_steps
    assert "randomforestclassifier" in pipe_lr_rf.named_steps

def test_build_l1_reg_random_forest_ovr_model(get_train_test_dfs_for_build):
    X_train = get_train_test_dfs_for_build[0]
    y_train = get_train_test_dfs_for_build[1]
    preprocessor = define_preprocessor(X_train)
    pipe_lr_rf_ovr = build_l1_reg_random_forest_ovr_model(X_train, y_train, preprocessor)
    assert isinstance(pipe_lr_rf_ovr, sklearn.pipeline.Pipeline)
    assert "columntransformer" in pipe_lr_rf_ovr.named_steps
    assert "selectfrommodel" in pipe_lr_rf_ovr.named_steps
    assert "onevsrestclassifier" in pipe_lr_rf_ovr.named_steps

def test_build_ensemble_model(get_train_test_dfs_for_build):
    X_train = get_train_test_dfs_for_build[0]
    y_train = get_train_test_dfs_for_build[1]
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

'''
Test smoothie_king_model_interpret.py
'''

@pytest.fixture
def get_train_test_dfs_for_interpret():
    train_df, test_df = sk_inter.read_data()
    le = LabelEncoder() 
    train_df["category"] = le.fit_transform(train_df["category"])
    test_df["category"] = le.transform(test_df["category"])
    X_train = train_df.drop(columns=["category"])
    y_train = train_df["category"]
    X_test = test_df.drop(columns=["category"])
    y_test = test_df["category"]
    return X_train, y_train, X_test, y_test

@pytest.fixture
def load_models():
    MODEL_DIR = "model_joblib/smoothie_king/"
    rf_model = load(MODEL_DIR + "rf_model.joblib")
    l1_reg_rf_model = load(MODEL_DIR + "l1_reg_rf_model.joblib")
    l1_reg_rf_ovr_model = load(MODEL_DIR + "l1_reg_rf_ovr_model.joblib")
    hard_voting_model = load(MODEL_DIR + "hard_voting_model.joblib")
    return rf_model, l1_reg_rf_model, l1_reg_rf_ovr_model, hard_voting_model

def test_generate_confuson_matrix(get_train_test_dfs_for_interpret, load_models):
    X_train = get_train_test_dfs_for_interpret[0]
    y_train = get_train_test_dfs_for_interpret[1]
    hard_voting_model = load_models[3]
    classes = [0, 1, 2, 3, 4]
    sk_inter.generate_confuson_matrix(hard_voting_model, X_train, y_train, classes, "TITLE", "train_cm.png")
    out_file = "img/smoothie_king/train_cm.png"
    assert os.path.isfile(out_file)

def test_get_prediction_mismatch(get_train_test_dfs_for_interpret, load_models):
    X_test = get_train_test_dfs_for_interpret[2]
    y_test = get_train_test_dfs_for_interpret[3]
    rf_model = load_models[0]
    predictions = pd.DataFrame({
        "true_label": y_test,
        "rf": rf_model.predict(X_test)
    })
    mismatched = sk_inter.get_prediction_mismatch(predictions, "rf", 2, 1)
    assert isinstance(mismatched, list)

def test_get_all_feature_names(get_train_test_dfs_for_interpret):
    X_train = get_train_test_dfs_for_interpret[0]
    all_feature_names = sk_inter.get_all_feature_names(X_train)
    assert isinstance(all_feature_names, list)

def test_shap_summary_plot(get_train_test_dfs_for_interpret, load_models):
    X_train = get_train_test_dfs_for_interpret[0]
    X_test = get_train_test_dfs_for_interpret[2]
    rf_model = load_models[0]
    all_feature_names = sk_inter.get_all_feature_names(X_train)
    X_test_enc = sk_inter.encode_X_test(rf_model, X_test, all_feature_names)
    explainer = shap.TreeExplainer(rf_model.named_steps["randomforestclassifier"])
    shap_values = explainer.shap_values(X_test_enc)
    sk_inter.shap_summary_plot(
        strategy_ovr=False,
        model_name="Random Forest",
        out_dir="img/smoothie_king/random_forest/",
        shap_values=shap_values,
        X_test_enc=X_test_enc
    )
    out_file = "img/smoothie_king/random_forest/HOME_shap_summary_plot.png"
    assert os.path.isfile(out_file)

def test_shap_force_plot(get_train_test_dfs_for_interpret, load_models):
    X_train = get_train_test_dfs_for_interpret[0]
    X_test = get_train_test_dfs_for_interpret[2]
    y_test = get_train_test_dfs_for_interpret[3]
    rf_model = load_models[0]
    all_feature_names = sk_inter.get_all_feature_names(X_train)
    X_test_enc = sk_inter.encode_X_test(rf_model, X_test, all_feature_names)
    explainer = shap.TreeExplainer(rf_model.named_steps["randomforestclassifier"])
    shap_values = explainer.shap_values(X_test_enc)
    y_test_index_reset = y_test.reset_index(drop=True)
    indices = y_test_index_reset[y_test_index_reset == 0].index.tolist()
    pred_probs = rf_model.predict_proba(X_test.iloc[indices])
    most_confident_pred_idx = np.argmax(pred_probs[:, 0])
    sk_inter.shap_force_plot(
        strategy_ovr=False,
        explainer=explainer,
        shap_values=shap_values,
        X_test_enc=X_test_enc,
        class_indices=indices,
        idx_to_explain=most_confident_pred_idx,
        target_class=0,
        out_dir="img/smoothie_king/random_forest/",
        title="Most Confident Prediction"
    )
    out_file = "img/smoothie_king/random_forest/most_confident_prediction_HOME_shap_force_plot.png"
    assert os.path.isfile(out_file)
