import os
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel

DIR = 'data/Smoothie King/'

def read_data():
    # read in the POI, stores, and trade area files, merge them, and set the index to the store ID
    poi = pd.read_csv(DIR + "processed_poi.csv")
    stores = pd.read_csv(DIR + "smoothie_king_stores.csv")
    trade_area = pd.read_csv(DIR + "processed_trade_area.csv")
    merged = stores.merge(trade_area, left_on="store", right_on="store_num").merge(poi)
    merged = merged.set_index("store")
    return merged

def drop_unused_cols(df):
    drop_cols = ["store_num", "country_code", "longitude", "latitude", "state_name", "cbsa_name", "dma_name"]
    df = df.drop(columns=drop_cols)
    df = df.dropna()
    return df

def define_preprocessor(X_train):
    market_levels = [
        "Small Town (6)",
        "Small City (5)",
        "Medium City (4)",
        "Large City (3)",
        "Large Metro (2)",
        "Very Large Metro (1)"
    ]
    density_levels = [
        "Rural",
        "Exurban",
        "Suburban",
        "Light Suburban",
        "Light Urban",
        "Urban",
        "Super Urban"
    ]
    ordinal_features = ["market_size", "store_density"]
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    ordinal_transformer = OrdinalEncoder(categories=[market_levels, density_levels], dtype=int)
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (ordinal_transformer, ordinal_features)
    )
    return preprocessor

def build_random_forest_model(X_train, y_train, preprocessor, class_weight):
    pipe_rf = make_pipeline(
        preprocessor,
        RandomForestClassifier(
            n_estimators=100, 
            max_depth=30, 
            max_leaf_nodes=30, 
            class_weight=class_weight, 
            min_samples_leaf=10, 
            min_samples_split=30, 
            n_jobs=-1, 
            random_state=42
        )
    )
    pipe_rf.fit(X_train, y_train)
    return pipe_rf

def build_l1_reg_random_forest(X_train, y_train, preprocessor, class_weight):
    pipe_lr_rf = make_pipeline(
        preprocessor,
        SelectFromModel(
            LogisticRegression(
                C=0.1, 
                penalty="l1",
                solver="saga", 
                max_iter=10000, 
                multi_class="ovr", 
                class_weight=None,
                n_jobs=-1, 
                random_state=42
            )
        ),
        RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            max_leaf_nodes=70, 
            class_weight=class_weight, 
            min_samples_leaf=10, 
            min_samples_split=30, 
            n_jobs=-1, 
            random_state=42
        )
    )
    pipe_lr_rf.fit(X_train, y_train)
    return pipe_lr_rf

def build_l1_reg_random_forest_ovr(X_train, y_train, preprocessor):
    pipe_lr_rf_ovr = make_pipeline(
        preprocessor,
        SelectFromModel(
            LogisticRegression(
                C=0.15, 
                penalty="l1",
                solver="saga", 
                max_iter=10000, 
                multi_class="multinomial", 
                class_weight=None,
                n_jobs=-1,
                random_state=42
            )
        ),
        OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=100, 
                max_depth=30, 
                max_leaf_nodes=30, 
                class_weight=None, 
                min_samples_leaf=10, 
                min_samples_split=30, 
                n_jobs=-1, 
                random_state=42
            )
        )
    )
    pipe_lr_rf_ovr.fit(X_train, y_train)
    return pipe_lr_rf_ovr

def build_ensemble_model(classifiers, X_train, y_train):
    hard_voting_model = VotingClassifier(
        list(classifiers.items()), voting="hard"
    )
    hard_voting_model.fit(X_train, y_train)
    return hard_voting_model

def print_train_test_scores(classifiers, X_train, y_train, X_test, y_test):
    for classifier_name, model in classifiers.items():
        print(f"{classifier_name} train score: {model.score(X_train, y_train)}")
        print(f"{classifier_name} test score: {model.score(X_test, y_test)}")

def main():
    sk_df = read_data()
    sk_df = drop_unused_cols(sk_df)
    le = LabelEncoder()
    sk_df["category"] = le.fit_transform(sk_df["category"])
    class_weight = {
        "HOME": 0.24,
        "OTHER": 0.16,
        "SHOPPING": 0.22,
        "TRAVEL": 0.15,
        "WORK": 0.23
    }
    class_weight = {i: class_weight[label] for i, label in enumerate(le.classes_)}
    train_df, test_df = train_test_split(sk_df, test_size=0.1, random_state=42)
    X_train = train_df.drop(columns=["category"])
    y_train = train_df["category"]
    X_test = test_df.drop(columns=["category"])
    y_test = test_df["category"]
    preprocessor = define_preprocessor(X_train)
    rf_model = build_random_forest_model(X_train, y_train, preprocessor, class_weight)
    l1_reg_rf_model = build_l1_reg_random_forest(X_train, y_train, preprocessor, class_weight)
    l1_reg_rf_ovr_model = build_l1_reg_random_forest_ovr(X_train, y_train, preprocessor)
    classifiers = {
        "Random Forest": rf_model,
        "L1 Regularization Random Forest": l1_reg_rf_model,
        "L1 Regularization Random Forest OVR": l1_reg_rf_ovr_model
    }
    hard_voting_model = build_ensemble_model(classifiers, X_train, y_train)
    classifiers["Ensemble"] = hard_voting_model
    print_train_test_scores(classifiers, X_train, y_train, X_test, y_test)
    try:
        dump(rf_model, "model_joblib/rf_model.joblib")
        dump(l1_reg_rf_model, "model_joblib/l1_reg_rf_model.joblib")
        dump(l1_reg_rf_ovr_model, "model_joblib/l1_reg_rf_ovr_model.joblib")
        dump(hard_voting_model, "model_joblib/hard_voting_model.joblib")
    except:
        os.mkdir("model_joblib/")
        dump(rf_model, "model_joblib/rf_model.joblib")
        dump(l1_reg_rf_model, "model_joblib/l1_reg_rf_model.joblib")
        dump(l1_reg_rf_ovr_model, "model_joblib/l1_reg_rf_ovr_model.joblib")
        dump(hard_voting_model, "model_joblib/hard_voting_model.joblib")
    train_df["category"] = le.inverse_transform(train_df["category"])
    test_df["category"] = le.inverse_transform(test_df["category"])
    train_df.to_csv(DIR + "train_df.csv")
    test_df.to_csv(DIR + "test_df.csv")

if __name__ == "__main__":
    main()