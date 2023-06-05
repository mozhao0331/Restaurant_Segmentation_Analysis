''' 
Script that fits three models that are then passed into a voting classifier ensemble model.
Saves the model into joblib files that can be loaded for later use.
'''

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
    '''Read in the POI, stores, and trade area files, merge them, and set the index to the store ID.

    Returns
    -------
    pandas DataFrame
    '''
    poi = pd.read_csv(DIR + "processed_poi.csv")
    stores = pd.read_csv(DIR + "smoothie_king_stores.csv")
    trade_area = pd.read_csv(DIR + "processed_trade_area.csv")
    merged = stores.merge(trade_area, left_on="store", right_on="store_num").merge(poi)
    merged = merged.set_index("store")
    return merged

def save_train_test_df(train_df, test_df):
    train_df.to_csv(DIR + "train_df.csv")
    test_df.to_csv(DIR + "test_df.csv")

def drop_unused_cols(df):
    '''Drops any unused columns and rows with missing entries.

    Parameters
    ----------
    df : pandas DataFrame

    Returns
    -------
    pandas DataFrame
    '''
    drop_cols = ["store_num", "country_code", "longitude", "latitude", "state_name", "cbsa_name", "dma_name"]
    df = df.drop(columns=drop_cols)
    df = df.dropna()
    return df

def define_preprocessor(X_train):
    '''Create the preprocessor for the classification model by applying scaling and ordinal encoding.

    Parameters
    ----------
    X_train : pandas DataFrame
        The training dataset without the target column
    
    Returns
    -------
    sklearn ColumnTransformer
    '''
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
    '''Train a random forest classifier on the dataset.

    Parameters
    ----------
    X_train : pandas DataFrame
        The training dataset without the target column
    y_train : pandas Series
        The target column of the train set to predict
    preprocessor : sklearn ColumnTransformer
        Preprocessor to apply before fitting the classifier
    class_weight : dict
        Weight to scale each target class
    
    Returns
    -------
    sklearn Pipeline
        Pipeline containing the fitted classifier
    '''
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

def build_l1_reg_random_forest_model(X_train, y_train, preprocessor, class_weight):
    '''Perform feature selection using L1 regularization and train a random forest classifier on the dataset.

    Parameters
    ----------
    X_train : pandas DataFrame
        The training dataset without the target column
    y_train : pandas Series
        The target column of the train set to predict
    preprocessor : sklearn ColumnTransformer
        Preprocessor to apply before fitting the classifier
    class_weight : dict
        Weight to scale each target class
    
    Returns
    -------
    sklearn Pipeline
        Pipeline containing the fitted classifier
    '''
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

def build_l1_reg_random_forest_ovr_model(X_train, y_train, preprocessor):
    '''Perform feature selection using L1 regularization and train a one vs rest random forest classifier on the dataset.

    Parameters
    ----------
    X_train : pandas DataFrame
        The training dataset without the target column
    y_train : pandas Series
        The target column of the train set to predict
    preprocessor : sklearn ColumnTransformer
        Preprocessor to apply before fitting the classifier
    
    Returns
    -------
    sklearn Pipeline
        Pipeline containing the fitted classifier
    '''
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
    '''Trains a voting classifier ensemble of classifiers.

    Parameters
    ----------
    classifier : dict
        Dictionary with classifier name as key and classifiers as the dictionary value
    X_train : pandas DataFrame
        The training dataset without the target column
    y_train : pandas Series
        The target column of the train set to predict
    
    Returns
    -------
    sklearn VotingClassifier
    '''
    hard_voting_model = VotingClassifier(
        list(classifiers.items()), voting="hard"
    )
    hard_voting_model.fit(X_train, y_train)
    return hard_voting_model

def print_train_test_scores(classifiers, X_train, y_train, X_test, y_test):
    '''Print the train and test scores for each classifier.

    Parameters
    ----------
    classifiers : dict
        Dictionary with classifier name as key and classifiers as the dictionary value
    X_train : pandas DataFrame
        The training dataset without the target column
    y_train : pandas Series
        The target column of the train set to predict
    X_test : pandas DataFrame
        The testing dataset without the target column
    y_test : pandas Series
        The target column of the test set to predict
    
    Returns
    -------
    None
    '''
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
    save_train_test_df(train_df, test_df)
    X_train = train_df.drop(columns=["category"])
    y_train = train_df["category"]
    X_test = test_df.drop(columns=["category"])
    y_test = test_df["category"]
    preprocessor = define_preprocessor(X_train)
    rf_model = build_random_forest_model(X_train, y_train, preprocessor, class_weight)
    l1_reg_rf_model = build_l1_reg_random_forest_model(X_train, y_train, preprocessor, class_weight)
    l1_reg_rf_ovr_model = build_l1_reg_random_forest_ovr_model(X_train, y_train, preprocessor)
    classifiers = {
        "Random Forest": rf_model,
        "L1 Regularization Random Forest": l1_reg_rf_model,
        "L1 Regularization Random Forest OVR": l1_reg_rf_ovr_model
    }
    hard_voting_model = build_ensemble_model(classifiers, X_train, y_train)
    classifiers["Ensemble"] = hard_voting_model
    print_train_test_scores(classifiers, X_train, y_train, X_test, y_test)
    # Save the models into joblib files
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

if __name__ == "__main__":
    main()