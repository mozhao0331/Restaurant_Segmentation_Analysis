import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier


DIR = 'data/Smoothie King/'
ALL_FEATURES = None


def read_data():
    poi = pd.read_csv(DIR + "processed_poi.csv")
    stores = pd.read_csv(DIR + "smoothie_king_stores.csv")
    trade_area = pd.read_csv(DIR + "processed_trade_area.csv")
    merged = stores.merge(trade_area, left_on="store", right_on="store_num").merge(poi)
    return merged


def drop_unused_cols(df):
    drop_cols = ["store_num", "country_code", "store", "longitude", "latitude", "state_name", "cbsa_name", "dma_name"]
    df = df.drop(columns=drop_cols)
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
    ALL_FEATURES = numeric_features + ordinal_features

    ordinal_transformer = OrdinalEncoder(categories=[market_levels, density_levels], dtype=int)
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (ordinal_transformer, ordinal_features),
    )
    return preprocessor


def main():
    sk_df = read_data()
    sk_df = drop_unused_cols(sk_df)
    le = LabelEncoder()
    sk_df["category"] = le.fit_transform(sk_df["category"])
    train_df, test_df = train_test_split(sk_df, test_size=0.1, random_state=42)
    X_train = train_df.drop(columns=["category"])
    y_train = train_df["category"]
    X_test = test_df.drop(columns=["category"])
    y_test = test_df["category"]

    class_weight = {
        "HOME": 0.24,
        "OTHER": 0.16,
        "SHOPPING": 0.22,
        "TRAVEL": 0.15,
        "WORK": 0.23
    }
    class_weight = {i: class_weight[label] for i, label in enumerate(le.classes_)}
    preprocessor = define_preprocessor(X_train)


if __name__ == "__main__":
    main()