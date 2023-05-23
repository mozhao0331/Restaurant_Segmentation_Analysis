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

def read_data():
    poi = pd.read_csv(DIR + "processed_poi.csv")
    stores = pd.read_csv(DIR + "smoothie_king_stores.csv")
    trade_area = pd.read_csv(DIR + "processed_trade_area.csv")
    merged = stores.merge(trade_area, left_on="store", right_on="store_num").merge(poi)
    return merged

def process_data(df):
    drop_cols = ["store_num", "country_code", "store", "longitude", "latitude", "state_name", "cbsa_name", "dma_name"]
    df = df.drop(columns=drop_cols)
    le = LabelEncoder()
    df["category"] = le.fit_transform(df["category"])
    return df

def main():
    sk_df = read_data()
    

if __name__ == "__main__":
    main()