import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


DIR = "data/"
SUBWAYUS = "Subway USA/subway_usa_"


def is_climate_feat(feat):
    return "avgmax" in feat or "temp" in feat or feat == "precip" or feat == "snowfall"


def is_sport_venue(feat):
    return "sports_venues" in feat


def drop_specific_columns(df):
    all_cols = df.columns.tolist()
    keep_columns = []
    
    # drop features with corresponding percentage measures
    percent_feats = [col for col in all_cols if "_p_" in col]
    remove_feats = ["_".join(feat.split("_p_")) for feat in percent_feats]

    for col in all_cols:
        if col in remove_feats:
            continue
        if "centerxy" in col:
            if "full" not in col and "effective" not in col:
                continue
        if is_climate_feat(col):
            continue
        # remove sports venues columns (seems to be all zeros)
        if "sports_venues" in col:
            continue
        if col.startswith('edu') and not col.startswith('edu_bachplus_p'):
            continue
        keep_columns.append(col)
    print(f'----- Removing {len(all_cols) - len(keep_columns)} columns -----')
    reduced_df = df[keep_columns]
    return reduced_df


def data_transform_pipeline(
    train, 
    test,
    drop_features, 
    ordinal_features_oth, 
    ordering_ordinal_oth, 
    categorical_features, 
    numeric_features
):
    train_index = train['store']
    test_index = test['store']
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), 
        StandardScaler()
    )

    ordinal_transformer_oth = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(categories=ordering_ordinal_oth),
    )
    if len(categorical_features) > 0:
        categorical_transformer = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            OneHotEncoder(handle_unknown="ignore", sparse=False),
        )

        preprocessor = make_column_transformer(
            ("drop", drop_features),
            (numeric_transformer, numeric_features),
            (ordinal_transformer_oth, ordinal_features_oth),
            (categorical_transformer, categorical_features),
        )
        column_names = (
            numeric_features
            + ordinal_features_oth
            + preprocessor.named_transformers_['pipeline-3']['onehotencoder'].get_feature_names_out().tolist()
        )
    else:
        preprocessor = make_column_transformer(
            ("drop", drop_features),
            (numeric_transformer, numeric_features),
            (ordinal_transformer_oth, ordinal_features_oth),
        )
        column_names = (
            numeric_features
            + ordinal_features_oth
        )
    preprocessor.fit(train)
    transformed_train = pd.DataFrame(preprocessor.transform(train), columns=column_names, index=train_index)
    transformed_test = pd.DataFrame(preprocessor.transform(test), columns=column_names, index=test_index)
    return transformed_train, transformed_test

def merge_split_data(stores, poi, trade_area, demographic):
    merged = stores.merge(
        poi, on="store"
    ).merge(
        trade_area, on="store"
    )
    merged = drop_specific_columns(merged)
    train_df, test_df = train_test_split(merged, test_size=0.1, random_state=42)
    return train_df, test_df


def main():
    subway_usa_demographic = pd.read_csv(DIR + SUBWAYUS + "demographic_variables.csv")
    subway_usa_poi = pd.read_csv(DIR + SUBWAYUS + "poi_variables.csv")
    subway_usa_trade_area = pd.read_csv(DIR + SUBWAYUS + "trade_area_variables.csv")
    subway_usa_stores = pd.read_csv(DIR + "Subway USA/subway_usa_stores.csv", encoding='latin-1')

    train_df, test_df = merge_split_data(
        stores=subway_usa_stores, 
        poi=subway_usa_poi,
        trade_area=subway_usa_trade_area,
        demographic=subway_usa_demographic
    )

    drop_features = [
        "store", 
        "longitude", 
        "latitude", 
    ]

    ordinal_features_oth = [
        "market_size",
        "store_density",
    ]
    ordering_ordinal_oth = [
        ["Very Large Metro (1)", "Large Metro (2)", "Large City (3)", "Medium City (4)", "Small City (5)", "Small Town (6)", "Small Community (7)"],
        ["Rural", "Exurban", "Suburban", "Light Suburban", "Light Urban", "Urban", "Super Urban"],
    ]
    categorical_features = ["cbsa_name", "dma_name", "censusdivision", "censusregion"]

    numeric_features = list(set(train_df.select_dtypes(include=np.number).columns.tolist()) - set(drop_features))
    processed_train, processed_test = data_transform_pipeline(
        train_df, 
        test_df, 
        drop_features + categorical_features, 
        ordinal_features_oth, 
        ordering_ordinal_oth, 
        [], 
        numeric_features
    )
    processed_train.to_csv(DIR + SUBWAYUS + "processed_train.csv")
    processed_test.to_csv(DIR + SUBWAYUS + "processed_test.csv")
    
    # subway_can_demographic = pd.read_csv(DIR + "Subway CAN/subway_can_demographic_variables.csv")
    # subway_can_poi = pd.read_csv(DIR + "Subway CAN/subway_can_poi_variables.csv")
    # subway_can_trade_area = pd.read_csv(DIR + "Subway CAN/subway_can_trade_area_variables.csv")
    # process_percent_non_percent_df(subway_can_demographic, "Subway CAN/processed_demographic.csv")
    # process_percent_non_percent_df(subway_can_trade_area, "Subway CAN/processed_trade_area.csv")
    # process_store_df(subway_can_poi, "Subway CAN/processed_poi.csv")

if __name__ == "__main__":
    main()
