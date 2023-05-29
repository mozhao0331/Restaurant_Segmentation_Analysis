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


def agg_inrix(df):
    inrix_columns = []
    for col in df.columns.tolist():
        if 'inrix_' in col:
            inrix_columns.append(col)
    df['inrix_total_ta'] = df[inrix_columns].sum(axis=1)
    df = df.drop(columns=inrix_columns)
    return df


def agg_veh(df, columns=None):
    if not columns:
        columns = [
            'hh_0vehicle_p_ta', 'hh_1vehicle_p_ta', 'hh_2vehicle_p_ta',
            'hh_3vehicle_p_ta', 'hh_4vehicle_p_ta', 'hh_5vehicle_p_ta',
        ]
    df['hh_expected_vehicle_ta'] = df['hh_1vehicle_p_ta'] * 1 + df['hh_2vehicle_p_ta'] * 2 + df['hh_3vehicle_p_ta'] * 3 \
                                   + df['hh_4vehicle_p_ta'] * 4 + df['hh_5vehicle_p_ta'] * 5
    print(f'----- Remove vehicle columns: {columns} -----')
    df = df.drop(columns=columns)
    return df


def agg_hh_pers(df, columns=None):
    if not columns:
        columns = [
            'hh_1pers_p_ta', 'hh_2pers_p_ta', 'hh_3pers_p_ta',
            'hh_4pers_p_ta', 'hh_5pers_p_ta', 'hh_6pers_p_ta',
            'hh_7pers_p_ta'
        ]
    df['hh_expected_pers_ta'] = df['hh_1pers_p_ta'] * 1 + df['hh_2pers_p_ta'] * 2 + df['hh_3pers_p_ta'] * 3 \
                                + df['hh_4pers_p_ta'] * 4 + df['hh_5pers_p_ta'] * 5 + df['hh_6pers_p_ta'] * 6\
                                + df['hh_7pers_p_ta'] * 7
    print(f'----- Remove household person count columns: {columns} -----')
    df = df.drop(columns=columns)
    return df


def drop_specific_columns(df):
    all_cols = df.columns.tolist()
    keep_columns = []
    
    # drop features with corresponding percentage measures
    percent_feats = [col for col in all_cols if "_p_" in col]
    remove_feats = ["_".join(feat.split("_p_")) for feat in percent_feats]
    emp_cols = []
    edu_cols = []
    
    for col in all_cols:
        if col in remove_feats:
            continue
        if "centerxy" in col:
            if "full" not in col and "effective" not in col:
                continue
        if is_climate_feat(col):
            continue
        if "emp_" in col and col != "emp_p_ta":  # remove employment specific column
            emp_cols.append(col)
            continue
        if "military" in col:  # remove military related columns
            continue
        if "_2mi" in col or "_3mi" in col or "_4mi" in col or "_10mi" in col:
            continue
        if "sports_venues" in col:  # remove sports venues columns (seems to be all zeros)
            continue
        if col.startswith('edu') and not col.startswith('edu_bachplus_p'):
            edu_cols.append(col)
            continue
        keep_columns.append(col)
    print(f'----- Remove employment columns: {emp_cols} -----')
    print(f'----- Remove education columns: {edu_cols} -----')
    print(f'----- Removing {len(all_cols) - len(keep_columns)} columns -----')
    reduced_df = df[keep_columns]
    
    # aggregate intrix columns
    reduced_df = agg_inrix(reduced_df)
    # aggregate vehicle columns
    reduced_df = agg_veh(reduced_df)
    # aggregate household headcount columns
    reduced_df = agg_hh_pers(reduced_df)
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


def merge_split_data(stores, poi, trade_area, demographic=None):
    if demographic:
        merged = stores.merge(
            poi, on="store"
        ).merge(
            trade_area, on="store"
        ).merge(
            demographic, on="store"
        )
    else:
        merged = stores.merge(
            poi, on="store"
        ).merge(
            trade_area, on="store"
        )
    merged = drop_specific_columns(merged)
    train_df, test_df = train_test_split(merged, test_size=0.1, random_state=42)
    return train_df, test_df


def process_subway_usa(include_demographic=False):
    subway_usa_poi = pd.read_csv(DIR + SUBWAYUS + "poi_variables.csv")
    subway_usa_trade_area = pd.read_csv(DIR + SUBWAYUS + "trade_area_variables.csv")
    subway_usa_stores = pd.read_csv(DIR + "Subway USA/subway_usa_stores.csv", encoding='latin-1')
    if include_demographic:
        subway_usa_demographic = pd.read_csv(DIR + SUBWAYUS + "demographic_variables.csv")
        train_df, test_df = merge_split_data(
            stores=subway_usa_stores,
            poi=subway_usa_poi,
            trade_area=subway_usa_trade_area,
            demographic=subway_usa_demographic
        )
    else:
        train_df, test_df = merge_split_data(
            stores=subway_usa_stores,
            poi=subway_usa_poi,
            trade_area=subway_usa_trade_area,
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
        ["Very Large Metro (1)", "Large Metro (2)", "Large City (3)", "Medium City (4)", "Small City (5)",
         "Small Town (6)", "Small Community (7)"],
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
    

def main():
    print('============= Starting to preprocess Subway USA =============')
    process_subway_usa(include_demographic=False)


if __name__ == "__main__":
    main()
