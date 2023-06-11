import numpy as np
import pandas as pd
import os

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


DIR = "data/"
SUBWAYUS = "Subway USA/subway_usa_"


def is_climate_feat(feat):
    """
    Check whether a given feature is a climate-related feature.
    
    Parameters
    ----------
    feat : str
        Feature to be checked.

    Returns
    -------
    bool
        True if the feature is climate-related, False otherwise.
    """
    return "avgmax" in feat or "temp" in feat or feat == "precip" or feat == "snowfall"


def is_sport_venue(feat):
    """
    Check whether a given feature is related to sports venues.
    
    Parameters
    ----------
    feat : str
        Feature to be checked.

    Returns
    -------
    bool
        True if the feature is related to sports venues, False otherwise.
    """
    return "sports_venues" in feat


def agg_inrix(df):
    """
    Aggregate 'inrix_' prefixed columns in a DataFrame and remove the original 'inrix_' columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be processed.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with a new column 'inrix_total_ta' and without the original 'inrix_' columns.
    """
    inrix_columns = []
    for col in df.columns.tolist():
        if 'inrix_' in col:
            inrix_columns.append(col)
    df['inrix_total_ta'] = df[inrix_columns].sum(axis=1)
    df = df.drop(columns=inrix_columns)
    return df


def agg_veh(df, columns=None):
    """
    Aggregate specified vehicle-related columns in a DataFrame, removes the original columns and adds a new 
    column that is the sum of the products of the column values and their corresponding weights (from 1 to 5).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be processed.
    columns : list, optional
        List of column names to be aggregated. Defaults to a specific list of column names related to vehicle count.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with a new column 'hh_expected_vehicle_ta' and without the original columns related to vehicle count.
    """
    if not columns:
        columns = [
            'hh_0vehicle_p_ta', 'hh_1vehicle_p_ta', 'hh_2vehicle_p_ta',
            'hh_3vehicle_p_ta', 'hh_4vehicle_p_ta', 'hh_5vehicle_p_ta',
        ]
    df['hh_expected_vehicle_ta'] = df['hh_1vehicle_p_ta'] * 1 + df['hh_2vehicle_p_ta'] * 2 + df['hh_3vehicle_p_ta'] * 3 \
                                   + df['hh_4vehicle_p_ta'] * 4 + df['hh_5vehicle_p_ta'] * 5
    # print(f'----- Remove vehicle columns: {columns} -----')
    df = df.drop(columns=columns)
    return df


def agg_hh_pers(df, columns=None):
    """
    Aggregate specified household-person-related columns in a DataFrame, removes the original columns and adds a new 
    column that is the sum of the products of the column values and their corresponding weights (from 1 to 7).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be processed.
    columns : list, optional
        List of column names to be aggregated. Defaults to a specific list of column names related to household person count.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with a new column 'hh_expected_pers_ta' and without the original columns related to household person count.
    """
    if not columns:
        columns = [
            'hh_1pers_p_ta', 'hh_2pers_p_ta', 'hh_3pers_p_ta',
            'hh_4pers_p_ta', 'hh_5pers_p_ta', 'hh_6pers_p_ta',
            'hh_7pers_p_ta'
        ]
    df['hh_expected_pers_ta'] = df['hh_1pers_p_ta'] * 1 + df['hh_2pers_p_ta'] * 2 + df['hh_3pers_p_ta'] * 3 \
                                + df['hh_4pers_p_ta'] * 4 + df['hh_5pers_p_ta'] * 5 + df['hh_6pers_p_ta'] * 6\
                                + df['hh_7pers_p_ta'] * 7
    # print(f'----- Remove household person count columns: {columns} -----')
    df = df.drop(columns=columns)
    return df


def drop_specific_columns(df):
    """
    Process a DataFrame by removing specific columns based on certain conditions, and then applying a series of aggregation functions.

    This function excludes columns which contain certain substrings, are flagged by other helper functions, or match certain patterns. 
    The function also retains all other columns. It then applies the `agg_inrix`, `agg_veh`, and `agg_hh_pers` functions to 
    aggregate certain groups of columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be processed.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with certain columns removed and others aggregated into new columns.
    """
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
        if "_2mi" in col or "_4mi" in col or "_5mi" in col or "_10mi" in col:
            continue
        if "sports_venues" in col:  # remove sports venues columns (seems to be all zeros)
            continue
        if col.startswith('edu') and not col.startswith('edu_bachplus_p'):
            edu_cols.append(col)
            continue
        keep_columns.append(col)
    # print(f'----- Remove employment columns: {emp_cols} -----')
    # print(f'----- Remove education columns: {edu_cols} -----')
    # print(f'----- Removing {len(all_cols) - len(keep_columns)} columns -----')
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
    """
    Perform a transformation pipeline on train and test datasets, which includes dropping specified features, imputing missing values, 
    applying standard scaling on numeric features, encoding ordinal and categorical features, and handling unknown categories in 
    categorical features.

    Parameters
    ----------
    train : pandas.DataFrame
        The training dataset.
    test : pandas.DataFrame
        The test dataset.
    drop_features : list
        List of features to be dropped from the datasets.
    ordinal_features_oth : list
        List of ordinal features in the datasets.
    ordering_ordinal_oth : list
        List of ordering for the ordinal features.
    categorical_features : list
        List of categorical features in the datasets.
    numeric_features : list
        List of numerical features in the datasets.

    Returns
    -------
    tuple
        Transformed training and test datasets as pandas.DataFrame objects. The returned tuple contains two items: (transformed_train, transformed_test).
    """
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
    """
    Merge datasets related to stores, points of interest (POI), and trade areas, optionally including demographic data. 
    The merged dataset is then split into training and testing sets.

    Parameters
    ----------
    stores : pandas.DataFrame
        DataFrame containing store data. Each row represents a store, and 'store' is expected as a column and used as the merge key.
    poi : pandas.DataFrame
        DataFrame containing points of interest data. 'store' is expected as a column and used as the merge key.
    trade_area : pandas.DataFrame
        DataFrame containing trade area data. 'store' is expected as a column and used as the merge key.
    demographic : pandas.DataFrame, optional
        DataFrame containing demographic data. If provided, 'store' is expected as a column and used as the merge key.

    Returns
    -------
    tuple
        A tuple of two pandas.DataFrame objects: (train_df, test_df), representing the training and testing datasets respectively.
    """
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
    """
    Load, merge, process, and split the Subway USA data. This includes loading the points of interest (poi), trade area, 
    and store data, optionally loading demographic data, merging these datasets, dropping specific features, transforming 
    the data, and saving the resulting processed training and test datasets as CSV files.

    Parameters
    ----------
    include_demographic : bool, optional
        If True, includes demographic data in the processing. Default is False.

    Returns
    -------
    None
    """
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
    processed_train.to_csv(DIR + "Subway_USA_Preprocessed/" + "subway_usa_processed_train.csv")
    processed_test.to_csv(DIR + "Subway_USA_Preprocessed/" + "subway_usa_processed_test.csv")
    

def main():
    print('============= Starting to preprocess Subway USA =============')
    try:
        process_subway_usa(include_demographic=False)
    except:
        os.makedirs(DIR + "Subway_USA_Preprocessed/")
        process_subway_usa(include_demographic=False)
    
if __name__ == "__main__":
    main()
