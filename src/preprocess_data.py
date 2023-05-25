import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


DIR = "data/"


def process_percent_non_percent_df(df, out_dir):
    all_cols = df.columns.tolist()
    percent_feats = [col for col in all_cols if "_p_" in col]
    non_percent_feats = list(set(all_cols) - set(percent_feats))
    overlapping_feats = []
    for feat in percent_feats:
        remove_percent_feat = "_".join(feat.split("_p_"))
        if remove_percent_feat in non_percent_feats:
            overlapping_feats.append(remove_percent_feat)
    no_overlap = list(set(non_percent_feats) - set(overlapping_feats))
    reduced_feats = percent_feats
    reduced_feats.extend(no_overlap)
    reduced_feats.sort()
    reduced_df = df[reduced_feats]
    reduced_df.to_csv(DIR + out_dir, index=False)


def process_store_df(df, out_dir):
    all_cols = df.columns.tolist()
    cols_to_remove = []
    for col in all_cols:
        if "centerxy" in col:
            if "full" not in col and "effective" not in col:
                cols_to_remove.append(col)
        elif is_climate_feat(col):
            cols_to_remove.append(col)
        elif is_sport_venue(col):
            cols_to_remove.append(col)
    reduced_feats = list(set(all_cols) - set(cols_to_remove))
    reduced_feats.sort()
    reduced_df = df[reduced_feats]
    reduced_df.to_csv(DIR + out_dir, index=False)


def is_climate_feat(feat):
    return "avgmax" in feat or "temp" in feat or feat == "precip" or feat == "snowfall"


def is_sport_venue(feat):
    return "sports_venues" in feat


def data_transform_pipeline(
    train, 
    test, 
    train_index,
    test_index,
    drop_features, 
    ordinal_features_oth, 
    ordering_ordinal_oth, 
    categorical_features, 
    numeric_features
):
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


def main():
    smoothie_king_demographic = pd.read_csv(DIR + "Smoothie King/smoothie_king_demographic_variables.csv")
    smoothie_king_trade_area = pd.read_csv(DIR + "Smoothie King/smoothie_king_trade_area_variables.csv")
    smoothie_king_poi = pd.read_csv(DIR + "Smoothie King/smoothie_king_poi_variables.csv")
    process_percent_non_percent_df(smoothie_king_demographic, "Smoothie King/processed_demographic.csv")
    process_percent_non_percent_df(smoothie_king_trade_area, "Smoothie King/processed_trade_area.csv")
    process_store_df(smoothie_king_poi, "Smoothie King/processed_poi.csv")
    
    # subway_usa_demographic = pd.read_csv(DIR + "Subway USA/subway_usa_demographic_variables.csv")
    # subway_usa_poi = pd.read_csv(DIR + "Subway USA/subway_usa_poi_variables.csv")
    # subway_usa_trade_area = pd.read_csv(DIR + "Subway USA/subway_usa_trade_area_variables.csv")
    # process_percent_non_percent_df(subway_usa_demographic, "Subway USA/processed_demographic.csv")
    # process_percent_non_percent_df(subway_usa_trade_area, "Subway USA/processed_trade_area.csv")
    # process_store_df(subway_usa_poi, "Subway USA/processed_poi.csv")
    
    # subway_can_demographic = pd.read_csv(DIR + "Subway CAN/subway_can_demographic_variables.csv")
    # subway_can_poi = pd.read_csv(DIR + "Subway CAN/subway_can_poi_variables.csv")
    # subway_can_trade_area = pd.read_csv(DIR + "Subway CAN/subway_can_trade_area_variables.csv")
    # process_percent_non_percent_df(subway_can_demographic, "Subway CAN/processed_demographic.csv")
    # process_percent_non_percent_df(subway_can_trade_area, "Subway CAN/processed_trade_area.csv")
    # process_store_df(subway_can_poi, "Subway CAN/processed_poi.csv")


if __name__ == "__main__":
    main()
