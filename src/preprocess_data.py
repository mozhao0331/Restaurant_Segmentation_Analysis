import pandas as pd

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

def main():
    demographic = pd.read_csv(DIR + "Smoothie King/smoothie_king_demographic_variables.csv")
    trade_area = pd.read_csv(DIR + "Smoothie King/smoothie_king_trade_area_variables.csv")
    process_percent_non_percent_df(demographic, "Smoothie King/processed_demographic.csv")
    process_percent_non_percent_df(trade_area, "Smoothie King/processed_trade_area.csv")

if __name__ == "__main__":
    main()