import os
import pandas as pd
import numpy as np
from src.preprocess_smoothie_king_data import *


def test_process_percent_non_percent_df():
    in_file = "data/Smoothie King/smoothie_king_trade_area_variables.csv"
    trade_area_df = pd.read_csv(in_file)
    out_file = "data/Smoothie King/processed_trade_area.csv"
    process_percent_non_percent_df(trade_area_df, out_file)
    assert os.path.isfile(out_file)

def test_process_poi_df():
    in_file = "data/Smoothie King/smoothie_king_poi_variables.csv"
    poi_df = pd.read_csv(in_file)
    out_file = "data/Smoothie King/processed_poi.csv"
    process_poi_df(poi_df, out_file)
    assert os.path.isfile(out_file)


