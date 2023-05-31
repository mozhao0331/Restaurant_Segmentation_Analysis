''' 
This script primarily:
1. Creates and saves Fuzzy C-means models into Joblib file.
2. Labels stores and exports these labels to a CSV files.
'''

import pandas as pd
import joblib
import os
import skfda

# Define dirctory
DIR = 'data/Subway USA/'

def read_data():
    """
    Function to load train and test data
    """
    train_df = pd.read_csv(DIR + "subway_usa_processed_train.csv", index_col="store")
    test_df = pd.read_csv(DIR + "subway_usa_processed_test.csv", index_col="store")
    return train_df, test_df

def save_df(train_df, test_df):
    """
    Function to save the dataframe with the new 'labels' column to a csv files
    """
    train_df.to_csv(DIR + "train_df_with_labels.csv")
    test_df.to_csv(DIR + "test_df_with_labels.csv")

def select_features(train_df, test_df):
    """
    Function to select useful features
    """
    selected_features = ['age0018_p_ta',
    'age65pl_p_ta',
    'age85pl_p_ta',
    #  'asian_p_ta',
    #  'avg_faminc_ta',
    #  'avghhinc_ta',
    'banks_1mi',
    'banks_5mi',
    #  'black_p_ta',
    #  'boomer_p_ta',
    'centerxy_gla_effective_1mi',
    'centerxy_gla_effective_5mi',
    'com0002_p_ta',
    #  'com0205_p_ta',
    'com0508_p_ta',
    #  'com0811_p_ta',
    'com12pl_p_ta',
    'crime_total_index_ta',
    'daypop_dens_ta',
    'disposable_inc_avg_ta',
    'dmm_count_1mi',
    'dmm_count_5mi',
    'dmm_gla_1mi',
    'dmm_gla_5mi',
    #  'dmm_nearest_dist',
    'dtpop_children_at_home_p_ta',
    'dtpop_homemakers_p_ta',
    'dtpop_retired_disabled_p_ta',
    'dtpop_students_9th_12th_p_ta',
    'dtpop_students_p_ta',
    'dtpop_students_post_secondary_p_ta',
    'dtpop_students_prek_8th_p_ta',
    'dtpop_ta',
    'dtpop_unemployed_p_ta',
    'dtpop_work_at_home_p_ta',
    'edu_bachplus_p_ta',
    'emp_p_ta',
    'empcy_ta',
    'gdp_ta',
    #  'genx_p_ta',
    #  'genz_p_ta',
    #  'gq_college_p_ta',
    #  'gq_other_p_ta',
    'hh_dens_ta',
    'hh_expected_pers_ta',
    'hh_expected_vehicle_ta',
    #  'hh_inc_gt_500k_p_ta',
    #  'hh_inc_gt_75k_p_ta',
    #  'hh_inc_lt_75k_p_ta',
    'hh_type_1pers_p_ta',
    'hh_type_fam_p_ta',
    #  'hh_type_female_child_p_ta',
    #  'hh_type_female_nochild_p_ta',
    #  'hh_type_female_p_ta',
    #  'hh_type_male_child_p_ta',
    #  'hh_type_male_nochild_p_ta',
    #  'hh_type_male_p_ta',
    #  'hh_type_married_child_p_ta',
    #  'hh_type_married_nochild_p_ta',
    #  'hh_type_married_p_ta',
    #  'hh_type_nonfam_p_ta',
    'hhcy_ta',
    #  'hhgrfypy_ta',
    #  'hhgrpycy_ta',
    'hhinc100pl_p_ta',
    'hhinc150pl_p_ta',
    'hhinc30lt_p_ta',
    #  'hispanic_p_ta',
    #  'hrsa_hospitals_1mi',
    #  'hrsa_hospitals_5mi',
    #  'hrsa_hospitals_nearest_dist',
    'hrsa_number_of_certified_beds_1mi',
    'hrsa_number_of_certified_beds_5mi',
    'hu_ownerocc_ta',
    'hu_renterocc_ta',
    'hu_vacant_ta',
    'inrix_total_ta',
    #  'ipeds_postsecondary_nearest_dist',
    #  'ipeds_postsecondary_schools_1mi',
    #  'ipeds_postsecondary_schools_5mi',
    'ipeds_postsecondary_schools_total_enrollment_1mi',
    'ipeds_postsecondary_schools_total_enrollment_5mi',
    'market_size',
    'medhhinc_ta',
    'medsalcy_ta',
    'millenial_p_ta',
    #  'mortgage_avgrisk_ta',
    'nces_private_schools_1mi',
    'nces_private_schools_5mi',
    #  'nces_private_schools_nearest_dist',
    'nces_private_schools_total_enrollment_1mi',
    'nces_private_schools_total_enrollment_5mi',
    'nces_public_schools_1mi',
    'nces_public_schools_5mi',
    #  'nces_public_schools_nearest_dist',
    'nces_public_schools_total_enrollment_1mi',
    'nces_public_schools_total_enrollment_5mi',
    'occ_bc_p_ta',
    #  'occ_unclassified_p_ta',
    'occ_wc_p_ta',
    'occhu_ta',
    'osm_highway_exits_count_1mi',
    'osm_highway_exits_count_5mi',
    'osm_nearest_exit_dist',
    #  'other_p_ta',
    'percapita_inc_ta',
    'places_of_worship_1mi',
    'places_of_worship_5mi',
    'pop5y_cagr_ta',
    'pop_dens_ta',
    'pop_migration_ta',
    'pop_seasonal_ta',
    'pop_transient_ta',
    'popcy_ta',
    'popgr10cn_ta',
    'popgrfy_ta',
    'popgrpy_ta',
    'poverty_inpoverty_p_ta',
    #  'spend_breakfastbrunch_ta',
    #  'spend_dinner_ta',
    'spend_foodbev_ta',
    #  'spend_lunch_ta',
    'store_density',
    'transitstop_nearest_dist',
    'transitstops',
    'wealth_hhavg_ta',
    #  'wealth_hhtotal_ta',
    #  'white_p_ta'
    ]

    train_df = train_df[selected_features]
    test_df = test_df[selected_features]
    return train_df, test_df
    
def build_fcm_model(train_df, n_clusters=5, fuzzifier = 1.1, max_iter = 1000,random_state = 42):
    """
    Function to fit fcm model
    """
    # Convert the data to an FDataGrid object
    fdata = skfda.FDataGrid(train_df.values)

    # Create the FuzzyCMeans object
    fcm = skfda.ml.clustering.FuzzyCMeans(n_clusters=n_clusters, 
                    fuzzifier=fuzzifier, 
                    max_iter=max_iter, 
                    random_state=random_state)

    # Fit the model
    fcm.fit(fdata)
    return fcm

def add_labels(train_df, test_df, fcm):
    """
    Function to get the cluster labels
    """
    # Convert the data to an FDataGrid object
    fdata = skfda.FDataGrid(test_df.values)
    # Get labels
    cluster_membership_train = fcm.labels_
    cluster_membership_test = fcm.predict(fdata)

    # Add labels to the original data
    train_df['labels'] = cluster_membership_train
    test_df['labels'] = cluster_membership_test
    return train_df, test_df

def main():
    # Load data
    train_df, test_df = read_data()
    
    # Select features
    train_df, test_df = select_features(train_df, test_df)
    
    # Create model
    fcm = build_fcm_model(train_df, 
                          n_clusters=5, 
                          fuzzifier = 1.1, 
                          max_iter = 1000,)
    
    # Label data ploints
    train_df, test_df = add_labels(train_df, test_df, fcm)
    
    # Save the model
    try:
        joblib.dump(fcm, "sub_usa_model_joblib/fcm_model.joblib")
    except:
        os.mkdir("sub_usa_model_joblib/")
        joblib.dump(fcm, "sub_usa_model_joblib/fcm_model.joblib")
        
    # Save the labelled data
    save_df(train_df, test_df)

if __name__ == "__main__":
    main()