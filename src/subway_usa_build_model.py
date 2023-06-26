''' 
This script primarily:
1. Creates and saves Fuzzy C-means models into Joblib file.
2. Labels stores and exports these labels to a CSV files.
3. Print clusters perentages
4. Random samples stores
'''

import pandas as pd
import numpy as np
import joblib
import os
import skfda

import sys
cwd = os.getcwd()
sys.path.append(cwd)
from subway_usa_cluster_verify import *

# Define dirctory 
DIR = 'data/Subway USA/'
processed = "../Subway_USA_Preprocessed/"

def read_data():
    '''Function to load train and test data

    Returns
    -------
    pandas.DataFrame
    '''
    train_df = pd.read_csv(DIR + processed + "subway_usa_processed_train.csv", index_col="store")
    test_df = pd.read_csv(DIR + processed + "subway_usa_processed_test.csv", index_col="store")
    stores = pd.read_csv(DIR + "subway_usa_stores.csv", index_col="store")
    return train_df, test_df, stores

def save_df(train_df, test_df):
    """Function to save the dataframe with the new 'labels' column to a csv files
    Parameters
    ----------
    train_df : pandas.DataFrame
        The training dataset
    test_df : pandas.DataFrame
        The training dataset
    
    Returns
    -------
    None
    """
    train_df.to_csv(DIR + processed + "train_df_with_labels.csv")
    test_df.to_csv(DIR + processed + "test_df_with_labels.csv")

def select_features(train_df, test_df):
    """Function to select useful features
    Parameters
    ----------
    train_df : pandas.DataFrame
        The training dataset
    test_df : pandas.DataFrame
        The training dataset
    
    Returns
    -------
    pandas.DataFrame
    """
    selected_features = ['age0018_p_ta',
    'age65pl_p_ta',
    'age85pl_p_ta',
    'banks_1mi',
    'banks_3mi',
    'centerxy_gla_effective_1mi',
    'centerxy_gla_effective_3mi',
    'com0002_p_ta',
    'com0508_p_ta',
    'com12pl_p_ta',
    'crime_total_index_ta',
    'daypop_dens_ta',
    'dmm_count_1mi',
    'dmm_count_3mi',
    'dmm_gla_1mi',
    'dmm_gla_3mi',
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
    'hh_dens_ta',
    'hh_expected_pers_ta',
    'hh_expected_vehicle_ta',
    'hh_type_1pers_p_ta',
    'hh_type_fam_p_ta',
    'hhcy_ta',
    'hrsa_number_of_certified_beds_1mi',
    'hrsa_number_of_certified_beds_3mi',
    'hu_ownerocc_ta',
    'hu_renterocc_ta',
    'hu_vacant_ta',
    'inrix_total_ta',
    'ipeds_postsecondary_schools_total_enrollment_1mi',
    'ipeds_postsecondary_schools_total_enrollment_3mi',
    'millenial_p_ta',
    'nces_private_schools_1mi',
    'nces_private_schools_3mi',
    'nces_private_schools_total_enrollment_1mi',
    'nces_private_schools_total_enrollment_3mi',
    'nces_public_schools_1mi',
    'nces_public_schools_3mi',
    'nces_public_schools_total_enrollment_1mi',
    'nces_public_schools_total_enrollment_3mi',
    'occ_bc_p_ta',
    'occ_wc_p_ta',
    'occhu_ta',
    'osm_highway_exits_count_1mi',
    'osm_highway_exits_count_3mi',
    'osm_nearest_exit_dist',
    'places_of_worship_1mi',
    'places_of_worship_3mi',
    'pop5y_cagr_ta',
    'pop_dens_ta',
    'pop_migration_ta',
    'pop_seasonal_ta',
    'pop_transient_ta',
    'popcy_ta',
    'popgr10cn_ta',
    'popgrfy_ta',
    'popgrpy_ta',
    'spend_foodbev_ta',
    'transitstop_nearest_dist',
    'transitstops',
    ]

    train_df = train_df[selected_features]
    test_df = test_df[selected_features]
    return train_df, test_df
    
def build_fcm_model(train_df, n_clusters=5, fuzzifier=1.1, max_iter=1000,random_state=42):
    """Function to fit fcm model
    Parameters
    ----------
    train_df : pandas.DataFrame
        The training dataset
    n_clusters=5 : num
        number of clusters
    fuzzifier=1.1 : num
        fuzziness of each cluster
    max_iter=1000 : num
        max iteration
    random_state=42 : num
        random state
    
    Returns
    -------
    FuzzyCMeans object
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
    Assign cluster labels to the training and test datasets based on a trained Fuzzy C-Means model.

    Parameters
    ----------
    train_df : pandas.DataFrame
        The training dataset.
    test_df : pandas.DataFrame
        The test dataset.
    fcm : skfuzzy.cluster.cmeans.CMeans
        The trained Fuzzy C-Means model.

    Returns
    -------
    train_df : pandas.DataFrame
        The training dataset with cluster labels.
    test_df : pandas.DataFrame
        The test dataset with cluster labels.

    Notes
    ------
    The function assigns the cluster labels to each observation in the training and test datasets using
    the labels and prediction method of the Fuzzy C-Means model. The labels are added as a new column in
    both datasets.
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

def print_cluster_percentages(n_clusters, test_df, fcm):
    """Funtion to print cluster_percentages
    Parameters
    ----------
    n_clusters : num
        number of clusters
    test_df : pandas.DataFrame
        The testing dataset
    fcm : FuzzyCMeans object
        The FuzzyCMeans model
    
    Returns
    -------
    None
    """
    # Convert the data to an FDataGrid object
    fdata = skfda.FDataGrid(test_df.values)
    # Get labels
    cluster_membership_train = fcm.labels_
    cluster_membership_test = fcm.predict(fdata)

    # Train
    cluster_percentages = []
    for cluster in range(n_clusters):
        percentage = np.sum(cluster_membership_train == cluster) / len(cluster_membership_train) * 100
        cluster_percentages.append(percentage)
    # Print the cluster percentages
    print("train data")
    for cluster, percentage in enumerate(cluster_percentages):
        print(f"Cluster {cluster} Percentage: {percentage:.2f}%")
    
    # Test
    cluster_percentages = []
    for cluster in range(n_clusters):
        percentage = np.sum(cluster_membership_test == cluster) / len(cluster_membership_test) * 100
        cluster_percentages.append(percentage)
    # Print the cluster percentages
    print("test data")
    for cluster, percentage in enumerate(cluster_percentages):
        print(f"Cluster {cluster} Percentage: {percentage:.2f}%")

def get_random_sample(n_clusters, train_df, test_df, stores, fcm, size=30):
    """Function to randomly generate samples with longitude and lattitude, along with store id
    Parameters
    ----------
    n_clusters : num
        number of clusters
    train_df : pandas.DataFrame
        The training dataset
    test_df : pandas.DataFrame
        The testing dataset
    store : pandas.DataFrame
        The stores information
    fcm : FuzzyCMeans object
        The FuzzyCMeans model
    size=20 : num
        number of sample size
    
    Returns
    -------
    dict
    """
    # Convert the data to an FDataGrid object
    fdata = skfda.FDataGrid(test_df.values)
    # Get labels
    cluster_membership_train = fcm.labels_
    cluster_membership_test = fcm.predict(fdata)
    
    # Train
    # Get sample stores
    cluster_samples_dict_train = {}
    # Select random samples from each cluster
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_membership_train == cluster)[0]
        if len(cluster_indices) >= size:
            np.random.seed(42)
            cluster_samples = np.random.choice(cluster_indices, size=size, replace=False)
        else:
            cluster_samples = cluster_indices
        
        # Store the samples in different cluster list
        cluster_samples_dict_train[cluster] = train_df.index[cluster_samples].tolist()
    
    # Get sample stores' longitude and lattitude
    cluster_coordinates_train = {}

    for cluster, samples in cluster_samples_dict_train.items():
        cluster_coordinates_train[cluster] = []
        for sample in samples:
            # Get the longitude and latitude, and store_id
            store_id = sample
            longitude = stores.loc[sample, "longitude"]
            latitude = stores.loc[sample, "latitude"]
            
            # Append the [longitude, latitude] pair to the corresponding cluster list
            cluster_coordinates_train[cluster].append([longitude, latitude, store_id])
    
    # Test
    # Get sample stores
    cluster_samples_dict_test = {}
    # Select random samples from each cluster
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_membership_test == cluster)[0]
        if len(cluster_indices) >= size:
            np.random.seed(42)
            cluster_samples = np.random.choice(cluster_indices, size=size, replace=False)
        else:
            cluster_samples = cluster_indices
        
        # Store the samples in different cluster list (train)
        cluster_samples_dict_test[cluster] = train_df.index[cluster_samples].tolist()
    
    # Get sample stores' longitude and lattitude
    cluster_coordinates_test = {}

    for cluster, samples in cluster_samples_dict_test.items():
        cluster_coordinates_test[cluster] = []
        for sample in samples:
            # Get the longitude and latitude, and store_id
            store_id = sample
            longitude = stores.loc[sample, "longitude"]
            latitude = stores.loc[sample, "latitude"]
            
            # Append the [longitude, latitude] pair to the corresponding cluster list
            cluster_coordinates_test[cluster].append([longitude, latitude, store_id])
    
    return cluster_coordinates_train, cluster_coordinates_test

def main():
    # Load data
    train_df, test_df, stores = read_data()
    
    # Select features
    train_df, test_df = select_features(train_df, test_df)
    
    # Create model
    n_clusters = 5
    fcm = build_fcm_model(train_df, 
                          n_clusters=n_clusters, 
                          fuzzifier = 1.1, 
                          max_iter = 1000,)
    
    # Print cluster percentages
    print_cluster_percentages(n_clusters, test_df, fcm)
    
    # Get samples
    cluster_coordinates_train, cluster_coordinates_test = get_random_sample(n_clusters, train_df, test_df, stores, fcm)
        
    #Verify cluster result for train split
    cluster_verify(cluster_coordinates_train)

    #Verify cluster result for test split
    # cluster_verify(cluster_coordinates_test)

    # Label data ploints
    train_df, test_df = add_labels(train_df, test_df, fcm)
    
    # Save the model
    try:
        joblib.dump(fcm, "model_joblib/subway_usa/fcm_model.joblib")
    except:
        os.makedirs("model_joblib/subway_usa/")
        joblib.dump(fcm, "model_joblib/subway_usa/fcm_model.joblib")
        
    # Save the labelled data
    save_df(train_df, test_df)

if __name__ == "__main__":
    main()