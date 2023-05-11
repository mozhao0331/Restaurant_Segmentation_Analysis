# Restaurant_Segmentation_Analysis

## Contributors and Maintainers

- Chen Lin
- Eric Tsai
- Morris Zhao
- Xinru Lu

## Motivation

The Restaurant Segmentation Analysis project is a collaboration between Sitewise Analytics and MDS students Chen Lin, Eric Tsai, Morris Zhao, and Xinru Lu. This project aims to use machine learning methods to determine factors that drive traffic to a particular location and  identify clusters of similar store locations. 

Restaurants seeking to open new stores in a region need to make marketing plans according to the major customer group. Therefore, restaurant franchise owners need to know the factors that drive traffic to a location, such as the surrounding population demographic and consumer behaviour in the region, as well as trade area and nearby competitor/sister store information. By having a strong grasp of these factors, owners can plan future expansions and market the new location strategically based on the demand of the region. The Restaurant Segmentation Analysis project will address this problem by using data from Smoothie King locations in the United States and Subway locations in Canada and the United States to build machine learning data pipelines for Sitewise Analytics to incorporate into their consulting service. At the end of the project, we expect to have human-interpretable machine learning models that cluster similar store locations, which will be helpful for Sitewise Analytics clients to identify factors that drive traffic in those similar locations.

[Project Proposal (Sitewise)](https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/blob/main/doc/Proposal_Report.pdf)

## Objectives

Given that Smoothie King, Subway US, and Subway Canada are all different clients of Sitewise, it is necessary to build three separate models for each respective client. Ultimately, the factors that drive traffic to each of the three restaurants may be different as well. Thus, the three machine learning pipelines are as follows:

1.  A supervised machine learning pipeline using data from Smoothie King locations to predict a store's category from one of five pre-labeled categories:

    -   Home
    -   Shopping
    -   Work
    -   Travel
    -   Other

The prediction will be human-interpretable in that users can identify features that determine the prediction of a store location's category.

2.  An unsupervised machine learning pipeline based on data of US Subway locations that cluster locations by similar features.

3.  An unsupervised machine learning pipeline based on data of Canadian Subway locations that cluster locations by similar features.

The two unsupervised machine learning pipelines will also have human-interpretable results, including ways to identify similar features that caused different locations to be clustered together.

## Data Summary

We received five datasets for each of the three popular chain restaurants: Smoothie King, Subway Canada, and Subway US. The datasets consist of CSV files for demographic, point of interest, store-specific data, competition sister store data, and trade area, where each row represents a single store location and the columns represent the variables/features of that store. All features in the demographic, point of interest, competition sister store, and trade area files are numeric, whereas the store-specific data files contain categorical features such as state and market size.

- For Smoothie King, there are over 1000 features combined for 796 stores.
- For Subway US, there are over 1000 features combined for approximately 14,000 stores.
- For Subway Canada, there are around 100 features combined for around 1,800 stores.

## Usage
To replicate the analysis, first to clone this GitHub repository along with installing the dependencies using the [environment file for Mac](/environment_mac.yml) and [environment file for Windows](/sitewise_python38_UBC2023.yaml).

### Create Conda environment

```
# For Mac
conda env create -n <ENVNAME> --file environment_mac.yaml

# For Windows
conda env create -n <ENVNAME> --file sitewise_python38_UBC2023.yaml
```

### Activate Conda environment

```
conda activate <ENVNAME>
```


### Using Makefile to generate the proposal report

Run the following command at the command line/terminal in the project root directory:

```
make all
```

To reset the project with cleaning file path/directory, without any intermeidate plot images or results csv files, run the following command at the command line/terminal in the project root directory:

```
make clean
```
## Licenses

The Restaurant Segmentation Analysis project here is licensed under the MIT License.  Please provide attribution and a link to this webpage if re-using/re-mixing any of these materials.
