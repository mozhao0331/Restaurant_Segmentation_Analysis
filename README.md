# Restaurant_Segmentation_Analysis

<hr>

## Repo Structure

<pre>
📦Restaurant_Segmentation_Analysis
 ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/data">data</a>
 ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/data/Smoothie%20King">Smoothie King</a>: Contains raw data
 ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/data/Smoothie_King_Preprocessed">Smoothie_King_Preprocessed</a>: Contains processed data
 ┃ ┣ 📂Subway CAN
 ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/data/Subway%20USA">Subway USA</a>: Contains raw data
 ┃ ┗ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/data/Subway_USA_Preprocessed">Subway_USA_Preprocessed</a>: Contains processed data
 ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/doc">doc</a>
 ┃ ┣ 📜EDA.ipynb
 ┃ ┣ 📜Final_Presentation.pptx
 ┃ ┣ 📜Final_Report.Rmd
 ┃ ┣ 📜Final_Report.pdf
 ┃ ┣ 📜Proposal_Presentation.pptx
 ┃ ┣ 📜Proposal_Report.Rmd
 ┃ ┗ 📜Proposal_Report.pdf
 ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img">img</a>
 ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img/eda">eda</a>
 ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img/info">info</a>
 ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img/smoothie_king">smoothie_king</a>: Generated image for Smoothie King model feature interpretation
 ┃ ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img/smoothie_king/l1_reg_random_forest">l1_reg_random_forest</a>
 ┃ ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img/smoothie_king/l1_reg_random_forest_ovr">l1_reg_random_forest_ovr</a>
 ┃ ┃ ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img/smoothie_king/random_forest">random_forest</a>
 ┃ ┗ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/img/subway_usa">subway_usa</a>: Generated image for Subway US model feature interpretation
 ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/src">src</a>
 ┃ ┣ 📜helper_create_eda_figure.py
 ┃ ┣ 📜helper_evaluation.py
 ┃ ┣ 📜helper_plotting_functions.py
 ┃ ┣ 📜smoothie_king_build_model.py
 ┃ ┣ 📜smoothie_king_model_interpret.py
 ┃ ┣ 📜smoothie_king_preprocess_data.py
 ┃ ┣ 📜subway_usa_build_model.py
 ┃ ┣ 📜subway_usa_cluster_verify.html
 ┃ ┣ 📜subway_usa_cluster_verify.ipynb
 ┃ ┣ 📜subway_usa_cluster_verify.py
 ┃ ┣ 📜subway_usa_model_interpret.py
 ┃ ┗ 📜subway_usa_preprocess_data.py
 ┣ 📂<a href="https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/tree/main/test">test</a>
 ┃ ┣ 📜test_smoothie_king.py
 ┃ ┗ 📜test_subway_usa.py
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜Makefile
 ┣ 📜README.md
 ┣ 📜sitewise_python38_UBC2023.yaml
 ┗ 📜sitewise_python38_UBC2023_mac.yaml
</pre>

<hr>

## Contributors and Maintainers

-   Chen Lin
-   Eric Tsai
-   Morris Zhao
-   Xinru Lu

<hr>

## Motivation

The Restaurant Segmentation Analysis project is a collaboration between Sitewise Analytics and MDS students Chen Lin, Eric Tsai, Morris Zhao, and Xinru Lu. This project aims to use machine learning methods to determine factors that drive traffic to a particular location and identify clusters of similar store locations.

Restaurants seeking to open new stores in a region need to make marketing plans according to the major customer group. Therefore, restaurant franchise owners need to know the factors that drive traffic to a location, such as the surrounding population demographic and consumer behavior in the region, as well as trade area and nearby competitor/sister store information. By having a strong grasp of these factors, owners can plan future expansions and market the new location strategically based on the demand of the region. The Restaurant Segmentation Analysis project will address this problem by using data from Smoothie King locations in the United States and Subway locations in the United States to build machine learning data pipelines for Sitewise Analytics to incorporate into their consulting service. At the end of the project, we expect to have human-interpretable machine learning models that cluster similar store locations, which will be helpful for Sitewise Analytics clients to identify factors that drive traffic in those similar locations.

[Project Proposal (Sitewise)](https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/blob/main/doc/Proposal_Report.pdf)

[Project Final Report (Sitewise)](https://github.com/mozhao0331/Restaurant_Segmentation_Analysis/blob/main/doc/Final_Report.pdf)

<hr>

## Objectives

Given that Smoothie King and Subway US are all different clients of Sitewise, it is necessary to build three separate models for each respective client. Ultimately, the factors that drive traffic to each of the three restaurants may be different as well. Thus, the three machine learning pipelines are as follows:

1.  A supervised machine learning pipeline using data from Smoothie King locations to predict a store's category from one of five pre-labeled categories:

    -   Home
    -   Shopping
    -   Work
    -   Travel
    -   Other

The prediction will be human-interpretable in that users can identify features that determine the prediction of a store location's category.

2.  An unsupervised machine learning pipeline based on data of US Subway locations that cluster locations by similar features.

The unsupervised machine learning pipeline will also have human-interpretable results, including ways to identify similar features that caused different locations to be clustered together.

<hr>

## Data Summary

We received five datasets for each of the 2 popular chain restaurants: Smoothie King and Subway US. The datasets consist of CSV files for demographic, point of interest, store-specific data, competition sister store data, and trade area, where each row represents a single store location and the columns represent the variables/features of that store. All features in the demographic, point of interest, competitor/sister store, and trade area files are numeric, whereas the store-specific data files contain categorical features such as state and market size.

-   For Smoothie King, there are over 1000 features combined for 796 stores.
-   For Subway US, there are over 1000 features combined for approximately 14,000 stores.

<hr>

## Usage

To replicate the analysis, clone this GitHub repository along with installing the dependencies using the [environment file for Mac](/sitewise_python38_UBC2023_mac.yaml) and the [environment file for Windows](/sitewise_python38_UBC2023.yaml).

### Create Conda environment

```         
# For Mac
conda env create -n <ENVNAME> --file sitewise_python38_UBC2023_mac.yaml

# For Windows
conda env create -n <ENVNAME> --file sitewise_python38_UBC2023.yaml
```

### Activate Conda environment

```         
conda activate <ENVNAME>
```

### 1. Using Makefile to generate the proposal report

Run the following command at the command line/terminal in the project root directory:

```         
make all
```

To reset the project by cleaning the file path/directory, without any intermediate plot images or results .csv files, run the following command at the command line/terminal in the project root directory:

```         
make clean
```

### 2. Using Makefile to train the **Smoothie King** classification model

To train the Smoothie King classification model and get the interpretation outputs, run the following command:

```         
make smoothie_king
```

To reset the Smoothie King model outputs, run the following command:

```         
make clean_sk
```

### 3. Using Makefile to train the **Subway USA** clustering model

To train the Subway USA clustering model, get the interpretation outputs, run the following command:

```         
make subway_usa
```

To reset the Subway USA model outputs, run the following command:

```         
make clean_sb
```

**NOTE:**<br> To make the cluster verification script work, users need to install **Selenium** to interact with the Chrome browser.

```         
pip install selenium
```

Also, need to download **chromedriver** from [here](https://chromedriver.storage.googleapis.com/index.html). Ensure the driver version matches the Chrome browser version and save it under this path for Mac users.

```         
'/usr/local/bin/chromedriver'
```

<hr>

## Licenses

The Restaurant Segmentation Analysis project here is licensed under the MIT License. Please provide attribution and a link to this webpage if re-using/re-mixing any of these materials.
