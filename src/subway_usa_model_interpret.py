import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sys
cwd = os.getcwd() 
sys.path.append(cwd)
from helper_plotting_functions import plot_boxplot


DIR = "data/Subway_USA_Preprocessed/"
FIG_DIR = "img/subway_usa/"

def read_data():
    """
    Read train and test data from CSV files.

    Returns
    -------
    tuple
        Returns a tuple with two pandas DataFrames (train_df, test_df).
    """
    train_df = pd.read_csv(DIR + "train_df_with_labels.csv", index_col="store")
    test_df = pd.read_csv(DIR + "test_df_with_labels.csv", index_col="store")
    return train_df, test_df

def fit_random_forest_classifier(X_train, y_train):
    """
    Fit a Random Forest Classifier model.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The input training data.
    y_train : pandas.Series
        The target output for the training data.

    Returns
    -------
    RandomForestClassifier
        The fitted Random Forest Classifier model.
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def save_figure(out_dir, file_name):
    """
    Save a figure to a specified directory. 

    Parameters
    ----------
    out_dir : str
        The output directory to save the figure.
    file_name : str
        The file name for the saved figure.

    Returns
    -------
    None
    """
    try:
        plt.savefig(out_dir + file_name, bbox_inches="tight")
    except:
        os.makedirs(out_dir)
        plt.savefig(out_dir + file_name, bbox_inches="tight")
        
def plot_shap_feature_importance(shap_values, X_enc, target_class):
    """
    Plot SHAP feature importance using a beeswarm plot.

    Parameters
    ----------
    shap_values : array-like
        The SHAP values to plot.
    X_enc : pandas.DataFrame
        The encoded input data used to generate the SHAP values.
    target_class : int
        The target class for which to plot feature importance.

    Returns
    -------
    None
    """
    shap.summary_plot(shap_values, X_enc, show=False)
    plt.title(f"SHAP Feature Importance for cluster {target_class}")
    save_figure(FIG_DIR, f"cluster_{target_class}_summary_plot.png")
    plt.close()

def plot_shap_force_plot(explainer, shap_values, X_enc, target_class, idx_to_explain):
    """
    Plot SHAP force plot for the most confident prediction.

    Parameters
    ----------
    explainer : shap.Explainer
        The SHAP explainer object.
    shap_values : array-like
        The SHAP values to plot.
    X_enc : pandas.DataFrame
        The encoded input data used to generate the SHAP values.
    target_class : int
        The target class for which to plot feature importance.
    idx_to_explain : int
        Index of the instance in the dataset to explain.

    Returns
    -------
    None
    """
    shap.force_plot(
        explainer.expected_value[target_class], 
        shap_values[idx_to_explain, :], 
        X_enc.iloc[idx_to_explain, :], 
        matplotlib=True,
        show=False,
        text_rotation=20
    )
    plt.title(f"SHAP Force Plot for Most Confident Prediction of Class {target_class}", y=1.75)
    save_figure(FIG_DIR, f"cluster_{target_class}_force_plot.png")
    plt.close()

def shap_interpretation(model, X_enc, y_test):
    """
    Interpret features for each target class from the classifier model using SHAP.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        The trained random forest classifier model.
    X_enc : pandas.DataFrame
        The encoded input features.
    y_test : pandas.Series
        The target variable corresponding to X_enc.

    Returns
    -------
    None
    
    Notes
    ------
    The function does the following for each class:
    1. Plots the SHAP feature importance plot and saves it.
    2. Finds the most confident prediction and plots the SHAP force plot.
    3. Creates a boxplot for the top 6 features with the highest mean SHAP value.
    All the plots are saved as PNG files.
    """
    plotted_cols = set()
    full_df = X_enc.copy(deep=True)
    full_df["labels"] = y_test
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_enc)
    y_test_index_reset = y_test.reset_index(drop=True)
    for i in range(len(shap_values)):
        plot_shap_feature_importance(shap_values[i], X_enc, i)
        indices = y_test_index_reset[y_test_index_reset == i].index.tolist()
        pred_probs = model.predict_proba(X_enc.iloc[indices])
        most_confident_pred_idx = np.argmax(pred_probs[:, i])
        plot_shap_force_plot(explainer, shap_values[i], X_enc, i, indices[most_confident_pred_idx])
        mean_shap_vals_by_cluster = pd.DataFrame(
            np.abs(shap_values[i].mean(0)),
            columns=["mean SHAP value"],
            index=X_enc.columns.tolist()
        ).sort_values(by="mean SHAP value", ascending=False)
        for col in mean_shap_vals_by_cluster.index[:6]:
            if col not in plotted_cols:
                ax = plot_boxplot(full_df, col, "labels")
                ax.suptitle(f"Boxplot distribution of {col}")
                save_figure(FIG_DIR, f"{col}_boxplot.png")
                plotted_cols.add(col)
                plt.clf()

def main():
    train_df, test_df = read_data()
    X_train = train_df.drop(columns=["labels"])
    y_train = train_df["labels"]
    X_test = test_df.drop(columns=["labels"])
    y_test = test_df["labels"]
    rf = fit_random_forest_classifier(X_train, y_train)
    print(f"Train accuracy: {rf.score(X_train, y_train)}")
    print(f"Test accuracy: {rf.score(X_test, y_test)}")
    shap_interpretation(rf, X_test, y_test)

if __name__ == "__main__":
    main()