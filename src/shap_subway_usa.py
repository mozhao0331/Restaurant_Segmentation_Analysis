import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import os
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DIR = "data/Subway USA/"
FIG_DIR = "img/subway_usa/"

def read_data():
    train_df = pd.read_csv(DIR + "train_df_with_labels.csv", index_col="store")
    test_df = pd.read_csv(DIR + "test_df_with_labels.csv", index_col="store")
    return train_df, test_df

def fit_random_forest_classifier(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def save_figure(out_dir, file_name):
    try:
        plt.savefig(out_dir + file_name, bbox_inches="tight")
    except:
        os.makedirs(out_dir)
        plt.savefig(out_dir + file_name, bbox_inches="tight")
        
def plot_shap_feature_importance(shap_values, X_enc, target):
    shap.summary_plot(shap_values, X_enc, show=False)
    plt.title(f"SHAP Feature Importance for cluster {target}")
    save_figure(FIG_DIR, f"cluster_{target}_summary_plot.png")
    plt.close()

def shap_interpretation(model, X_enc):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_enc)
    for i in range(len(shap_values)):
        plot_shap_feature_importance(shap_values[i], X_enc, i)
        # shap.summary_plot(shap_values[i], X_enc, show=False)
        # plt.title(f"SHAP Feature Importance for cluster {i}")
        # save_figure(FIG_DIR, f"cluster{i}_summary_plot.png")

def main():
    train_df, test_df = read_data()
    X_train = train_df.drop(columns=["labels"])
    y_train = train_df["labels"]
    X_test = test_df.drop(columns=["labels"])
    y_test = test_df["labels"]
    rf = fit_random_forest_classifier(X_train, y_train)
    print(f"Train accuracy: {rf.score(X_train, y_train)}")
    print(f"Test accuracy: {rf.score(X_test, y_test)}")
    shap_interpretation(rf, X_test)

if __name__ == "__main__":
    main()