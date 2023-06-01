import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
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
        
def plot_shap_feature_importance(shap_values, X_enc, target_class):
    shap.summary_plot(shap_values, X_enc, show=False)
    plt.title(f"SHAP Feature Importance for cluster {target_class}")
    save_figure(FIG_DIR, f"cluster_{target_class}_summary_plot.png")
    plt.close()

def plot_shap_force_plot(explainer, shap_values, X_enc, target_class, idx_to_explain):
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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_enc)
    y_test_index_reset = y_test.reset_index(drop=True)
    for i in range(len(shap_values)):
        plot_shap_feature_importance(shap_values[i], X_enc, i)
        indices = y_test_index_reset[y_test_index_reset == i].index.tolist()
        pred_probs = model.predict_proba(X_enc.iloc[indices])
        most_confident_pred_idx = np.argmax(pred_probs[:, i])
        plot_shap_force_plot(explainer, shap_values[i], X_enc, i, indices[most_confident_pred_idx])

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