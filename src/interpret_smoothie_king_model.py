import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


DIR = 'data/Smoothie King/'
MODEL_DIR = "model_joblib/"
TARGET_MAP = {
    0: "HOME",
    1: "OTHER",
    2: "SHOPPING",
    3: "TRAVEL",
    4: "WORK"
}

def read_data():
    train_df = pd.read_csv(DIR + "train_df.csv", index_col="store")
    test_df = pd.read_csv(DIR + "test_df.csv", index_col="store")
    return train_df, test_df

def generate_confuson_matrix(model, X, y, labels, title, out_file):
    cm = confusion_matrix(y, model.predict(X))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    # plt.show()
    plt.title(title)
    plt.savefig(out_file)

def get_prediction_mismatch(prediction_result, model, true_label, predicted_label):
    # model: one of {"rf", "l1_reg_rf", "l1_reg_rf_ovr", "voting"}
    mismatch = prediction_result[(prediction_result["true_label"] == true_label) & (prediction_result[model] == predicted_label)]
    return mismatch.index.tolist()
    # print(mismatch.index)

def get_all_feature_names(df):
    ordinal_features = ["market_size", "store_density"]
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = numeric_features + ordinal_features
    return all_features

def shap_summary_plot(shap_values, X_test_enc, model_name, out_dir):
    plt.clf()
    for i in range(len(shap_values)):
        shap.summary_plot(shap_values[i], X_test_enc, max_display=10, plot_type="bar", show=False)
        plt.title(f"{model_name}\nSHAP Feature Importance for {TARGET_MAP[i]}")
        plt.xlabel("Mean absolute SHAP value")
        try:
            plt.savefig(out_dir + f"{TARGET_MAP[i]}_shap_summary_plot", bbox_inches="tight")
        except:
            # print("========Creating new directory=========")
            os.makedirs(out_dir)
            plt.savefig(out_dir + f"{TARGET_MAP[i]}_shap_summary_plot", bbox_inches="tight")

def shap_force_plot(explainer, shap_values, X_test_enc, class_indices, idx_to_explain, target_class):
    # TODO
    plt.clf()
    shap.force_plot(
        explainer.expected_value[target_class], 
        shap_values[0][class_indices[idx_to_explain], :], 
        X_test_enc.iloc[class_indices[idx_to_explain], :], 
        matplotlib=True,
    )
    plt.title("SHAP Force Plot")


def get_most_confident_and_correct(model, X_test, class_indices, target_class):
    # TODO
    model_pred_prob = model.predict_proba(X_test.iloc[class_indices])
    return np.argmax(model_pred_prob[:, target_class])

def shap_interpretation_for_rf_model(rf_model, X_test, y_test, feature_names):
    preprocessor = rf_model.named_steps["columntransformer"]
    X_test_enc = pd.DataFrame(
        data=preprocessor.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )
    rf_explainer = shap.TreeExplainer(rf_model.named_steps["randomforestclassifier"])
    test_rf_shap_values = rf_explainer.shap_values(X_test_enc)
    y_test_index_reset = y_test.reset_index(drop=True)
    class_0_indices = y_test_index_reset[y_test_index_reset == 0].index.tolist()
    class_1_indices = y_test_index_reset[y_test_index_reset == 1].index.tolist()
    class_2_indices = y_test_index_reset[y_test_index_reset == 2].index.tolist()
    class_3_indices = y_test_index_reset[y_test_index_reset == 3].index.tolist()
    class_4_indices = y_test_index_reset[y_test_index_reset == 4].index.tolist()
    # get top most confident correct and incorrect predictions for all classes
    shap_summary_plot(test_rf_shap_values, X_test_enc, "Random Forest", "img/smoothie_king/random_forest/")


def shap_interpretation_for_l1_reg_rf_model(l1_reg_rf_model, X_test, y_test, feature_names):
    preprocessor = l1_reg_rf_model.named_steps["column_transformer"]
    selected_features_mask = l1_reg_rf_model.named_steps['selectfrommodel'].get_support()
    selected_features = [feat for (feat, is_selected) in zip(feature_names, selected_features_mask) if is_selected]
    X_test_enc = pd.DataFrame(
        data=l1_reg_rf_model.named_steps["selectfrommodel"].transform(preprocessor.transform(X_test)),
        columns=selected_features,
        index=X_test.index,
    )

def main():
    train_df, test_df = read_data()
    le = LabelEncoder()
    train_df["category"] = le.fit_transform(train_df["category"])
    test_df["category"] = le.transform(test_df["category"])
    X_train = train_df.drop(columns=["category"])
    y_train = train_df["category"]
    X_test = test_df.drop(columns=["category"])
    y_test = test_df["category"]
    rf_model = load(MODEL_DIR + "rf_model.joblib")
    l1_reg_rf_model = load(MODEL_DIR + "l1_reg_rf_model.joblib")
    l1_reg_rf_ovr_model = load(MODEL_DIR + "l1_reg_rf_ovr_model.joblib")
    hard_voting_model = load(MODEL_DIR + "hard_voting_model.joblib")
    generate_confuson_matrix(
        hard_voting_model, X_train, y_train, le.classes_, "Confusion matrix for prediction on train set","train_cm.png"
    )
    generate_confuson_matrix(
        hard_voting_model, X_test, y_test, le.classes_, "Confusion matrix for prediction on test set", "test_cm.png"
    )
    prediction_result = pd.DataFrame({
        "true_label": y_test,
        "rf": rf_model.predict(X_test),
        "l1_reg_rf": l1_reg_rf_model.predict(X_test),
        "l1_reg_rf_ovr": l1_reg_rf_ovr_model.predict(X_test),
        "voting": hard_voting_model.predict(X_test)
    })
    print(get_prediction_mismatch(prediction_result, "voting", 0, 2))
    print(get_prediction_mismatch(prediction_result, "voting", 2, 0))
    all_feature_names = get_all_feature_names(X_train)
    shap_interpretation_for_rf_model(rf_model, X_test, y_test, all_feature_names)
    shap_interpretation_for_l1_reg_rf_model(l1_reg_rf_model, X_test, y_test, all_feature_names)



if __name__ == "__main__":
    main()