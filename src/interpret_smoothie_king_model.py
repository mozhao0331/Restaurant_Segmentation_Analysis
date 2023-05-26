import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
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
    plt.title(title)
    try:
        plt.savefig("img/smoothie_king/" + out_file)
    except:
        os.makedirs("img/smoothie_king/")
        plt.savefig("img/smoothie_king/" + out_file)


def get_prediction_mismatch(prediction_result, model, true_label, predicted_label):
    # Helper function to get mismatched predictions
    # model: one of {"rf", "l1_reg_rf", "l1_reg_rf_ovr", "voting"}
    mismatch = prediction_result[(prediction_result["true_label"] == true_label) & (prediction_result[model] == predicted_label)]
    return mismatch.index.tolist()

def get_all_feature_names(df):
    ordinal_features = ["market_size", "store_density"]
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = numeric_features + ordinal_features
    return all_features

def shap_summary_plot(strategy_ovr, model_name, out_dir, shap_values=None, X_test_enc=None, estimators=None):
    plt.clf()
    if strategy_ovr:
        for i in range(len(estimators)):
            explainer = shap.TreeExplainer(estimators[i])
            shap_values = explainer.shap_values(X_test_enc)
            shap.summary_plot(shap_values[1], X_test_enc, show=False)
            plt.title(f"{model_name}\nSHAP Feature Importance for {TARGET_MAP[i]}")
            try:
                plt.savefig(out_dir + f"{TARGET_MAP[i]}_shap_summary_plot", bbox_inches="tight")
            except:
                os.makedirs(out_dir)
                plt.savefig(out_dir + f"{TARGET_MAP[i]}_shap_summary_plot", bbox_inches="tight")
            plt.clf()
    else:
        for i in range(len(shap_values)):
            # shap.summary_plot(shap_values[i], X_test_enc, max_display=10, plot_type="bar", show=False)
            shap.summary_plot(shap_values[i], X_test_enc, show=False)
            plt.title(f"{model_name}\nSHAP Feature Importance for {TARGET_MAP[i]}")
            try:
                plt.savefig(out_dir + f"{TARGET_MAP[i]}_shap_summary_plot", bbox_inches="tight")
            except:
                os.makedirs(out_dir)
                plt.savefig(out_dir + f"{TARGET_MAP[i]}_shap_summary_plot", bbox_inches="tight")
            plt.clf()



def shap_force_plot(explainer, shap_values, X_test_enc, class_indices, idx_to_explain, target_class, out_dir, pred_type):
    # TODO
    plt.clf()
    shap.force_plot(
        explainer.expected_value[target_class], 
        shap_values[target_class][class_indices[idx_to_explain], :], 
        X_test_enc.iloc[class_indices[idx_to_explain], :], 
        matplotlib=True,
        show=False
    )
    plt.suptitle("SHAP Force Plot")
    out_file = out_dir + pred_type + f"{TARGET_MAP[target_class]}_shap_force_plot"
    try:
        plt.savefig(out_file, bbox_inches="tight")
    except:
        os.makedirs(out_dir)
        plt.savefig(out_file, bbox_inches="tight")


def get_most_confident_and_correct(model, X_test, class_indices, target_class):
    # TODO
    model_pred_prob = model.predict_proba(X_test.iloc[class_indices])
    return np.argmax(model_pred_prob[:, target_class])

def encode_X_test(model, X_test, feature_names):
    preprocessor = model.named_steps["columntransformer"]
    try:
        selected_features_mask = model.named_steps['selectfrommodel'].get_support()
        selected_features = [feat for (feat, is_selected) in zip(feature_names, selected_features_mask) if is_selected]
        X_test_enc = pd.DataFrame(
            data=model.named_steps["selectfrommodel"].transform(preprocessor.transform(X_test)),
            columns=selected_features,
            index=X_test.index
        )
    except:
        X_test_enc = pd.DataFrame(
            data=preprocessor.transform(X_test),
            columns=feature_names,
            index=X_test.index,
        )
    return X_test_enc

def shap_interpretation_for_rf_model(rf_model, X_test, y_test, feature_names):
    X_test_enc = encode_X_test(rf_model, X_test, feature_names)
    explainer = shap.TreeExplainer(rf_model.named_steps["randomforestclassifier"])
    shap_values = explainer.shap_values(X_test_enc)
    shap_summary_plot( 
        strategy_ovr=False,
        model_name="Random Forest", 
        out_dir="img/smoothie_king/random_forest/",
        shap_values=shap_values, 
        X_test_enc=X_test_enc
    )
    y_test_index_reset = y_test.reset_index(drop=True)
    for i in TARGET_MAP.keys():
        print(i)
        indices = y_test_index_reset[y_test_index_reset == i].index.tolist()
        pred_probs = rf_model.predict_proba(X_test.iloc[indices])
        most_confident_pred_idx = np.argmax(pred_probs[:, i])
        least_confident_prd_idx = np.argmin(pred_probs[:, i])
        # most_confident_pred_row = X_test.iloc[indices[most_confident_pred_idx]]
        # least_confident_pred_row = X_test.iloc[indices[least_confident_prd_idx]]
        shap_force_plot(explainer, shap_values, X_test_enc, indices, most_confident_pred_idx, i, 
                        "img/smoothie_king/random_forest/", "most_confident_predict_")
        shap_force_plot(explainer, shap_values, X_test_enc, indices, least_confident_prd_idx, i, 
                        "img/smoothie_king/random_forest/", "least_confident_predict_")
    # y_test_index_reset = y_test.reset_index(drop=True)
    # class_0_indices = y_test_index_reset[y_test_index_reset == 0].index.tolist()
    # class_1_indices = y_test_index_reset[y_test_index_reset == 1].index.tolist()
    # class_2_indices = y_test_index_reset[y_test_index_reset == 2].index.tolist()
    # class_3_indices = y_test_index_reset[y_test_index_reset == 3].index.tolist()
    # class_4_indices = y_test_index_reset[y_test_index_reset == 4].index.tolist()
    # TODO: get top most confident correct and incorrect predictions for all classes


def shap_interpretation_for_l1_reg_rf_model(l1_reg_rf_model, X_test, y_test, feature_names):
    X_test_enc = encode_X_test(l1_reg_rf_model, X_test, feature_names)
    explainer = shap.TreeExplainer(l1_reg_rf_model.named_steps["randomforestclassifier"])
    shap_values = explainer.shap_values(X_test_enc)
    shap_summary_plot(
        strategy_ovr=False,
        model_name="L1 Regularized Random Forest", 
        out_dir="img/smoothie_king/l1_reg_random_forest/",
        shap_values=shap_values, 
        X_test_enc=X_test_enc
    )

def shap_interpretation_for_l1_reg_rf_ovr_model(l1_reg_rf_ovr_model, X_test, y_test, feature_names):
    X_test_enc = encode_X_test(l1_reg_rf_ovr_model, X_test, feature_names)
    estimators = l1_reg_rf_ovr_model.named_steps["onevsrestclassifier"].estimators_
    shap_summary_plot(
        strategy_ovr=True,
        model_name="L1 Regularized Random Forest OVR", 
        out_dir="img/smoothie_king/l1_reg_random_forest_ovr/",
        X_test_enc=X_test_enc,
        estimators=estimators
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
        hard_voting_model, X_train, y_train, le.classes_, "Confusion matrix for prediction on train set", "train_cm.png"
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
    # print(get_prediction_mismatch(prediction_result, "voting", 0, 2))
    # print(get_prediction_mismatch(prediction_result, "voting", 2, 0))
    all_feature_names = get_all_feature_names(X_train)
    shap_interpretation_for_rf_model(rf_model, X_test, y_test, all_feature_names)
    shap_interpretation_for_l1_reg_rf_model(l1_reg_rf_model, X_test, y_test, all_feature_names)
    shap_interpretation_for_l1_reg_rf_ovr_model(l1_reg_rf_ovr_model, X_test, y_test, all_feature_names)



if __name__ == "__main__":
    main()