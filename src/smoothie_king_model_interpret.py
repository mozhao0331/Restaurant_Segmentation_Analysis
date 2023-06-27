''' 
Script that performs model interpretation on the fitted Smoothie King models using SHAP.
Saves the SHAP output plots for human interpretation.
'''

import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DIR = 'data/Smoothie_King_Preprocessed/'
MODEL_DIR = "model_joblib/smoothie_king/"
TARGET_MAP = {
    0: "HOME",
    1: "OTHER",
    2: "SHOPPING",
    3: "TRAVEL",
    4: "WORK"
}

def read_data():
    '''Read in the train and test files.

    Returns
    -------
    train_df : pandas.DataFrame
        The train dataset
    test_df : pandas.DataFrame
        The test dataset
    '''
    train_df = pd.read_csv(DIR + "train_df.csv", index_col="store")
    test_df = pd.read_csv(DIR + "test_df.csv", index_col="store")
    return train_df, test_df

def generate_confuson_matrix(model, X, y, labels, title, out_file):
    '''Plots a confusion matrix.

    Parameters
    ----------
    model : sklearn classifier model
    X : pandas.DataFrame
        Dataset with features used for classification
    y : pandas Series
        Target column of dataset
    labels : array
        Array of target class labels
    title : str
    out_file : str
        File name to save confusion matrix image as
    
    Returns
    -------
    None
    '''
    cm = confusion_matrix(y, model.predict(X))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title(title)
    try:
        plt.savefig("img/smoothie_king/" + out_file)
    except:
        os.makedirs("img/smoothie_king/")
        plt.savefig("img/smoothie_king/" + out_file)
    plt.close()

def get_prediction_mismatch(prediction_result, model, true_label, predicted_label):
    '''Helper function to get mismatched predictions

    Parameters
    ----------
    prediction_result : pandas.DataFrame
    model : str
        one of {"rf", "l1_reg_rf", "l1_reg_rf_ovr", "voting"}
    true_label : int
    predicted_label : int

    Returns
    -------
    list
        List of indices where the predicted label is not equal to the true label
    '''
    mismatch = prediction_result[(prediction_result["true_label"] == true_label) & (prediction_result[model] == predicted_label)]
    return mismatch.index.tolist()

def get_all_feature_names(df):
    '''Get all column names of a DataFrame in order of numeric features first.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list
        List of column names
    '''
    ordinal_features = ["market_size", "store_density"]
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = numeric_features + ordinal_features
    return all_features

def shap_summary_plot(strategy_ovr, model_name, out_dir, shap_values=None, X_test_enc=None, estimators=None):
    '''Draw and save SHAP beeswarm plot for explaining feature importance
    
    Parameters
    ----------
    strategy_ovr : bool
        True if model is OVR, False otherwise
    model_name : str
    out_dir : str
        Directory to save output figures
    shap_values : array
        Matrix of SHAP values from SHAP Explainer shap_values() method
    X_test_enc : pandas.DataFrame
        Transformed X_test
    estimators : list
        List of estimators for each OVR case; used only if strategy_ovr=True
    
    Returns
    -------
    None
    ''' 
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
            # plt.clf()
            plt.close()
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
            plt.close()

def shap_force_plot(strategy_ovr, explainer, shap_values, X_test_enc, class_indices, idx_to_explain, target_class, out_dir, title):
    '''Draw SHAP force plot for a prediction point

    Parameters
    ----------
    strategy_ovr : bool
        True if model is OVR, False otherwise
    explainer : shap.TreeExplainer
    shap_values : array
        Matrix of SHAP values from SHAP Explainer shap_values() method
    X_test_enc : pandas.DataFrame
        Transformed X_test
    class_indices : list
    idx_to_explain : int
        Index of example to explain
    target_class : int
        True target class
    out_dir : str
        Directory to save output figures
    title : str

    Returns
    -------
    None
    '''
    out_file = out_dir + title.lower().replace(" ", "_") + f"_{TARGET_MAP[target_class]}_shap_force_plot"
    X_test_enc = X_test_enc.round(4)
    if strategy_ovr:
        shap.force_plot(
            explainer.expected_value[1], 
            shap_values[1][class_indices[idx_to_explain], :], 
            X_test_enc.iloc[class_indices[idx_to_explain], :], 
            matplotlib=True,
            show=False,
            text_rotation=20
        )
    else:
        shap.force_plot(
            explainer.expected_value[target_class], 
            shap_values[target_class][class_indices[idx_to_explain], :], 
            X_test_enc.iloc[class_indices[idx_to_explain], :], 
            matplotlib=True,
            show=False,
            text_rotation=20
        )
    title = title + f" SHAP Force Plot for {TARGET_MAP[target_class]} Class"
    plt.title(title, y=1.75)
    try:
        plt.savefig(out_file, bbox_inches="tight")
    except:
        os.makedirs(out_dir)
        plt.savefig(out_file, bbox_inches="tight")
    plt.close()

def encode_X_test(model, X_test, feature_names):
    '''Transform the test set using the pipeline's ColumnTransformer.

    model : sklearn Pipeline
        Pipeline containing, at a minimum, a ColumnTransformer and a classifier
    X_test : pandas.DataFrame
        Test dataset without the target column
    feature_names : list
        List of all column names with numeric features first
    
    Returns
    -------
    pandas.DataFrame
        Transformed X_test
    '''
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
    '''Perform interpretation on the random forest classifier model using SHAP

    Parameters
    ----------
    rf_model : sklearn Pipeline
        Pipeline containing the random forest classifier
    X_test : pandas.DataFrame
        The testing dataset without the target column
    y_test : pandas Series
        The target column of the test set to predict
    feature_names : list
        All column names with numeric features first
    
    Returns
    -------
    None
    '''
    out_dir = "img/smoothie_king/random_forest/"
    X_test_enc = encode_X_test(rf_model, X_test, feature_names)
    explainer = shap.TreeExplainer(rf_model.named_steps["randomforestclassifier"])
    shap_values = explainer.shap_values(X_test_enc)
    shap_summary_plot( 
        strategy_ovr=False,
        model_name="Random Forest", 
        out_dir=out_dir,
        shap_values=shap_values, 
        X_test_enc=X_test_enc
    )
    y_test_index_reset = y_test.reset_index(drop=True)
    for i in TARGET_MAP.keys():
        indices = y_test_index_reset[y_test_index_reset == i].index.tolist()
        pred_probs = rf_model.predict_proba(X_test.iloc[indices])
        most_confident_pred_idx = np.argmax(pred_probs[:, i])
        least_confident_prd_idx = np.argmin(pred_probs[:, i])
        shap_force_plot(
            strategy_ovr=False,
            explainer=explainer, 
            shap_values=shap_values, 
            X_test_enc=X_test_enc, 
            class_indices=indices, 
            idx_to_explain=most_confident_pred_idx, 
            target_class=i, 
            out_dir=out_dir, 
            title="Most Confident Prediction"
        )
        shap_force_plot(
            strategy_ovr=False, 
            explainer=explainer, 
            shap_values=shap_values, 
            X_test_enc=X_test_enc, 
            class_indices=indices, 
            idx_to_explain=least_confident_prd_idx, 
            target_class=i, 
            out_dir=out_dir, 
            title="Least Confident Prediction"
        )

def shap_interpretation_for_l1_reg_rf_model(l1_reg_rf_model, X_test, y_test, feature_names):
    '''Perform interpretation on the L1 regularized random forest classifier model using SHAP

    Parameters
    ----------
    l1_reg_rf_model : sklearn Pipeline
        Pipeline containing the L1 regularized random forest classifier
    X_test : pandas.DataFrame
        The testing dataset without the target column
    y_test : pandas Series
        The target column of the test set to predict
    feature_names : list
        All column names with numeric features first
    
    Returns
    -------
    None
    '''
    out_dir = "img/smoothie_king/l1_reg_random_forest/"
    X_test_enc = encode_X_test(l1_reg_rf_model, X_test, feature_names)
    explainer = shap.TreeExplainer(l1_reg_rf_model.named_steps["randomforestclassifier"])
    shap_values = explainer.shap_values(X_test_enc)
    shap_summary_plot(
        strategy_ovr=False,
        model_name="L1 Regularized Random Forest", 
        out_dir=out_dir,
        shap_values=shap_values, 
        X_test_enc=X_test_enc
    )
    y_test_index_reset = y_test.reset_index(drop=True)
    for i in TARGET_MAP.keys():
        indices = y_test_index_reset[y_test_index_reset == i].index.tolist()
        pred_probs = l1_reg_rf_model.predict_proba(X_test.iloc[indices])
        most_confident_pred_idx = np.argmax(pred_probs[:, i])
        least_confident_prd_idx = np.argmin(pred_probs[:, i])
        shap_force_plot(
            strategy_ovr=False, 
            explainer=explainer, 
            shap_values=shap_values, 
            X_test_enc=X_test_enc, 
            class_indices=indices, 
            idx_to_explain=most_confident_pred_idx, 
            target_class=i, 
            out_dir=out_dir, 
            title="Most Confident Prediction"
        )
        shap_force_plot(
            strategy_ovr=False, 
            explainer=explainer, 
            shap_values=shap_values, 
            X_test_enc=X_test_enc, 
            class_indices=indices, 
            idx_to_explain=least_confident_prd_idx, 
            target_class=i, 
            out_dir=out_dir, 
            title="Least Confident Prediction"
        )

def shap_interpretation_for_l1_reg_rf_ovr_model(l1_reg_rf_ovr_model, X_test, y_test, feature_names):
    '''Perform interpretation on the L1 regularized one vs rest random forest classifier model using SHAP

    Parameters
    ----------
    l1_reg_rf_ovr_model : sklearn Pipeline
        Pipeline containing the L1 regularized one vs rest random forest classifier
    X_test : pandas.DataFrame
        The testing dataset without the target column
    y_test : pandas Series
        The target column of the test set to predict
    feature_names : list
        All column names with numeric features first
    
    Returns
    -------
    None
    '''
    out_dir = "img/smoothie_king/l1_reg_random_forest_ovr/"
    X_test_enc = encode_X_test(l1_reg_rf_ovr_model, X_test, feature_names)
    estimators = l1_reg_rf_ovr_model.named_steps["onevsrestclassifier"].estimators_
    for i in range(len(estimators)):
        shap_summary_plot(
            strategy_ovr=True,
            model_name="L1 Regularized Random Forest OVR", 
            out_dir= out_dir,
            X_test_enc=X_test_enc,
            estimators=estimators
        )
    y_test_index_reset = y_test.reset_index(drop=True)
    for i in TARGET_MAP.keys():
        explainer = shap.TreeExplainer(estimators[i])
        shap_values = explainer.shap_values(X_test_enc)
        indices = y_test_index_reset[y_test_index_reset == i].index.tolist()
        pred_probs = estimators[i].predict_proba(X_test_enc.iloc[indices])
        most_confident_pred_idx = np.argmax(pred_probs[:, 1])
        least_confident_prd_idx = np.argmin(pred_probs[:, 1])
        shap_force_plot(
            strategy_ovr=True, 
            explainer=explainer, 
            shap_values=shap_values, 
            X_test_enc=X_test_enc, 
            class_indices=indices, 
            idx_to_explain=most_confident_pred_idx, 
            target_class=i, 
            out_dir=out_dir, 
            title="Most Confident Prediction"
        )
        shap_force_plot(
            strategy_ovr=True, 
            explainer=explainer, 
            shap_values=shap_values, 
            X_test_enc=X_test_enc, 
            class_indices=indices, 
            idx_to_explain=least_confident_prd_idx, 
            target_class=i, 
            out_dir=out_dir, 
            title="Least Confident Prediction"
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
    all_feature_names = get_all_feature_names(X_train)
    shap_interpretation_for_rf_model(rf_model, X_test, y_test, all_feature_names)
    shap_interpretation_for_l1_reg_rf_model(l1_reg_rf_model, X_test, y_test, all_feature_names)
    shap_interpretation_for_l1_reg_rf_ovr_model(l1_reg_rf_ovr_model, X_test, y_test, all_feature_names)

if __name__ == "__main__":
    main()