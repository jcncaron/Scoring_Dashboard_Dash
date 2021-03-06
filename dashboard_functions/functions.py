import pandas as pd
import numpy as np
import joblib
from dash import dcc
import shap


# function to prepare dataframe test_df_predict and X_test
def prepare_data():
    """
        Create X_test and test_df_predict
    """
    # load the dataframe with 1000 customers
    test_df = pd.read_csv("data/test_df_1000.csv")
    # remove special characters in test_df feature names
    test_df.columns = test_df.columns.str.replace(':', '')
    test_df.columns = test_df.columns.str.replace(',', '')
    test_df.columns = test_df.columns.str.replace(']', '')
    test_df.columns = test_df.columns.str.replace('[', '')
    test_df.columns = test_df.columns.str.replace('{', '')
    test_df.columns = test_df.columns.str.replace('}', '')
    test_df.columns = test_df.columns.str.replace('"', '')
    # Load machine learning model
    model = joblib.load("data/model_lgbm_1.joblib")
    # Predict_proba and predict class for X_test
    X_test = test_df.drop(['index', 'SK_ID_CURR'], 1)
    # Predict proba
    y_pred_proba = np.round(model.predict_proba(X_test)[:, 1], 3)
    # Predict class with a threshold = 0.09
    y_pred_binary = (model.predict_proba(X_test)[:, 1] >= 0.09).astype(int)
    # Add predict_proba and predict_class into test_df_predict
    test_df_predict = test_df
    test_df_predict['y_pred_proba'] = y_pred_proba
    test_df_predict['y_pred_binary'] = y_pred_binary
    # Create a list with all customer ids
    customer_ids = test_df_predict['SK_ID_CURR'].values.tolist()

    return X_test, test_df_predict, model, customer_ids


def shap_dependence_plot(explainer, X_test, df):
    """
    Computes necessary steps to generate shap dependence plot and shap feature importance

    Arguments:
    - explainer: SHAP tree explainer
    - X_test: test set for the features
    - df : dataframe with X_test features + y_pred_proba and y_pred_binary features

    Returns:
    - feature_importance_name: list containing the name of the features order by importance
    - feature_importance_value: numpy array containing the importance value per feature
    - temp_df: dataframe containing X_train and shapley values for each instance/feature
    - list_shap_features: list containing the name of shap features
    """

    # Create shap values with shap explainer and X_test
    shap_values = explainer.shap_values(X_test)
    # Create a dataframe with the shap values for each feature
    dataframe_shap = pd.DataFrame(
        shap_values[1],
        columns=list(map(lambda x: x + "_shap", X_test.columns.tolist())),
    )
    # Sort the features by decreasing importance in dataframe_shap
    dataframe_shap = dataframe_shap[
        dataframe_shap.abs().sum().sort_values(ascending=False).index.tolist()
    ]
    # Create a list of feature names and a list of feature shap values both sorted by decreasing feature importance
    feature_importance = (
        dataframe_shap.abs().sum().sort_values(ascending=False)
    )
    feature_importance_name = feature_importance.index.tolist()
    feature_importance_value = feature_importance.values
    # Create new temp_df with feature values and feature shap values
    dataframe_shap.index = X_test.index
    temp_df = pd.concat([df, dataframe_shap], 1)
    # Create a list of features from dataframe_shap
    list_shap_features = list(dataframe_shap.columns)

    return (
        feature_importance_name,  # list of features sorted by decreasing importance
        feature_importance_value,  # list of mean absolute shap values by decreasing importance
        temp_df,  # dataframe with feature values and feature shap values
        list_shap_features,
        shap_values
    )


def return_cust_index_from_id(customer_id, df):
    """
        Return the customer index in the dataframe from the customer id 'SK_ID_CURR
        & the 1 row dataframe corresponding to the customer_id

        Arguments:
        - customer_id : SK_ID_CURR value (int)
        - df : dataframe

        Returns:
        - customer_id
        - 1 row dataframe
        """
    # Save column index for column 'index'
    index_idx = df.columns.get_loc('index')
    # Create a dataframe with only 1 row, corresponding to customer_id
    one_row_df = df.loc[df['SK_ID_CURR'] == customer_id]
    # Save the customer index corresponding to the customer id
    cust_index = one_row_df.iloc[0, index_idx]

    return cust_index, one_row_df


def shap_single_explanation(X_test, customer_id, df, shap_values, ml_model):
    """
    Compute a single force plot for an instance from the test set

    Arguments:
    - X_test: test set for the features
    - explanation: index of X_test (instance to explain)
    - ml_model: trained lightgbm model
    - shap_values: shap_values calculated from explainer

    Returns:
    - feature_importance_single_explanation_value: sorted importance values for the features
    - sum_list: list containing  strings of the name of each feature and its value
    - color: Blue for negative values and Crimson for the positive values
    - title_single: title of shap force plot

    """
    # Return the customer index as 'explanation' and the corresponding dataframe 'one_row_df'
    explanation, one_row_df = return_cust_index_from_id(customer_id, df)
    # Create a one-row dataframe with the shap values for a single customer (index = explanation)
    dataframe_single_explanation = pd.DataFrame(
        [shap_values[0][explanation]],
        columns=X_test.columns,
    )
    # Sort the features by decreasing importance for this customer
    sorted_importance = dataframe_single_explanation.iloc[0, :]
    sorted_importance = sorted_importance[sorted_importance.abs().sort_values(ascending=False).index.tolist()]
    # Create lists of feature names and feature shap values both by decreasing importance
    feature_importance_single_explanation_name = (
        sorted_importance.index.tolist()
    )
    feature_importance_single_explanation_value = sorted_importance.values
    # Define the colors for the future bar plot
    color = np.array(
        ["rgb(255,255,255)"]
        * feature_importance_single_explanation_value.shape[0]
    )
    color[feature_importance_single_explanation_value < 0] = "Blue"
    color[feature_importance_single_explanation_value > 0] = "Crimson"
    # List of ordered shap values for the single customer
    list_ordered_values = X_test.iloc[explanation, :][
        feature_importance_single_explanation_name
    ].values
    # Create a list by joining feature names and corresponding shap values
    sum_list = []
    for (item1, item2) in zip(
            feature_importance_single_explanation_name, list_ordered_values
    ):
        sum_list.append(" = ".join([item1, str(item2)]))
    # Save the predicted class (0 or 1) for the single customer
    predicted_class = (ml_model.predict_proba(X_test)[:, 1] >= 0.09).astype(int)[explanation]
    # Save the title of the bar plot for the single customer
    title_single = "Local feature importance - Predicted class: {}".format(predicted_class)
    return (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    )


def create_explanations(ml_model, X_test, df):
    """
    Create SHAP explanations : feature importance and force plot.

    Arguments:
    - ml_model: machine learning model
    - X_test: test set for the features
    - df : dataframe with X_test features + y_pred_proba and y_pred_binary features

    Returns:
    - explainer: SHAP tree explainer
    - feature_importance_name: list containing the name of the features order by importance
    - feature_importance_value: numpy array containing the importance value per feature
    - temp_df: dataframe containing X_train and shapley values for each instance/feature
    - feature_importance_single_explanation_value:
    - sum_list: list containing  strings of the name of each feature and its value
    - title_single: title of shap force plot
    - color: Blue for negative values and Crimson for the positive values
    - list_shap_features: list containing the name of shap features
    """
    # Create the shap explainer
    explainer = shap.TreeExplainer(ml_model)

    # Run shap_dependence_plot function
    (
        feature_importance_name,
        feature_importance_value,
        temp_df,
        list_shap_features,
        shap_values
    ) = shap_dependence_plot(explainer, X_test, df)

    # Run shap_single_explanation function
    (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    ) = shap_single_explanation(X_test, 100001, df, shap_values, ml_model)

    return (
        shap_values,
        explainer,
        feature_importance_name,
        feature_importance_value,
        temp_df,
        feature_importance_single_explanation_value,
        sum_list,
        title_single,
        color,
        list_shap_features,
    )


def create_dcc(possible_values, name, default_value):
    """
    Create a html dcc component

    Arguments:
    - possible_values: possible values for label and value of each option
    - name: id name of the dcc component
    - default_value: default value of the dcc in the initialization of the app

    Returns:
    - dcc component
    """
    return dcc.Dropdown(
        id=name,
        options=[{"label": c, "value": c} for c in possible_values],
        value=default_value,
    )


def create_radio_shape(name):
    """
    Create a html radio component to choose the way to plot the distribution
    3 possibles modes: violin plot, box plot, histogram

    Arguments:
    - name : id name of the radio component

    Returns:
    - radio component
    """
    return dcc.RadioItems(
        id=name,
        options=[
            {"label": "box", "value": "box"},
            {"label": "hist", "value": "histogram"},
        ],
        value="box",
        className="pretty_container",
    )


def return_score_from_id(customer_id, df):
    """
        Return the predict_proba score from the customer id 'SK_ID_CURR'

        Arguments:
        - customer_id : SK_ID_CURR value (int)
        - df : dataframe

        Returns:
        - score
        """
    cust_index, one_row_df = return_cust_index_from_id(customer_id, df)
    score = df['y_pred_proba'][cust_index]

    return score
