import pandas as pd
import numpy as np
from dash import dcc
import shap


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


def create_slider(name, mini, maxi, marks):
    """
    Create an html slider

    Arguments:
    - name : id name of the slider component
    - mini : minimum value
    - maxi : maximum value
    - marks : marks for each value ; example {0:"None", 1:"Output"}

    Returns:
    - slider component
    """
    return dcc.Slider(
        id=name,
        min=mini,
        max=maxi,
        marks=marks,
        value=0,
        className="pretty_container",
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
        value="histogram",
        className="pretty_container",
    )


def create_explainer(ml_model):
    """
    Creates a Tree explainer for machine learning model using shap values and lightgbm

    Argument:
    - ml_model: machine learning model

    Returns:
    - base_value: mean value of the predictions on the train set
    - explainer: SHAP tree explainer

    """

    explainer = shap.TreeExplainer(ml_model)
    base_value = explainer.expected_value[0]
    return base_value, explainer


def shap_dependence_plot(explainer, X_test, df):
    """
    Computes necessary steps to generate shap dependence plot and shap feature importance

    Arguments:
    - X_test: test set for the features
    - explainer: SHAP tree explainer
    - df : dataframe with X_test features + y_pred_proba and y_pred_binary features

    Returns:
    - feature_importance_name: list containing the name of the features order by importance
    - feature_importance_value: numpy array containing the importance value per feature
    - temp_df: dataframe containing X_train and shapley values for each instance/feature
    - list_shap_features: list containing the name of shap features

    """

    shap_values = explainer.shap_values(X_test)
    dataframe_shap = pd.DataFrame(
        shap_values[1],
        columns=list(map(lambda x: x + "_shap", X_test.columns.tolist())),
    )
    dataframe_shap = dataframe_shap[
        dataframe_shap.abs().sum().sort_values(ascending=False).index.tolist()
    ]
    feature_importance = (
        dataframe_shap.abs().sum().sort_values(ascending=False)
    )
    feature_importance_name = feature_importance.index.tolist()
    feature_importance_value = feature_importance.values
    dataframe_shap.index = X_test.index
    temp_df = pd.concat([df, dataframe_shap], 1)
    list_shap_features = list(dataframe_shap.columns)

    return (
        feature_importance_name,
        feature_importance_value,
        temp_df,
        list_shap_features,
        shap_values
    )


def shap_single_explanation(X_test, explanation, base_value, shap_values, temp_df):
    """
    Compute a single force plot for an instance from the test set

    Arguments:
    - X_test: test set for the features
    - explanation: index of X_test (instance to explain)
    - ml_model: trained lightgbm model
    - base_value: mean value of the predictions on the train set
    - shap_values: shap_values calculated from explainer

    Returns:
    - feature_importance_single_explanation_value: sorted importance values for the features
    - sum_list: list containing  strings of the name of each feature and its value
    - color: Blue for negative values and Crimson for the positive values
    - title_single: title of shap force plot

    """
    dataframe_single_explanation = pd.DataFrame(
        [shap_values[0][explanation]],
        columns=X_test.columns,
    )
    sorted_importance = dataframe_single_explanation.iloc[0, :]
    sorted_importance = sorted_importance[sorted_importance.abs().sort_values(ascending=False).index.tolist()]

    feature_importance_single_explanation_name = (
        sorted_importance.index.tolist()
    )
    feature_importance_single_explanation_value = sorted_importance.values
    color = np.array(
        ["rgb(255,255,255)"]
        * feature_importance_single_explanation_value.shape[0]
    )
    color[feature_importance_single_explanation_value < 0] = "Blue"
    color[feature_importance_single_explanation_value > 0] = "Crimson"
    list_ordered_values = X_test.iloc[explanation, :][
        feature_importance_single_explanation_name
    ].values
    sum_list = []
    for (item1, item2) in zip(
        feature_importance_single_explanation_name, list_ordered_values
    ):
        sum_list.append(" = ".join([item1, str(item2)]))
    predicted_value = temp_df['y_pred_binary'][explanation]
    title_single = "Feature importance: Base value: {} , Predicted value: {}".format(
        base_value, predicted_value
    )
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
    - base_value: mean value of the predictions on the train set
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

    base_value, explainer = create_explainer(ml_model)

    (
        feature_importance_name,
        feature_importance_value,
        temp_df,
        list_shap_features,
        shap_values
    ) = shap_dependence_plot(explainer, X_test, df)

    (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    ) = shap_single_explanation(X_test, 0, base_value, shap_values, temp_df)

    return (
        base_value,
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
