import dash
from dash import dcc, html
import joblib
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import lightgbm

from dashboard_functions.functions import (
    create_dcc,
    create_slider,
    create_radio_shape,
    shap_single_explanation,
    create_explanations,
)
from dashboard_functions.figures import (
    fig_scatter,
    fig_hist,
    fig_bar_shap,
    fig_force_plot, fig_countplot,
)

# initialize the app
app = dash.Dash(__name__)

# load the dataframe
test_df = pd.read_csv("C:/Users/33624/PycharmProjects/Scoring dashboard with dash/data/test_df.csv")

# remove special characters in test_df feature names
test_df.columns = test_df.columns.str.replace(':', '')
test_df.columns = test_df.columns.str.replace(',', '')
test_df.columns = test_df.columns.str.replace(']', '')
test_df.columns = test_df.columns.str.replace('[', '')
test_df.columns = test_df.columns.str.replace('{', '')
test_df.columns = test_df.columns.str.replace('}', '')
test_df.columns = test_df.columns.str.replace('"', '')

# Load machine learning model
model = joblib.load("C:/Users/33624/model_lgbm_1.joblib")

# Predict_proba and predict class for X_test
X_test = test_df.drop(['index', 'SK_ID_CURR'], 1)
# Predict proba
y_pred_proba = np.round(model.predict_proba(X_test)[:, 1], 3)
# Predict class with a threshold = 0.09
y_pred_binary = (model.predict_proba(X_test)[:, 1] >= 0.09).astype(int)

# Add predict_proba and predict_class into test_df
test_df_predict = test_df
test_df_predict['y_pred_proba'] = y_pred_proba
test_df_predict['y_pred_binary'] = y_pred_binary

# Create explanations from model and X_test
(
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
) = create_explanations(model, X_test, test_df_predict)

# Define most important features for each type of features
most_important_features = [x.replace('_shap', '') for x in feature_importance_name]

# Define categorical and numeric features
cat_feat = []
num_feat = []

for feat in most_important_features:
    if len(test_df_predict[feat].unique()) > 20:
        num_feat.append(feat)
    else:
        cat_feat.append(feat)

most_important_num_feat = num_feat[0:20]
most_important_cat_feat = cat_feat[0:20]

# Plots at the initialization
style_plot = {
    "border-radius": "5px",
    "background-color": "#f9f9f9",
    "box-shadow": "2px 2px 2px lightgrey",
    "margin": "10px",
}

fig1 = fig_hist(test_df_predict,
                test_df_predict.columns[0],
                "Histogram plot for the continuous features",
                test_df_predict['y_pred_binary'],
                'box')

fig2 = fig_countplot(test_df_predict,
                     test_df_predict.columns[0],
                     "Countplot for the categorical features",
                     test_df_predict['y_pred_binary'])

# application layout
app.layout = html.Div(
    style={"backgroundColor": "#F9F9F9"},
    children=[html.Div(children=[
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    html.H3("Data Visualization for Scoring with Dash",
                            style={"textAlign": "center"})])],
                className="twelve column",
                id="title")],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"}),
        html.Div(children=[
            html.H4("Basic visualizations for the features",
                    style={"textAlign": "center"}),
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.Div(children=[
                            html.P("Choose the coutinuous feature you want to plot for the histogram")],
                            className="row"),
                        html.Div(children=[
                            html.Div(children=[
                                create_dcc(test_df_predict[most_important_num_feat].columns,
                                           "value_histo",
                                           test_df_predict[most_important_num_feat].columns[0])],
                                className="four columns")],
                            className="row"),
                        dcc.Graph(id="histogram",
                                  figure=fig1,
                                  style=style_plot)])],
                    className="seven columns pretty_container")]),
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.Div(children=[
                            html.P("Choose the categorical feature you want to plot for the bar chart")],
                            className="row"),
                        html.Div(children=[
                            html.Div(children=[
                                create_dcc(test_df_predict[most_important_cat_feat].columns,
                                           "value_bar",
                                           test_df_predict[most_important_cat_feat].columns[0])],
                                className="ten columns")],
                            className="row"),
                        dcc.Graph(id="bar chart",
                                  figure=fig2,
                                  style=style_plot)])],
                    className="five columns pretty_container")])
        ]
        )
    ]
    )
    ]
)


@app.callback(Output("histogram", "figure"), [Input("value_histo", "value")])
def plot_histogram(x_data):
    figure = fig_hist(test_df_predict,
                      x_data,
                      "Histogram boxplot",
                      test_df_predict['y_pred_binary'],
                      'box')
    return figure


@app.callback(Output("bar chart", "figure"), [Input("value_bar", "value")])
def plot_countplot(x_data):
    figure = fig_countplot(test_df_predict,
                           x_data,
                           "Countplot",
                           test_df_predict['y_pred_binary'])
    return figure


if __name__ == "__main__":
    app.run_server(debug=True, port=1100)
