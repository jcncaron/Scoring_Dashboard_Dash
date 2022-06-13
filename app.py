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
    fig_force_plot,
    fig_countplot,
    fig_overlaid_hist,
    fig_scatter_dependence,
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

# Define list of most important features for numerical and categorical features
most_important_num_feat = num_feat[0:20]
most_important_cat_feat = cat_feat[0:20]


# Plots at the initialization
style_plot = {
    "border-radius": "5px",
    "background-color": "#f9f9f9",
    "box-shadow": "2px 2px 2px lightgrey",
    "margin": "3px",
}
# Define overlaid histogram figure
fig1 = fig_overlaid_hist(test_df_predict,
                         test_df_predict.columns[0],
                         "Histogram for the continuous features")

# Define overlaid histogram figure
fig2 = fig_countplot(test_df_predict,
                     test_df_predict.columns[0],
                     "Countplot for the categorical features",
                     'y_pred_binary')

# Define scatter plot figure
fig3 = fig_scatter(test_df_predict,
                   test_df_predict.columns[0],
                   test_df_predict.columns[0],
                   "Scatter plot for the continuous and categorical features",
                   marginal_x='histogram',
                   marginal_y='histogram',
                   color='y_pred_binary')

# Define barplot figure for shap values
fig4 = fig_bar_shap(feature_importance_name[0:20],
                    feature_importance_value[0:20])

# Define shap dependence plot figure
fig5 = fig_scatter_dependence(temp_df,
                              "EXT_SOURCE_2",
                              "EXT_SOURCE_2_shap",
                              "SHAP dependence plot",
                              color="EXT_SOURCE_2")

# Define local bar plot importance feature
fig6 = fig_force_plot(feature_importance_single_explanation_value[0:10],
                      sum_list[0:10],
                      color,
                      title_single)

# application layout
app.layout = html.Div(
    style={"backgroundColor": "#F9F9F9"},
    children=[html.Div(children=[
        ##################################################################################################
        ##################################### Header #####################################################
        html.Div(children=[
            html.Div(children=[
                html.Div(children=[
                    html.H3("Data Visualization for Scoring with Dash",
                            style={"textAlign": "center"})])],
                className="twelve columns",
                id="title")],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"}),
        ##################################################################################################
        ############################# 1D and 2D feature visualization ####################################
        html.Div(children=[
            html.H4("Basic visualizations for the features",
                    style={"textAlign": "center"}),
            html.Div(children=[
                ########################### Histogram Plot #####################################
                html.Div(children=[
                    html.Div(children=[
                        html.Div(children=[
                            html.P("Choose the continuous feature you want to plot for the histogram")],
                            className="row"),
                        html.Div(children=[
                            html.Div(children=[
                                create_dcc(test_df_predict[most_important_num_feat].columns,
                                           "value_histo",
                                           test_df_predict[most_important_num_feat].columns[0])],
                                className="twelve columns")],
                            className="row"),
                        dcc.Graph(id="histogram",
                                  figure=fig1,
                                  style=style_plot)])],
                    className="one-third column pretty_container"),
                ############################## Bar Plot ########################################
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
                                className="twelve columns")],
                            className="row"),
                        dcc.Graph(id="bar_chart",
                                  figure=fig2,
                                  style=style_plot)])],
                    className="one-third column pretty_container"),
                ############################ Scatter Plot ######################################
                html.Div(children=[
                    html.Div(children=[
                        html.Div(children=[
                            html.P("Choose the features you want to plot for the scatter plot")],
                            className="row"),
                        html.Div(children=[
                            html.Div(children=[
                                html.Div(children=[
                                    create_dcc(
                                        test_df_predict[most_important_num_feat].columns,
                                        "value_x_scatter",
                                        test_df_predict[most_important_num_feat].columns[0])],
                                    className="row"),
                                html.Div(children=[
                                    create_dcc(
                                        test_df_predict[most_important_num_feat].columns,
                                        "value_y_scatter",
                                        test_df_predict[most_important_num_feat].columns[0])],
                                    className="row")],
                                className="one-half column"),
                            html.Div(children=[
                                html.Div(children=[
                                    create_radio_shape("radio_value_x_scatter")],
                                    className="one-half column"),
                                html.Div(children=[
                                    create_radio_shape("radio_value_y_scatter")],
                                    className="one-half column")],
                                className='one-half column')],
                            className="row"),
                        dcc.Graph(id="scatter-plot",
                                  figure=fig3,
                                  style=style_plot)])],
                    className="one-third column pretty_container")
            ]
            )
        ]
        ),
        ###############################################################################################
        ######################### Global feature importance / SHAP values #############################
        html.Div(children=[
            html.H4("Global feature importance using SHAP values",
                    style={"textAlign": "center"}),
            ######################### Feature importance plot #############################
            html.Div(children=[
                dcc.Graph(id="feature-importance-plot",
                          figure=fig4,
                          style=style_plot)],
                className="one-half column pretty_container"),
            ######################### Feature dependance plot #############################
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        create_dcc(list_shap_features[0:20],
                                   "id-shap-feature",
                                   list_shap_features[0])],
                        className="four columns"),
                    html.Div(children=[
                        create_dcc(most_important_features[0:20],
                                   "id-feature1",
                                   most_important_features[0])],
                        className="four columns"),
                    html.Div(children=[
                        create_dcc(most_important_features[0:20],
                                   "id-feature2",
                                   most_important_features[0])],
                        className="four columns")],
                    className="row pretty_container"),
                html.Div(children=[
                    dcc.Graph(id="dependence-plot",
                              figure=fig5,
                              style=style_plot)],
                    className="row")],
                className="one-half column pretty_container")]),
        ###############################################################################################
        ######################### Local feature importance / SHAP values #############################
        html.Div(children=[
            html.H4("Local feature importance using SHAP values",
                    style={"textAlign": "center"}),
            ######################### Feature importance plot #############################
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.P("Choose a row from the Test set to explain")],
                        className="two-thirds column"),
                    html.Div(children=[
                        dcc.Dropdown(id="explanation",
                                     options=[{"label": i, "value": i} for i in range(len(X_test))],
                                     value=0)],
                        className="one-third column")],
                    className="row"),
                dcc.Graph(id="single-explanation-plot",
                          figure=fig6,
                          style=style_plot)],
                className="two-third column pretty_container")
        ]
        )
    ]
    )
    ]
)


@app.callback(Output("histogram", "figure"), [Input("value_histo", "value")])
def plot_histogram(x_data):
    figure = fig_overlaid_hist(test_df_predict,
                               x_data,
                               'Histogram')
    return figure


@app.callback(Output("bar_chart", "figure"), [Input("value_bar", "value")])
def plot_countplot(x_data):
    figure = fig_countplot(test_df_predict,
                           x_data,
                           "Countplot",
                           'y_pred_binary')
    return figure


@app.callback(
    Output("scatter-plot", "figure"),
    [Input("value_x_scatter", "value"),
     Input("value_y_scatter", "value"),
     Input("radio_value_x_scatter", "value"),
     Input("radio_value_y_scatter", "value")])
def plot_scatter(x_data, y_data, x_radio, y_radio):
    figure = fig_scatter(
        test_df_predict,
        x_data,
        y_data,
        "Scatter plot",
        marginal_x=x_radio,
        marginal_y=y_radio,
        color='y_pred_binary')
    return figure


@app.callback(
    Output("dependence-plot", "figure"),
    [Input("id-shap-feature", "value"),
     Input("id-feature1", "value"),
     Input("id-feature2", "value")])
def plot_dependence_shap(id_shap_feature, id_feature1, id_feature2):
    figure = fig_scatter_dependence(
        temp_df,
        id_feature1,
        id_shap_feature,
        "SHAP dependence plot",
        color=id_feature2)
    return figure


@app.callback(Output("single-explanation-plot", "figure"),
             [Input("explanation", "value")])
def plot_single_explanation(explanation):
    (feature_importance_single_explanation_value[0:10],
     sum_list[0:10],
     color,
     title_single) \
        = shap_single_explanation(explainer, X_test, explanation, model, base_value)
    figure = fig_force_plot(
        feature_importance_single_explanation_value[0:10],
        sum_list[0:10],
        color,
        title_single)
    return figure


if __name__ == "__main__":
    app.run_server(debug=True, port=2850)
