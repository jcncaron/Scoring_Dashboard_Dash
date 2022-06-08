import dash
import html as html
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
                "Histogram plot for the features",
                test_df_predict['y_pred_binary'],
                'box')

fig2 = fig_scatter(
    test_df_predict,
    test_df_predict.columns[0],
    test_df_predict.columns[0],
    "Scatter plot for features",
    marginal_x="histogram",
    marginal_y="histogram",
)

fig7 = fig_bar_shap(feature_importance_name, feature_importance_value)

fig8 = fig_scatter(
    temp_df, "EXT_SOURCE_2", "EXT_SOURCE_2_shap", "SHAP dependence plot", color="EXT_SOURCE_3"
)

fig9 = fig_force_plot(
    feature_importance_single_explanation_value, sum_list, color, title_single
)

# application layout

app.layout = html.Div(
    style={"backgroundColor": "#F9F9F9"},
    children=[
        ##################################################################################################
        ##################################### Header #####################################################
        html.Div(children=[html.H3("Data Visualization for Scoring with Dash",
                                   style={"textAlign": "center"})],
                 id="header"),
        html.Div(children=[html.H4("Basic visualizations for the features",
                                   style={"textAlign": "center"}),html.Div(
                    [
                        ##################################################################################################
                        ########################### Histogram Plot ###################################################
                        html.Div(
                            [
                                html.P(
                                    "Choose the feature you want to plot for the histogram"
                                )
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        create_dcc(
                                            test_df_predict[most_important_num_feat].columns,
                                            "value_histo",
                                            test_df_predict[most_important_num_feat].columns[0],
                                        )
                                    ],
                                    className="three columns",
                                )
                            ],
                            className="row",
                        ),
                        dcc.Graph(
                            id="histogram",
                            figure=fig1,
                            style=style_plot,
                        ),
                    ],
                    className="nine columns"
                ),
                ##################################################################################################
                ########################### Scatter plot ##########################################
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    "Choose the features you want to plot for the scatter plot"
                                )
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                create_dcc(
                                                    test_df_predict.columns,
                                                    "value_x_scatter",
                                                    test_df_predict.columns[0],
                                                )
                                            ],
                                            className="row",
                                        ),
                                        html.Div(
                                            [
                                                create_dcc(
                                                    test_df_predict.columns,
                                                    "value_y_scatter",
                                                    test_df_predict.columns[0],
                                                )
                                            ],
                                            className="row",
                                        ),
                                        html.Div(
                                            [
                                                create_slider(
                                                    "value_slider2",
                                                    0,
                                                    1,
                                                    {
                                                        0: "None",
                                                        1: "Output",
                                                    },
                                                )
                                            ],
                                            className="row",
                                        ),
                                    ],
                                    className="twelve columns",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                create_radio_shape(
                                                    "radio_value_x_scatter"
                                                )
                                            ],
                                            className="one-half column",
                                        ),
                                        html.Div(
                                            [
                                                create_radio_shape(
                                                    "radio_value_y_scatter"
                                                )
                                            ],
                                            className="one-half column",
                                        ),
                                    ],
                                    className="one-half column",
                                ),
                            ],
                            className="row",
                        ),
                        dcc.Graph(
                            id="scatter-plot",
                            figure=fig2,
                            style=style_plot,
                        ),
                    ],
                    className="one-third column pretty_container",
                ), ]
        ),
        html.Div(
            [
                html.H4(
                    "Model Interpretability using SHAP values",
                    style={"textAlign": "center"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="feature-importance-plot",
                            figure=fig7,
                            style=style_plot,
                        )
                    ],
                    className="one-third column pretty_container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        create_dcc(
                                            list_shap_features,
                                            "id-shap-feature",
                                            list_shap_features[0],
                                        )
                                    ],
                                    className="one-third column",
                                ),
                                html.Div(
                                    [
                                        create_dcc(
                                            test_df_predict.columns,
                                            "id-feature1",
                                            test_df_predict.columns[0],
                                        )
                                    ],
                                    className="one-third column",
                                ),
                                html.Div(
                                    [
                                        create_dcc(
                                            test_df_predict.columns,
                                            "id-feature2",
                                            test_df_predict.columns[0],
                                        )
                                    ],
                                    className="one-third column",
                                ),
                            ],
                            className="row pretty_container",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="dependence-plot",
                                    figure=fig8,
                                    style=style_plot,
                                )
                            ],
                            className="row",
                        ),
                    ],
                    className="one-third column pretty_container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            "Choose a row from the Test set to explain"
                                        )
                                    ],
                                    className="two-thids column",
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="explanation",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in range(len(X_test))
                                            ],
                                            value=0,
                                        )
                                    ],
                                    className="one-third column",
                                ),
                            ],
                            className="row",
                        ),
                        dcc.Graph(
                            id="single-explanation-plot",
                            figure=fig9,
                            style=style_plot,
                        ),
                    ],
                    className="one-third column pretty_container",
                ),
            ],
            className="row pretty_container",
        ),
    ],
)


@app.callback(Output("histogram", "figure"), [Input("value_histo", "value")])
def plot_histogram(x_data):
    figure = fig_hist(test_df_predict,
                      x_data,
                      "Histogram plot",
                      test_df_predict['y_pred_binary'],
                      'box')
    return figure


@app.callback(
    Output("scatter-plot", "figure"),
    [
        Input("value_x_scatter", "value"),
        Input("value_y_scatter", "value"),
        Input("radio_value_x_scatter", "value"),
        Input("radio_value_y_scatter", "value"),
        Input("value_slider2", "value"),
    ],
)
def plot_scatter(x_data, y_data, x_radio, y_radio, slider2):
    if slider2 == 0:
        figure = fig_scatter(
            test_df_predict,
            x_data,
            y_data,
            "Scatter plot",
            marginal_x=x_radio,
            marginal_y=y_radio,
        )
    else:
        figure = fig_scatter(
            test_df_predict,
            x_data,
            y_data,
            "Scatter plot",
            marginal_x=x_radio,
            marginal_y=y_radio,
            color="Width",
        )
    return figure


@app.callback(
    Output("dependence-plot", "figure"),
    [
        Input("id-shap-feature", "value"),
        Input("id-feature1", "value"),
        Input("id-feature2", "value"),
    ],
)
def plot_dependence_shap(id_shap_feature, id_feature1, id_feature2):
    figure = fig_scatter(
        temp_df,
        id_feature1,
        id_shap_feature,
        "SHAP dependence plot",
        color=id_feature2,
    )
    return figure


@app.callback(
    Output("single-explanation-plot", "figure"),
    [Input("explanation", "value")],
)
def plot_single_explanation(explanation):
    (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    ) = shap_single_explanation(
        explainer, X_test, explanation, model, base_value
    )
    figure = fig_force_plot(
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single,
    )
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
