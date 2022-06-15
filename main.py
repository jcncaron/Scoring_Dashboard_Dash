from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import lightgbm

from dashboard_functions.functions import (
    create_dcc,
    create_radio_shape,
    shap_single_explanation,
    create_explanations,
    prepare_data
)
from dashboard_functions.figures import (
    fig_scatter,
    fig_bar_shap,
    fig_force_plot,
    fig_countplot,
    fig_overlaid_hist,
    fig_scatter_dependence,
    fig_score, fig_gauge
)

# initialize the app
app = Dash(__name__)

# Prepare dataframes and return machine learning model
X_test, test_df_predict, model, customer_ids = prepare_data()

# Create explanations from model and X_test
(
    shap_values,
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
                         "Histogram for the continuous features",
                         customer_ids[0])

# Define overlaid histogram figure
fig2 = fig_countplot(test_df_predict,
                     test_df_predict.columns[0],
                     "Countplot for the categorical features",
                     'y_pred_binary',
                     customer_ids[0])

# Define scatter plot figure
fig3 = fig_scatter(test_df_predict,
                   test_df_predict.columns[0],
                   test_df_predict.columns[0],
                   "Scatter plot for the continuous features",
                   customer_ids[0],
                   marginal_x='histogram',
                   marginal_y='histogram',
                   color='y_pred_binary')

# Define barplot figure for shap values
fig4 = fig_bar_shap(feature_importance_name[0:20],
                    feature_importance_value[0:20])

# Define shap dependence plot figure
fig5 = fig_scatter_dependence(temp_df,
                              "EXT_SOURCE_2",
                              "EXT_SOURCE_3",
                              "Scatter plot",
                              customer_ids[0],
                              color="y_pred_proba")

# Define local bar plot importance feature
fig6 = fig_force_plot(feature_importance_single_explanation_value[0:15],
                      sum_list[0:15],
                      color,
                      title_single)

# Define predict_proba score plot for customers
fig7 = fig_score(customer_ids[0],
                 test_df_predict)

# Define gauge score for customers
fig8 = fig_gauge(customer_ids[0], test_df_predict)


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
        ###############################################################################################
        ######################### Local feature importance / SHAP values #############################
        html.Div(children=[
            html.H4("Scoring & local feature importance using SHAP values",
                    style={"textAlign": "center"}),
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.P("Choose a customer ID")],
                        className="two-thirds column"),
                    dcc.Dropdown(id="customer",
                                 options=[{"label": i, "value": i} for i in customer_ids],
                                 value=customer_ids[0])],
                    className="two-thirds column")],
                className="row pretty_container"),
            ######################### Predict proba score gauge ##########################
            html.Div(children=[
                dcc.Graph(id="predict-proba-gauge",
                          figure=fig8,
                          style=style_plot)],
                className="five columns pretty_container"),
            ######################### Feature importance plot #############################
            html.Div(children=[
                dcc.Graph(id="single-explanation-plot",
                          figure=fig6,
                          style=style_plot)],
                className="seven columns pretty_container"),
        ]
        ),
        ###############################################################################################
        ######################### Global feature importance / SHAP values #############################
        html.Div(children=[
            html.H4("Scatter plot & Global feature importance using SHAP values",
                    style={"textAlign": "center"}),
            ######################### Feature importance plot #############################
            html.Div(children=[
                dcc.Graph(id="feature-importance-plot",
                          figure=fig4,
                          style=style_plot)],
                className="one-half column pretty_container"),
            ######################### Feature scatter plot #############################
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        create_dcc(most_important_features[0:20],
                                   "id-feature1",
                                   most_important_features[0])],
                        className="six columns"),
                    html.Div(children=[
                        create_dcc(most_important_features[0:20],
                                   "id-feature2",
                                   most_important_features[1])],
                        className="six columns")],
                    className="row pretty_container"),
                html.Div(children=[
                    dcc.Graph(id="dependence-plot",
                              figure=fig5,
                              style=style_plot)],
                    className="row")],
                className="one-half column pretty_container"),
        ]),
        ##################################################################################################
        ############################# 1D and 2D feature visualization ####################################
        html.Div(children=[
            html.H4("Features distributions",
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
                                        test_df_predict[most_important_num_feat].columns[1])],
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
    ]
    )
    ]
)


@app.callback(Output('predict-proba-gauge', 'figure'), Input('customer', 'value'))
def plot_fig_gauge(customer_id):
    figure = fig_gauge(customer_id, test_df_predict)
    return figure


@app.callback(Output("single-explanation-plot", "figure"),
              [Input("customer", "value")])
def plot_single_explanation(customer_id):
    (
        feature_importance_single_explanation_value,
        sum_list,
        color,
        title_single
    ) = shap_single_explanation(X_test, customer_id, test_df_predict, base_value, shap_values, model)
    figure = fig_force_plot(
        feature_importance_single_explanation_value[0:15],
        sum_list[0:15],
        color,
        title_single)
    return figure


@app.callback(Output("histogram", "figure"),
              [Input("value_histo", "value"),
               Input("customer", "value")])
def plot_histogram(x_data, customer):
    figure = fig_overlaid_hist(test_df_predict,
                               x_data,
                               'Histogram',
                               customer)
    return figure


@app.callback(Output("bar_chart", "figure"),
              [Input("value_bar", "value"),
               Input("customer", "value")])
def plot_countplot(x_data, customer):
    figure = fig_countplot(test_df_predict,
                           x_data,
                           "Countplot",
                           'y_pred_binary',
                           customer)
    return figure


@app.callback(
    Output("scatter-plot", "figure"),
    [Input("value_x_scatter", "value"),
     Input("value_y_scatter", "value"),
     Input("radio_value_x_scatter", "value"),
     Input("radio_value_y_scatter", "value"),
     Input("customer", "value")])
def plot_scatter(x_data, y_data, x_radio, y_radio, customer):
    figure = fig_scatter(
        test_df_predict,
        x_data,
        y_data,
        "Scatter plot",
        customer,
        marginal_x=x_radio,
        marginal_y=y_radio,
        color='y_pred_binary')
    return figure


@app.callback(
    Output("dependence-plot", "figure"),
    [Input("id-feature1", "value"),
     Input("id-feature2", "value"),
     Input("customer", "value")])
def plot_dependence_shap(id_feature1, id_feature2, customer):
    figure = fig_scatter_dependence(
        temp_df,
        id_feature1,
        id_feature2,
        "Scatter plot",
        customer,
        color='y_pred_proba')
    return figure


# Run app
if __name__ == "__main__":
    app.run(debug=True)
