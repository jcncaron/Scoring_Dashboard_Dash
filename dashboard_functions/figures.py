import plotly.express as px
import plotly.graph_objects as go


def fig_update_layout(fig):
    """
    Update the figure with a specific layout
    """
    fig.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return fig


def fig_scatter(
    df, df_x, df_y, title, marginal_x=None, marginal_y=None, color=None
):
    """
    Scatter plot for the first feature of the dataframe
    """
    fig = px.scatter(
        df,
        x=df_x,
        y=df_y,
        marginal_x=marginal_x,
        marginal_y=marginal_y,
        title=title,
        color=color,
    )
    fig = fig_update_layout(fig)
    return fig


def fig_hist(df, df_x, title):
    """
    Histogram plot for the first feature of the dataframe
    """
    fig = px.histogram(df, x=df_x, title=title)
    fig = fig_update_layout(fig)
    return fig


def fig_bar_shap(feature_importance_name, feature_importance_value):
    """
    Global feature importance using shap values

    Arguments:
    - feature_importance_name : list containing the names of features
    - feature_importance_value : list containing the values of importance of features
    """
    fig = go.Figure(
        [go.Bar(x=feature_importance_name, y=feature_importance_value)],
        layout={"title": "Feature importance"},
    )
    fig = fig_update_layout(fig)
    return fig


def fig_force_plot(
    feature_importance_single_explanation_value, sum_list, color, title_single
):
    """
    Force plot for each value of X_test

    Arguments:
    - feature_importance_single_explanation_value: sorted importance values for the features
    - sum_list: list containing  strings of the name of each feature and its value
    - color: Blue for negative values and Crimson for the positive values
    - title_single : title of shap force plot
    """
    fig = go.Figure(
        [
            go.Bar(
                x=feature_importance_single_explanation_value,
                y=sum_list,
                orientation="h",
                marker_color=color,
            )
        ],
        layout={"title": title_single},
    )
    fig = fig_update_layout(fig)
    return fig