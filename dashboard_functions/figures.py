import plotly.express as px
import plotly.graph_objects as go
from dashboard_functions.functions import return_cust_index_from_id, return_score_from_id


def fig_update_layout(fig):
    """
    Update the figure with a specific layout
    """
    fig.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return fig


def fig_scatter(df, df_x, df_y, title, marginal_x=None, marginal_y=None, color=None):
    """
    Scatter plot for the first feature of the dataframe
    """
    df[color] = df[color].astype(object)
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


def fig_scatter_dependence(df, df_x, df_y, title, marginal_x=None, marginal_y=None, color=None):
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


def fig_hist(df, df_x, title, color, marginal):
    """
    Histogram plot for the continuous features of the dataframe
    """
    fig = px.histogram(df, x=df_x, title=title, color=color, marginal=marginal)
    fig = fig_update_layout(fig)
    return fig


def fig_overlaid_hist(df, df_x, title):
    """
    Histogram plot for the continuous features of the dataframe
    """
    x0 = df.loc[df['y_pred_binary'] == 0][df_x]
    x1 = df.loc[df['y_pred_binary'] == 1][df_x]
    name_x0 = 'y_pred = 0'
    name_x1 = 'y_pred = 1'
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0,
                               name=name_x0,
                               nbinsx=30))
    fig.add_trace(go.Histogram(x=x1,
                               name=name_x1,
                               nbinsx=30))
    fig.update_traces(opacity=0.6)
    fig.update_layout(
        barmode='overlay',
        title_text=title,  # title of plot
        xaxis_title_text=df_x,  # xaxis label
        yaxis_title_text='Count',  # yaxis label
    )
    return fig


def fig_gauge(customer_id, df):
    score = return_score_from_id(customer_id, df)
    fig = go.Figure(go.Indicator(
        mode="gauge + number + delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predict_proba Score",
               'font': {'size': 40}},
        delta={'reference': 0.09,
               'increasing': {'color': "red"},
               'decreasing': {'color': "green"}, },
        gauge={
            'axis': {'range': [0, 1],
                     'dtick': 0.05,
                     'tickwidth': 2,
                     'tickcolor': "black",
                     'tickfont': {'size': 25},
                     'ticklabelstep': 2},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 0.09], 'color': '#2ca02c'},
                {'range': [0.09, 1], 'color': '#d62728'}],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 1,
                'value': 0.09}}))
    fig.update_layout(paper_bgcolor="white", font={'color': "black", 'family': "Arial"})
    fig = fig_update_layout(fig)
    return fig


def fig_countplot(df, df_x, title, color):
    """
    Count plot for the categorical features of the dataframe
    """
    df = df.groupby(by=[df_x, color]).size().reset_index(name="counts")
    df[df_x] = df[df_x].astype(object)
    df[color] = df[color].astype(object)
    fig = px.bar(df, x=df_x, y='counts', title=title, color=color, barmode="group", text_auto='.3s')
    fig = fig_update_layout(fig)
    return fig


def fig_bar_shap(feature_importance_name, feature_importance_value):
    """
    Global feature importance using shap values

    Arguments:
    - feature_importance_name : list containing the names of features
    - feature_importance_value : list containing the values of importance of features
    """
    feature_importance_name_w_shap = [x.replace('_shap', '') for x in feature_importance_name]
    fig = go.Figure(
        [go.Bar(x=feature_importance_name_w_shap, y=feature_importance_value)],
        layout={"title": "Global feature importance (Absolute values)"})
    fig.update_xaxes(tickangle=-45)
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
        [go.Bar(x=feature_importance_single_explanation_value,
                y=sum_list,
                orientation="h",
                marker_color=color)],
        layout={"title": title_single})
    fig.update_layout(title={'text': title_single,
                             'x': 0.5,
                             'xanchor': 'center'})
    fig = fig_update_layout(fig)
    return fig


def fig_score(customer_id, df):
    """
    Bar plot to display predict_proba score in comparison of predict threshold

    Arguments:
    - customer_id : SK_ID_CURR value (int)
    - df : dataframe
    """
    # Save cust_index and one_row_df
    cust_index, one_row_df = return_cust_index_from_id(customer_id, df)
    fig = px.bar(one_row_df,
                 x='SK_ID_CURR',
                 y='y_pred_proba',
                 text_auto=True)
    # Display predict threshold
    fig.add_hline(y=0.09)
    fig.update_layout(title={'text': "Score",
                             'x': 0.5,
                             'xanchor': 'center'},
                      xaxis_title=f'Customer n°{customer_id}',
                      yaxis_range=[0, 1],
                      autosize=False,
                      width=300,
                      height=500)
    fig.update_xaxes(showticklabels=False)

    if df['y_pred_proba'][cust_index] < 0.09:
        bar_color = 'green'
    else:
        bar_color = 'red'

    fig.update_traces(marker_color=bar_color)

    return fig