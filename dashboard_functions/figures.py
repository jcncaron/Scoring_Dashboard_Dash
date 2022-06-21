import plotly.express as px
import plotly.graph_objects as go
from dashboard_functions.functions import return_score_from_id


def fig_update_layout(fig):
    """
    Update the figure with a specific layout
    """
    fig.update_layout(
        paper_bgcolor="#f9f9f9",
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
    )
    return fig


def fig_scatter(df, df_x, df_y, title, customer_id, marginal_x=None, marginal_y=None, color=None):
    """
    Scatter plot for the first feature of the dataframe

    Arguments:
    - df: dataframe
    - df_x: feature name for the x-axis
    - df_y: feature name for the y-axis
    - title: title of the scatter plot
    - customer_id: customer_id to highlight his dot on the scatter plot
    - marginal_x: show boxplot or histogram on x-axis  in addition of the scatter plot
    - marginal_y: show boxplot or histogram on y-axis  in addition of the scatter plot
    - color: to change the color of the dot according to the class
    """
    df[color] = df[color].astype(object)
    # Main plot
    fig = px.scatter(
        df,
        x=df_x,
        y=df_y,
        marginal_x=marginal_x,
        marginal_y=marginal_y,
        title=title,
        color=color,
    )

    ### Code to highlight the dot of a single customer (customer_id) on plot
    # Create a dataframe with only 1 row, corresponding to customer_id
    one_row_df = df.loc[df['SK_ID_CURR'] == customer_id]
    # Save column index for column df_x
    df_x_idx = one_row_df.columns.get_loc(df_x)
    # Save column index for column df_y
    df_y_idx = one_row_df.columns.get_loc(df_y)
    # Save x coordinate for customer_id and features df_x and df_y
    x = one_row_df.iloc[0, df_x_idx]
    # Save y coordinate for customer_id and features df_x and df_y
    y = one_row_df.iloc[0, df_y_idx]
    # Add the trace defined by x and y coordinates
    fig.add_trace(go.Scatter(x=[x], y=[y],
                             mode='markers',
                             marker_symbol='hexagram',
                             marker_size=12,
                             marker_color='gold',
                             name=f'cust n°{customer_id}'))

    fig = fig_update_layout(fig)
    return fig


def fig_scatter_dependence(df, df_x, df_y, title, customer_id, marginal_x=None, marginal_y=None, color=None):
    """
    Scatter plot for the first feature of the dataframe

    Arguments:
    - df: dataframe
    - df_x: feature name for the x-axis
    - df_y: feature name for the y-axis
    - title: title of the scatter plot
    - customer_id: customer_id to highlight his dot on the scatter plot
    - marginal_x: show boxplot or histogram on x-axis  in addition of the scatter plot
    - marginal_y: show boxplot or histogram on y-axis  in addition of the scatter plot
    - color: to change the color of the dot according to the class
    """
    # Main plot
    fig = px.scatter(
        df,
        x=df_x,
        y=df_y,
        marginal_x=marginal_x,
        marginal_y=marginal_y,
        title=title,
        color=color,
        color_continuous_scale='picnic',
    )

    ### Code to highlight the dot of a single customer (customer_id) on plot
    # Create a dataframe with only 1 row, corresponding to customer_id
    one_row_df = df.loc[df['SK_ID_CURR'] == customer_id]
    # Save column index for column df_x
    df_x_idx = one_row_df.columns.get_loc(df_x)
    # Save column index for column df_y
    df_y_idx = one_row_df.columns.get_loc(df_y)
    # Save x coordinate for customer_id and features df_x and df_y
    x = one_row_df.iloc[0, df_x_idx]
    # Save y coordinate for customer_id and features df_x and df_y
    y = one_row_df.iloc[0, df_y_idx]
    # Add the trace defined by x and y coordinates
    fig.add_trace(go.Scatter(x=[x], y=[y],
                             mode='markers',
                             marker_symbol='hexagram',
                             marker_size=12,
                             marker_color='gold',
                             name=f'cust n°{customer_id}'))

    fig.update_traces(showlegend=True,
                      textposition='top center')
    fig.update_coloraxes(colorbar_len=0.8)

    fig = fig_update_layout(fig)
    return fig


def fig_overlaid_hist(df, df_x, title, customer_id):
    """
    Histogram plot for the continuous features of the dataframe

    Arguments:
    - df: dataframe
    - df_x: continuous feature name for the x-axis
    - title: title of the histogram plot
    - customer_id: customer_id to highlight his position on the histogram plot
    """
    # Create values for a feature df_x according to the 'y_pred_binary' class
    x0 = df.loc[df['y_pred_binary'] == 0][df_x]
    x1 = df.loc[df['y_pred_binary'] == 1][df_x]
    # Create the name of the classes (for the legend)
    name_x0 = 'y_pred = 0'
    name_x1 = 'y_pred = 1'
    # Main histograms
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

    ### Code to highlight the position of a single customer (customer_id) on plot
    # Create a dataframe with only 1 row, corresponding to customer_id
    one_row_df = df.loc[df['SK_ID_CURR'] == customer_id]
    # Save column index for column df_x
    df_x_idx = one_row_df.columns.get_loc(df_x)
    # Save x coordinate for customer_id and features df_x and df_y
    x = one_row_df.iloc[0, df_x_idx]
    # Add the trace defined by x coordinate
    fig.add_trace(go.Scatter(x=[x], y=[0],
                             mode='markers',
                             marker_symbol='hexagram',
                             marker_size=12,
                             marker_color='gold',
                             name=f'cust n°{customer_id}'))

    return fig


def fig_gauge(customer_id, df):
    """
    Gauge to display the predict_proba score of each customer

    Arguments:
    - customer_id: customer_id 'SK_ID_CURR' of a single customer
    - df: dataframe
    """
    # Run score function
    score = return_score_from_id(customer_id, df)
    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge + number + delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predict_proba Score",
               'font': {'size': 30}},
        delta={'reference': 0.09,
               'increasing': {'color': "red"},
               'decreasing': {'color': "green"}, },
        gauge={
            'axis': {'range': [0, 1],
                     'dtick': 0.05,
                     'tickwidth': 2,
                     'tickcolor': "black",
                     'tickfont': {'size': 20},
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


def fig_countplot(df, df_x, title, color, customer_id):
    """
    Count plot for the categorical features of the dataframe

    Arguments:
    - df: dataframe
    - df_x: feature name for the x-axis
    - title: title of the scatter plot
    - customer_id: customer_id to highlight his position on the count plot
    - color: to change the color of the dot according to the class
    """
    # Create a dataframe with only 1 row, corresponding to customer_id
    one_row_df = df.loc[df['SK_ID_CURR'] == customer_id]
    # Save column index for column df_x
    df_x_idx = one_row_df.columns.get_loc(df_x)
    # Save x coordinate for customer_id and features df_x and df_y
    x = one_row_df.iloc[0, df_x_idx]
    # Create a dataframe with counts for each feature category and each class in this category
    df = df.groupby(by=[df_x, color]).size().reset_index(name="counts")
    df[df_x] = df[df_x].astype(object)
    df[color] = df[color].astype(object)
    # Main plot
    fig = px.bar(df,
                 x=df_x,
                 y='counts',
                 title=title,
                 color=color,
                 barmode="group",
                 text_auto='.3s')
    # Add the trace defined by x coordinate
    fig.add_trace(go.Scatter(x=[x], y=[0],
                             mode='markers',
                             marker_symbol='hexagram',
                             marker_size=12,
                             marker_color='gold',
                             name=f'cust n°{customer_id}'))

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
