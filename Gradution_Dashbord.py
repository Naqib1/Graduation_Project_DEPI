import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Load data
df = pd.read_csv('Telco_customer_churn.csv')

# Convert Total Charges to numeric and handle NaN values
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'] = df['Total Charges'].fillna(0)

# Initialize Dash app
app = dash.Dash(__name__)

# Convert Churn Label to string once
df['Churn Label'] = df['Churn Label'].astype(str)

# Plot generation functions
def plot_monthly_charges_analysis():
    """Generate improved plots for monthly charges distribution"""
    hist_data = [df['Monthly Charges'].dropna()]
    group_labels = ['Monthly Charges']
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Distribution of Monthly Charges', 
                                      'Density of Monthly Charges'))
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df['Monthly Charges'], name='Monthly Charges', 
                     marker_color='blue', nbinsx=30),
        row=1, col=1
    )
    
    # KDE Density Plot
    density_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    for trace in density_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    fig.update_layout(
        title='Monthly Charges Analysis',
        height=500,
        width=1000
    )
    return fig

def plot_tenure_vs_total_charges():
    """Generate plots comparing tenure and total charges"""
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=('Scatter Plot: Tenure vs. Total Charges',
                                      'Line Plot: Tenure vs. Total Charges',
                                      'Correlation: Tenure vs. Total Charges'))
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=df['Tenure Months'], y=df['Total Charges'], 
                   mode='markers', marker_color='blue', name='Data Points'),
        row=1, col=1
    )
    
    # Line plot (using mean values per tenure month)
    avg_by_tenure = df.groupby('Tenure Months')['Total Charges'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=avg_by_tenure['Tenure Months'], y=avg_by_tenure['Total Charges'], 
                  mode='lines+markers', marker_color='green', name='Average'),
        row=1, col=2
    )
    
    # Correlation heatmap
    corr = df[['Tenure Months', 'Total Charges']].corr().values
    fig.add_trace(
        go.Heatmap(z=corr, x=['Tenure Months', 'Total Charges'], 
                  y=['Tenure Months', 'Total Charges'],
                  colorscale='RdBu', zmin=-1, zmax=1, text=corr.round(2),
                  texttemplate="%{text}", colorbar_thickness=5),
        row=1, col=3
    )
    
    fig.update_layout(
        height=500,
        width=1200
    )
    
    return fig

def plot_contract_vs_churn():
    """Generate plots comparing contract type and churn"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Contract Type vs. Churn (Stacked)',
                                      'Contract Type vs. Churn (Count)'))
    
    # Stacked bar chart
    contract_churn = df.groupby(['Contract', 'Churn Label']).size().unstack()
    
    for churn_type in contract_churn.columns:
        fig.add_trace(
            go.Bar(x=contract_churn.index, y=contract_churn[churn_type], 
                   name=churn_type),
            row=1, col=1
        )
    
    # Count plot
    for i, churn_type in enumerate(df['Churn Label'].unique()):
        churn_data = df[df['Churn Label'] == churn_type]
        contract_counts = churn_data.groupby('Contract').size().reset_index(name='count')
        
        fig.add_trace(
            go.Bar(x=contract_counts['Contract'], y=contract_counts['count'], 
                  name=f"{churn_type}", offsetgroup=i),
            row=1, col=2
        )
    
    fig.update_layout(
        barmode='stack', 
        height=500,
        width=1200
    )
    
    # Update 2nd subplot to use 'group' mode
    fig.update_layout(barmode='group', bargap=0.15, bargroupgap=0.1)
    
    return fig

def plot_churn_analysis():
    """Generate plots analyzing churn based on monthly charges"""
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=('Box Plot: Monthly Charges by Churn Status',
                                      'Violin Plot: Monthly Charges by Churn Status',
                                      'Bar Plot: Avg Monthly Charges by Churn Status'))
    
    # Box plot
    for i, churn in enumerate(df['Churn Label'].unique()):
        churn_data = df[df['Churn Label'] == churn]['Monthly Charges']
        fig.add_trace(
            go.Box(y=churn_data, name=churn),
            row=1, col=1
        )
    
    # Violin plot
    for i, churn in enumerate(df['Churn Label'].unique()):
        churn_data = df[df['Churn Label'] == churn]['Monthly Charges']
        fig.add_trace(
            go.Violin(y=churn_data, name=churn, box_visible=True, 
                     meanline_visible=True),
            row=1, col=2
        )
    
    # Bar plot of means
    churn_means = df.groupby('Churn Label')['Monthly Charges'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=churn_means['Churn Label'], y=churn_means['Monthly Charges'],
              marker_color=['blue', 'red']),
        row=1, col=3
    )
    
    fig.update_layout(
        height=500,
        width=1600
    )
    
    return fig

def plot_correlation_heatmap():
    """Generate correlation heatmap for key metrics"""
    corr_features = df[["Total Charges", "Monthly Charges", "Tenure Months"]].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_features.values,
        x=corr_features.columns,
        y=corr_features.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_features.values.round(2),
        texttemplate="%{text}",
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Correlation Heatmap: Total Charges, Monthly Charges, and Tenure",
        height=600,
        width=600
    )
    
    return fig

def plot_scatter_matrix():
    """Generate scatter matrix with clean numeric data"""
    df_numeric = df[['Total Charges', 'Monthly Charges', 'Tenure Months', 'Churn Label']].dropna().copy()
    df_numeric = df_numeric.select_dtypes(include=['number'])
    
def plot_churn_vs_metrics():
    """Generate plots comparing churn with monthly charges and tenure"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Monthly Charges by Churn Status',
                                      'Tenure Months by Churn Status'))
    
    # Box plot for Monthly Charges
    for i, churn in enumerate(df['Churn Label'].unique()):
        fig.add_trace(
            go.Box(y=df[df['Churn Label'] == churn]['Monthly Charges'], name=churn),
            row=1, col=1
        )
    
    # Box plot for Tenure
    for i, churn in enumerate(df['Churn Label'].unique()):
        fig.add_trace(
            go.Box(y=df[df['Churn Label'] == churn]['Tenure Months'], name=churn),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        width=1000
    )
    
    return fig

def plot_services_vs_churn():
    """Generate plots comparing internet service and contract with churn"""
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Internet Service vs. Churn',
                                      'Contract Type vs. Churn'))
    
    # Internet Service vs Churn
    internet_churn = df.groupby(['Internet Service', 'Churn Label']).size().reset_index(name='count')
    
    for i, churn in enumerate(df['Churn Label'].unique()):
        data = internet_churn[internet_churn['Churn Label'] == churn]
        fig.add_trace(
            go.Bar(x=data['Internet Service'], y=data['count'], name=churn, offsetgroup=i),
            row=1, col=1
        )
        
    # Contract vs Churn
    contract_churn = df.groupby(['Contract', 'Churn Label']).size().reset_index(name='count')
    
    for i, churn in enumerate(df['Churn Label'].unique()):
        data = contract_churn[contract_churn['Churn Label'] == churn]
        fig.add_trace(
            go.Bar(x=data['Contract'], y=data['count'], name=churn, offsetgroup=i),
            row=1, col=2
        )
    
    fig.update_layout(
        barmode='group',
        height=500,
        width=1000
    )
    
    return fig

def plot_sampled_scatterplots():
    """Generate scatterplots from sampled data"""
    df_sampled = df.sample(frac=0.3, random_state=42)
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Monthly Charges vs. Total Charges',
                                      'CLTV vs. Churn Score'))
    
    # Monthly vs Total Charges
    fig.add_trace(
        go.Scatter(x=df_sampled['Monthly Charges'], y=df_sampled['Total Charges'],
                  mode='markers', marker=dict(color='blue', opacity=0.5),
                  name='Charges'),
        row=1, col=1
    )
    
    # CLTV vs Churn Score
    fig.add_trace(
        go.Scatter(x=df_sampled['CLTV'], y=df_sampled['Churn Score'],
                  mode='markers', marker=dict(color='red', opacity=0.5),
                  name='Churn Risk'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        width=1000
    )
    
    return fig

def plot_feature_correlation():
    """Generate heatmap for selected feature correlations"""
    columns_of_interest = ["Monthly Charges", "Total Charges", "Tenure Months", "Churn Score"]
    df_selected = df[columns_of_interest]
    df_selected["Total Charges"] = pd.to_numeric(df_selected["Total Charges"], errors='coerce')
    df_selected = df_selected.dropna()
    correlation_matrix = df_selected.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=correlation_matrix.values.round(2),
        texttemplate="%{text}"
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        width=600
    )
    
    return fig

def plot_full_correlation():
    """Generate full correlation matrix for all numeric columns"""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[num_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=correlation_matrix.values.round(2),
        texttemplate="%{text}"
    ))
    
    fig.update_layout(
        title="Full Correlation Matrix",
        height=800,
        width=800
    )
    
    return fig

# Define the main layout of the application
app.layout = html.Div([
    html.H1("Telco Customer Churn Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label='Monthly Charges Analysis', value='tab1'),
        dcc.Tab(label='Tenure vs. Total Charges Analysis', value='tab2'),
        dcc.Tab(label='Contract Type vs. Churn', value='tab3'),
        dcc.Tab(label='Churn Analysis', value='tab4'),
        dcc.Tab(label='Correlation Heatmap', value='tab5'),
        dcc.Tab(label='Churn vs. Monthly Charges & Tenure', value='tab8'),
        dcc.Tab(label='Internet Service & Contract vs. Churn', value='tab9'),
        dcc.Tab(label='Sampled Scatterplots', value='tab10'),
        dcc.Tab(label='Feature Correlation Heatmap', value='tab11'),
        dcc.Tab(label='Full Correlation Matrix', value='tab12')
    ]),
        html.Div(id='tabs-content', style={'padding': '20px'})
    ])

# Map tab names to their corresponding plot functions
TAB_FUNCTION_MAP = {
    'tab1': plot_monthly_charges_analysis,
    'tab2': plot_tenure_vs_total_charges,
    'tab3': plot_contract_vs_churn,
    'tab4': plot_churn_analysis,
    'tab5': plot_correlation_heatmap,
    'tab8': plot_churn_vs_metrics,
    'tab9': plot_services_vs_churn,
    'tab10': plot_sampled_scatterplots,
    'tab11': plot_feature_correlation,
    'tab12': plot_full_correlation
}

# Callback to update tab content
@app.callback(
    dash.dependencies.Output('tabs-content', 'children'),
    [dash.dependencies.Input('tabs', 'value')]
)
def update_tab(tab_name):
    """Update the tab content based on the selected tab"""
    if tab_name in TAB_FUNCTION_MAP:
        plot_function = TAB_FUNCTION_MAP[tab_name]
        fig = plot_function()
        return html.Div([
            dcc.Graph(figure=fig)
        ])
    return html.Div([html.H3("Tab content not found")])

if __name__ == '__main__':
    app.run_server(debug=True)