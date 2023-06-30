import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import colorlover
import plotly.io as pio
import plotly.express.colors as colors
from base64 import b64encode
import io

# create a function to clear outlier data in the dataset
def cleanColumn(data, columns, thr=2):
    describe = df.describe()
    column_desc = describe[columns]

    q3 = column_desc[6]
    q1 = column_desc[4]
    IQR = q3 - q1

    top_limit_clm = q3 + thr * IQR
    bottom_limit_clm = q1 - thr * IQR

    filter_clm_bottom = bottom_limit_clm < data[columns]
    filter_clm_top = data[columns] < top_limit_clm

    filters = filter_clm_bottom & filter_clm_top

    data = data[filters]

    return data

# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")

# Apply the function to the dataset
for i in df.columns:
    df = cleanColumn(df, i)

# Separate features and target variable
X = df.drop("Outcome", axis=1)
y = df['Outcome']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Calculate the probability of having diabetes for each cluster
cluster_probabilities = df.groupby('Cluster')['Outcome'].mean().sort_values()

# Map cluster labels to cluster names according to the probability of having diabetes
cluster_names = {
    cluster: f"Cluster {cluster} - {round(probability * 100, 2)}% probability to have diabetes"
    for cluster, probability in enumerate(cluster_probabilities)
}
df['Cluster'] = df['Cluster'].map(cluster_names)
color_palette = colors.qualitative.Pastel

# Train a random forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Calculate feature importance using permutation importance
result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
feature_importance = pd.Series(result.importances_mean, index=X.columns)
sorted_importance = feature_importance.sort_values(ascending=False)

# Create the correlation matrix heatmap figure
corr_matrix = df.corr()
fig_heatmap = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu')

# Add correlation values to the heatmap
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        if corr_matrix.iloc[i, j] == 1:
            text_color = 'white'
        else:
            text_color = 'black'
        fig_heatmap.add_annotation(
            x=i, y=j,
            text=str(round(corr_matrix.iloc[i, j], 2)),
            showarrow=False,
            font=dict(color=text_color, size=10)
        )

# Create the feature importance figure
fig_importance = px.bar(
    x=sorted_importance.values,
    y=sorted_importance.index,
    orientation='h',
    color=sorted_importance.values,
    color_continuous_scale='RdBu',
    labels={'x': 'Feature Importance', 'y': 'Feature'},
    title='Feature Importance',
    template='plotly_white'
)

# Define the Dash app
app = dash.Dash(__name__)

buffer = io.StringIO()

# Define the app layout
app.layout = html.Div([
    html.H1("Diabetes Dataset Dashboard", style={'color': 'black'}),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            html.H2("Correlation Matrix Heatmap", style={'color': 'black'}),
            dcc.Graph(id='heatmap', figure=fig_heatmap)
        ]),

        html.Div(className='six columns', children=[
            html.H2("Feature Importance", style={'color': 'black'}),
            dcc.Graph(id='feature-importance', figure=fig_importance)
        ])
    ]),

    html.H2("Distribution Plot", style={'color': 'black'}),
    html.Label("Select a feature:", style={'color': 'black'}),
    dcc.Dropdown(
        id='distribution-dropdown',
        options=[{'label': col, 'value': col} for col in X.columns],
        value=X.columns[0]
    ),
    dcc.Graph(id='distribution-plot', figure={}),  # Updated to provide an initial empty figure

    html.H2("Combination of Age, BMI and Glucose", style={'color': 'black'}),
    dcc.Graph(
        id='clustering-plot',
        figure=px.scatter(
            df,
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            color='Cluster',
            hover_name='Cluster',
            hover_data=['Age', 'BMI', 'Glucose'],
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            title='Clustering - Age, BMI, Glucose after t-SNE',
            template='plotly_white'
        ).update_traces(marker=dict(size=8, line=dict(width=0.5, color='black')), showlegend=True)
    )
])

# Define callback for distribution plot
@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('distribution-dropdown', 'value')]
)
def update_distribution_plot(feature):
    fig = go.Figure()

    # Define color scheme
    colors = colorlover.scales['9']['seq']['Blues']

    # Add histogram trace for diabetic patients
    fig.add_trace(go.Histogram(
        x=df[df['Outcome'] == 1][feature],
        nbinsx=20,
        marker_color=colors[3],
        opacity=0.7,
        name='Diabetic',
        showlegend=True,
        hovertemplate='<b>Diabetic</b>'+'<br> %{x} '+ f'{feature}' +'<br> Number of patients: %{y}<extra></extra>'
    ))

    # Add histogram trace for non-diabetic patients
    fig.add_trace(go.Histogram(
        x=df[df['Outcome'] == 0][feature],
        nbinsx=20,
        marker_color=colors[6],
        opacity=0.7,
        name='Non-Diabetic',
        showlegend=True,
        hovertemplate='<b>Non-Diabetic</b>'+'<br> %{x} '+ f'{feature}' +'<br> Number of patients: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Distribution Plot for <b>{feature}</b>",
        xaxis_title=feature,
        yaxis_title='Number of patients',
        showlegend=True,
        legend=dict(x=1, y=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Arial',
        font_size=12,
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(color='black')
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
