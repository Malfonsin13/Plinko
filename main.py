import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import base64
import io

# Load the data
def load_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            pitch_log_df = pitch_log_df.dropna(subset=['Situation'])
            pitch_log_df['Count'] = pitch_log_df['Situation'].str.extract(r'(\d-\d)', expand=False)
            pitch_log_df['Pitch Result'] = pitch_log_df['Pitch Result'].str.strip()
            pitch_log_df['Pitch Type'] = pitch_log_df['Pitch Type'].str.strip()
            pitch_log_df['Batter Hand'] = pitch_log_df['Batter'].str.extract(r'\((R|L)\)', expand=False)
            pitch_log_df = pitch_log_df.dropna(subset=['Batter Hand'])
            return df
    except Exception as e:
        print(e)
        return None

# Calculate swing percentages
def calculate_swing_percentage(df):
    swing_results = ['Foul', 'Swinging Strike', 'In play, no out', 'In play, out(s)', 'In play, run(s)', 'Foul Tip']
    swing_percentages = {}
    for count in df['Count'].unique():
        count_df = df[df['Count'] == count]
        total_pitches = len(count_df)
        swings = count_df[count_df['Pitch Result'].isin(swing_results)]
        swing_percent = len(swings) / total_pitches if total_pitches > 0 else 0
        swing_percentages[count] = swing_percent
    return swing_percentages

# Calculate average pitch velocity
def calculate_avg_velocity(df):
    velocities = {}
    for count in df['Count'].unique():
        count_df = df[df['Count'] == count]
        avg_velo = count_df['Velo'].mean() if not count_df['Velo'].isnull().all() else 0
        velocities[count] = avg_velo
    return velocities

# Process the data and generate the graph
def process_batter_hand(filtered_df, batter_hand, color_option):
    valid_counts = set(['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2'])
    strikeout_counts = set(['0-2', '1-2', '2-2', '3-2'])
    walk_counts = set(['3-0', '3-1', '3-2'])

    G = nx.MultiDiGraph()

    # Add transitions for each count
    transitions = {count: {next_count: {} for next_count in valid_counts} for count in valid_counts}
    for count in strikeout_counts:
        transitions[count]['K'] = {}
    for count in walk_counts:
        transitions[count]['BB'] = {}

    # Track counts for node metrics and transition information for hover
    count_totals = {count: 0 for count in valid_counts.union({'K', 'BB'})}
    pitch_info = {count: {} for count in valid_counts.union({'K', 'BB'})}

    # Edge colors and weights based on frequency
    for i in range(len(filtered_df) - 1):
        current_count = filtered_df.iloc[i]['Count']
        next_count = filtered_df.iloc[i + 1]['Count']
        pitch_type = filtered_df.iloc[i]['Pitch Type']
        pitch_result = filtered_df.iloc[i]['Pitch Result']

        # Handling for strikeouts (K) and walks (BB)
        if current_count in strikeout_counts and pitch_result in ['Swinging Strike', 'Called Strike', 'Foul Tip']:
            next_count = 'K'
        elif current_count in walk_counts and pitch_result == 'Ball':
            next_count = 'BB'

        if current_count in valid_counts and next_count in valid_counts.union({'K', 'BB'}):
            if pitch_type not in transitions[current_count][next_count]:
                transitions[current_count][next_count][pitch_type] = 0
            transitions[current_count][next_count][pitch_type] += 1
            G.add_edge(current_count, next_count, pitch_type=pitch_type, weight=transitions[current_count][next_count][pitch_type])

            # Add pitch information to hover
            if pitch_type in pitch_info[next_count]:
                pitch_info[next_count][pitch_type] += 1
            else:
                pitch_info[next_count][pitch_type] = 1

        # Update count totals
        if current_count in count_totals:
            count_totals[current_count] += 1
        if next_count in count_totals:
            count_totals[next_count] += 1

    # Compute swing percentages, velocities, or use count frequency based on color_option
    if color_option == 'swing_percentage':
        metric_data = calculate_swing_percentage(filtered_df)
    elif color_option == 'velocity':
        metric_data = calculate_avg_velocity(filtered_df)
    else:
        metric_data = count_totals  # Count frequency is the default

    # Create Plotly traces for nodes and edges
    edge_x, edge_y, edge_text = [], [], []
    positions = {
        '0-0': (0, 3), '1-0': (1, 2), '2-0': (2, 1), '3-0': (3, 0),
        '0-1': (-1, 2), '1-1': (0, 1), '2-1': (1, 0), '3-1': (2, -1),
        '0-2': (-2, 1), '1-2': (-1, 0), '2-2': (0, -1), '3-2': (1, -2),
        'K': (-1, -3),  # Strikeout position
        'BB': (2, -3)   # Walk position
    }

    # Edge trace with hover info
    for u, v, data in G.edges(data=True):
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f'{data["pitch_type"]}: {data["weight"]}')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )

    # Set node colors based on frequency or selected metric
    node_colors = []
    node_texts = []
    for node in G.nodes():
        metric_value = metric_data.get(node, 0)
        node_colors.append(metric_value)
        hover_pitch_info = "\n".join([f"{k}: {v}" for k, v in pitch_info[node].items()])

        if color_option == 'swing_percentage':
            node_texts.append(f'{node} (Swing% {metric_value:.1f})')
        elif color_option == 'velocity':
            node_texts.append(f'{node} (Velo {metric_value:.1f})')
        else:
            node_texts.append(f'{node} ({metric_value})')

    node_x, node_y = [], []
    for node in G.nodes():
        node_x.append(positions[node][0])
        node_y.append(positions[node][1])

    # Place node label above the node and handle hover info
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=50,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title=color_option,
                xanchor='left',
                titleside='right'
            )
        ),
        text=node_texts
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title={
                            'text': f'Count Progression for {batter_hand} Batters',
                            'y': 0.9,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        font=dict(
                            family="Arial, sans-serif",
                            size=16,
                            color="#7f7f7f"
                        ),
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        plot_bgcolor='white'
                    ))

    return fig

# Dash app setup
app = dash.Dash(__name__)

# Custom CSS styling for better design
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

app.layout = html.Div([
    html.H1('Interactive Plinko Chart', style={'textAlign': 'center', 'font-family': 'Arial'}),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select PitchLog.xlsx File')
            ]),
            style={
                'width': '98%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload')
    ]),
    html.Div([
        dcc.Dropdown(
            id='batter-hand-dropdown',
            options=[
                {'label': 'Right Handed', 'value': 'R'},
                {'label': 'Left Handed', 'value': 'L'},
                {'label': 'Both', 'value': 'B'}
            ],
            value='B',
            placeholder='Select Batter Hand',
            style={'width': '30%', 'display': 'inline-block'}
        ),
        dcc.Dropdown(
            id='color-option-dropdown',
            options=[
                {'label': 'Count Frequency', 'value': 'frequency'},
                {'label': 'Swing Percentage', 'value': 'swing_percentage'},
                {'label': 'Avg Pitch Velocity', 'value': 'velocity'}
            ],
            value='frequency',
            placeholder='Color Nodes Based On',
            style={'width': '30%', 'display': 'inline-block', 'marginLeft': '10px'}
        ),
        dcc.Dropdown(
            id='pitch-type-dropdown',
            placeholder='Select Pitch Type',
            style={'width': '30%', 'display': 'inline-block', 'marginLeft': '10px'}
        )
    ], style={'padding': '20px', 'textAlign': 'center'}),
    dcc.Graph(id='plinko-graph')
])


# Callback to update the graph
@app.callback(
    [Output('plinko-graph', 'figure'),
     Output('pitch-type-dropdown', 'options'),
     Output('pitch-type-dropdown', 'value')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('batter-hand-dropdown', 'value'),
     Input('color-option-dropdown', 'value'),
     Input('pitch-type-dropdown', 'value')]
)

def update_chart(contents, filename, batter_hand, color_option, pitch_type):
    if contents is None:
        # Return an empty figure and empty pitch type options
        return go.Figure(), [], None

    df = load_data(contents, filename)
    if df is None:
        return go.Figure(), [], None

    # Dynamically generate pitch type options based on the data
    pitch_types = [{'label': 'All', 'value': 'All'}] + \
                  [{'label': pt, 'value': pt} for pt in sorted(df['Pitch Type'].unique())]

    # Set default pitch type if none is selected
    if pitch_type is None:
        pitch_type = 'All'

    # Filter by batter hand
    if batter_hand != 'B':
        df = df[df['Batter Hand'] == batter_hand]

    # Filter by pitch type
    if pitch_type != 'All':
        df = df[df['Pitch Type'] == pitch_type]

    fig = process_batter_hand(df, batter_hand, color_option)
    return fig, pitch_types, pitch_type
