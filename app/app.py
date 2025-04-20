import dash
from dash import dcc, html
import plotly.io as pio
from hmmstock.plots.plot_results import HMMResultVisualization
from omegaconf import OmegaConf
import pandas as pd


# Set Plotly theme to dark
pio.templates.default = "plotly_dark"

# Load config and model output
config_plots = OmegaConf.load("../config/visualization_config.yaml")
model_output_path = "../results/csv/AAPL_regime_states.csv"
transition_matrix_path = "../results/csv/AAPL_transition_matrix.csv"
# Set Plotly theme to dark


# Load data
state_labeled_df = pd.read_csv(model_output_path, index_col=0, parse_dates=True)
transition_matrix = pd.read_csv(transition_matrix_path, index_col=0)

# Create visualizer
vis = HMMResultVisualization(state_labeled_df, transition_matrix, config_plots)

# Layout helper
def section(title, figure):
    return html.Div(style={
        "backgroundColor": "#2b2b2b",
        "padding": "1rem",
        "borderRadius": "12px",
        "marginBottom": "2rem",
        "boxShadow": "0 0 10px rgba(0,0,0,0.5)"
    }, children=[
        html.H2(title, style={"color": "#ffffff", "marginBottom": "1rem"}),
        dcc.Graph(figure=figure, style={"height": "600px"})
    ])

# Create Dash app
app = dash.Dash(__name__)
server = app.server
app.title = "HMM Regime Viewer"

app.layout = html.Div(style={"backgroundColor": "#1e1e1e", "padding": "2rem"}, children=[
    html.H1("HMM Stock Regime Visualization", style={"textAlign": "center", "color": "#ffffff", "marginBottom": "3rem"}),

    section("Feature Means per Regime", vis.plot_feature_means()),
    section("HMM Transition Matrix", vis.plot_transition_matrix()),
    section("Normalized Returns by Regime", vis.plot_time_series_by_regime()),
    section("HMM States vs. Volatility", vis.plot_states_vs_volatility())
])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)