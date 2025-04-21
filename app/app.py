import dash
from dash import dcc, html, Input, Output
import plotly.io as pio
from hmmstock.plots.plot_results import HMMResultVisualization
from omegaconf import OmegaConf
import pandas as pd
import os

# Set Plotly theme to dark
pio.templates.default = "plotly_dark"

# Available tickers
AVAILABLE_TICKERS = ['AAPL', 'MSFT', 'GSPC', 'AMZN']

# Load config
config_plots = OmegaConf.load("../config/visualization_config.yaml")

# Dash app setup
app = dash.Dash(__name__)
server = app.server
app.title = "HMM Regime Viewer"

# Markdown descriptions for each plot
descriptions = {
    "means": "### Feature Means per Regime\nThis bar chart shows the average value of each feature within each HMM-identified regime. It helps interpret what characterizes each market regime.",
    "transition": "### HMM Transition Matrix\nThis heatmap visualizes the probability of switching from one regime to another. Diagonal values represent self-persistence; higher values indicate more stable regimes.",
    "returns": "### Normalized Returns by Regime\nThis line plot shows stock returns over time, color-coded by the regime assigned by the HMM. It reveals how regimes align with price movement.",
    "volatility": "### Regimes vs Volatility\nThis plot overlays regime states with selected volatility measures (e.g., VIX). Useful for understanding how regimes relate to market stress.",
    "quantiles": "### Regimes vs Volatility (VIX)\nThis plot overlays regime states with VIX quantiles. Useful for understanding how regimes relate to market stress.",
    "correlations": "### Regimes vs Volatility\nThis plot overlays regime states with selected volatility measures (e.g., VIX). Useful for understanding how regimes relate to market stress."
}


# Layout helper
def section(title, figure, description_md):
    return html.Div(style={
        "backgroundColor": "#2b2b2b",
        "padding": "1rem",
        "borderRadius": "12px",
        "marginBottom": "2rem",
        "boxShadow": "0 0 10px rgba(0,0,0,0.5)"
    }, children=[
        html.H2(title, style={"color": "#ffffff", "marginBottom": "1rem"}),
        dcc.Markdown(description_md, style={"color": "#dddddd", "marginBottom": "1rem"}),
        dcc.Graph(figure=figure, style={"height": "600px"})
    ])


# App layout
app.layout = html.Div(style={"backgroundColor": "#1e1e1e", "padding": "2rem"}, children=[
    html.H1("📉 HMM Stock Regime Dashboard", style={"textAlign": "center", "color": "#ffffff", "marginBottom": "2rem"}),

    html.Div([
        html.Label("Select Ticker:", style={"color": "white", "fontWeight": "bold", "marginRight": "1rem"}),
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{"label": t, "value": t} for t in AVAILABLE_TICKERS],
            value='AAPL',
            clearable=False,
            style={"width": "300px", "color": "#000"}
        )
    ], style={"display": "flex", "justifyContent": "center", "marginBottom": "2rem"}),

    html.Div(id='plots-container')
])


@app.callback(
    Output('plots-container', 'children'),
    Input('ticker-dropdown', 'value')
)
def update_plots(ticker):
    # Construct file paths
    safe_ticker = ticker.replace("^", "")  # for filename safety
    state_path = f"../results/csv/{safe_ticker}_regime_states.csv"
    trans_path = f"../results/csv/{safe_ticker}_transition_matrix.csv"

    # Check files exist
    if not os.path.exists(state_path) or not os.path.exists(trans_path):
        return html.Div(f"No data available for {ticker}", style={"color": "red"})

    # Load data
    df = pd.read_csv(state_path, index_col=0, parse_dates=True)
    trans = pd.read_csv(trans_path, index_col=0)

    vis = HMMResultVisualization(df, trans, config_plots)

    return [
        section("Feature Means per Regime", vis.plot_feature_means(), descriptions["means"]),
        section("HMM Transition Matrix", vis.plot_transition_matrix(), descriptions["transition"]),
        section("Normalized Returns by Regime", vis.plot_time_series_by_regime(), descriptions["returns"]),
        section("HMM States vs. Volatility", vis.plot_states_vs_volatility(), descriptions["volatility"]),
        section("HMM States vs. Volatility (VIX)", vis.plot_regime_capture_of_vol_quantiles(), descriptions["quantiles"]),
        section("HMM States Correlations", vis.plot_state_volatility_correlations(), descriptions["correlations"]),
    ]


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)