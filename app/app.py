import dash
from dash import dcc, html, Input, Output
import plotly.io as pio
from hmmstock.plots.plot_results import HMMResultVisualization
from omegaconf import OmegaConf
import os

# Set Plotly theme to dark
pio.templates.default = "plotly_dark"

# Available tickers and models
AVAILABLE_TICKERS = ['AAPL', 'MSFT', 'GSPC', 'AMZN']
AVAILABLE_MODELS = ['HMMModel', 'LayeredHMMModel', 'HierarchicalHMMModel']

# Load config
config_plots = OmegaConf.load("../config/visualization_config.yaml")

# Dash app setup
app = dash.Dash(__name__)
server = app.server
app.title = "HMM Regime Viewer"

# Markdown descriptions
descriptions = {
    "means_returns": "### Normalized Returns per Regime\nThis histogram shows the distribution of normalized returns for each regime identified by the HMM. It helps visualize how returns vary across different market states.",
    "anova_test": "### ANOVA Test on Returns\nThis table summarizes the results of an ANOVA test applied to the normalized returns across different regimes. It indicates whether there are statistically significant differences in returns between regimes.",
    "3ma": "### 3-Month Moving Average\nThis plot shows the 3-month moving average of volatility, color-coded by HMM regimes. It helps visualize how regimes align with longer-term trends.",
    "means": "### Feature Means per Regime\nThis bar chart shows the average value of each feature within each HMM-identified regime. It helps interpret what characterizes each market regime.",
    "transition": "### HMM Transition Matrix\nThis heatmap visualizes the probability of switching from one regime to another. Diagonal values represent self-persistence; higher values indicate more stable regimes.",
    "returns": "### Normalized Returns by Regime\nThis line plot shows stock returns over time, color-coded by the regime assigned by the HMM. It reveals how regimes align with price movement.",
    "volatility": "### Regimes vs Volatility\nThis plot overlays regime states with selected volatility measures (e.g., VIX). Useful for understanding how regimes relate to market stress.",
    "quantiles": "### Regimes vs Volatility (VIX)\nThis plot overlays regime states with VIX quantiles. Useful for understanding how regimes relate to market stress.",
    "correlations": "### Regimes vs Volatility\nThis plot overlays regime states with selected volatility measures (e.g., VIX). Useful for understanding how regimes relate to market stress.",
    "distribution": "### Regimes vs Volatility\nThis plot overlays regime states with selected volatility measures (e.g., VIX). Useful for understanding how regimes relate to market stress.", 
}

# Layout helper
def section(title, content, description_md) -> html.Div:
    is_figure = hasattr(content, "to_plotly_json")
    return html.Div(style={
        "backgroundColor": "#2b2b2b",
        "padding": "1rem",
        "borderRadius": "12px",
        "marginBottom": "2rem",
        "boxShadow": "0 0 10px rgba(0,0,0,0.5)"
    }, children=[
        html.H2(title, style={"color": "#ffffff", "marginBottom": "1rem"}),
        dcc.Markdown(description_md, style={"color": "#dddddd", "marginBottom": "1rem"}),
        dcc.Graph(figure=content, style={"height": "600px"}) if is_figure else
        html.Pre(content, style={"color": "#dddddd", "whiteSpace": "pre-wrap", "fontSize": "16px"})
    ])

# App layout
app.layout = html.Div(style={"backgroundColor": "#1e1e1e", "padding": "2rem"}, children=[
    html.H1("HMM Stock Regime Dashboard", style={"textAlign": "center", "color": "#ffffff", "marginBottom": "2rem"}),

    html.Div([
        html.Label("Select Model:", style={"color": "white", "fontWeight": "bold", "marginRight": "1rem"}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{"label": m, "value": m} for m in AVAILABLE_MODELS],
            value='HMMModel',
            clearable=False,
            style={"width": "300px", "color": "#000"}
        ),
        html.Label("Select Ticker:", style={"color": "white", "fontWeight": "bold", "marginLeft": "2rem", "marginRight": "1rem"}),
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{"label": t, "value": t} for t in AVAILABLE_TICKERS],
            value='AAPL',
            clearable=False,
            style={"width": "300px", "color": "#000"}
        )
    ], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "marginBottom": "2rem", "gap": "1rem"}),

    html.Label("Select Layer:", id="layer-label", style={"color": "white", "fontWeight": "bold", "marginLeft": "2rem", "marginRight": "1rem", "display": "none"}),
    dcc.Dropdown(
        id='layer-dropdown',
        options=[],
        value=None,
        clearable=False,
        style={"width": "200px", "color": "#000", "display": "none"}
    ),

    dcc.Tabs(
        id='tabs',
        value='plots',
        children=[
            dcc.Tab(label='Plots', value='plots', style={"backgroundColor": "#1e1e1e", "color": "#fff"}),
            dcc.Tab(label='Findings', value='findings', style={"backgroundColor": "#1e1e1e", "color": "#fff"}),
        ],
        style={"marginBottom": "2rem"},
    ),

    html.Div(id='tab-content')
])

@app.callback(
    Output('layer-dropdown', 'style'),
    Output('layer-label', 'style'),
    Output('layer-dropdown', 'options'),
    Output('layer-dropdown', 'value'),
    Input('model-dropdown', 'value')
)
def toggle_layer_dropdown(model_name):
    if model_name == 'LayeredHMMModel':
        dropdown_style = {"width": "200px", "color": "#000", "display": "block"}
        label_style = {"color": "white", "fontWeight": "bold", "marginLeft": "2rem", "marginRight": "1rem", "display": "block"}
        options = [{"label": f"Layer {i}", "value": i} for i in range(2)]
        value = 0
    elif model_name == 'HierarchicalHMMModel':
        dropdown_style = {"width": "200px", "color": "#000", "display": "block"}
        label_style = {"color": "white", "fontWeight": "bold", "marginLeft": "2rem", "marginRight": "1rem", "display": "block"}
        options = [
            {"label": "Top Level", "value": "top_level_state"},
            {"label": "Sub Level", "value": "sub_level_state"}
        ]
        value = "top_level_state"
    else:
        dropdown_style = {"width": "200px", "color": "#000", "display": "none"}
        label_style = {"color": "white", "fontWeight": "bold", "marginLeft": "2rem", "marginRight": "1rem", "display": "none"}
        options = []
        value = None

    return dropdown_style, label_style, options, value

def update_plots(selected_model, selected_ticker, selected_layer):
    vis = HMMResultVisualization(selected_model, selected_ticker, config_plots)
    #plot_normalized_return_means
    if selected_model in ['LayeredHMMModel', 'HierarchicalHMMModel']:
        return [
            section("Normalized Returns per Regime", vis.plot_normalized_return_means(layer=selected_layer), descriptions["means_returns"]),
            section("ANOVA Test Results", vis.report_anova_test_on_returns(layer=selected_layer), descriptions["anova_test"]),
            section("Moving Averages per Regime", vis.plot_vol_trend_bar_by_state(layer=selected_layer), descriptions["3ma"]),
            section("Feature Means per Regime", vis.plot_feature_means(layer=selected_layer), descriptions["means"]),
            section("HMM Transition Matrix", vis.plot_transition_matrix(layer=selected_layer), descriptions["transition"]),
            section("Normalized Returns by Regime", vis.plot_time_series_by_regime(layer=selected_layer), descriptions["returns"]),
            section("HMM States vs. Volatility", vis.plot_states_vs_volatility(layer=selected_layer), descriptions["volatility"]),
            section("HMM States vs. Volatility (VIX)", vis.plot_regime_capture_of_vol_quantiles(layer=selected_layer), descriptions["quantiles"]),
            section("HMM States Correlations", vis.plot_state_volatility_correlations(layer=selected_layer), descriptions["correlations"]),
        ]
    else:
        return [
            section("Normalized Returns per Regime", vis.plot_normalized_return_means(), descriptions["means_returns"]),
            section("ANOVA Test Results", vis.report_anova_test_on_returns(), descriptions["anova_test"]),
            section("Moving Averages per Regime", vis.plot_vol_trend_bar_by_state(), descriptions["3ma"]),
            section("Feature Means per Regime", vis.plot_feature_means(), descriptions["means"]),
            section("HMM Transition Matrix", vis.plot_transition_matrix(), descriptions["transition"]),
            section("Normalized Returns by Regime", vis.plot_time_series_by_regime(), descriptions["returns"]),
            section("HMM States vs. Volatility", vis.plot_states_vs_volatility(), descriptions["volatility"]),
            section("HMM States vs. Volatility (VIX)", vis.plot_regime_capture_of_vol_quantiles(), descriptions["quantiles"]),
            section("HMM States Correlations", vis.plot_state_volatility_correlations(), descriptions["correlations"]),
        ]

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('model-dropdown', 'value'),
    Input('ticker-dropdown', 'value'),
    Input('layer-dropdown', 'value'),
)
def render_tab(tab, selected_model, selected_ticker, selected_layer):
    if tab == 'plots':
        return update_plots(selected_model, selected_ticker, selected_layer)
    elif tab == 'findings':
        filename = f"markdown/findings_{selected_model}.md"
        try:
            with open(filename, "r") as f:
                markdown_text = f.read()
        except FileNotFoundError:
            markdown_text = f"### Findings file not found for {selected_model}"
        return html.Div([
            dcc.Markdown(markdown_text, style={"color": "#dddddd", "backgroundColor": "#2b2b2b", "padding": "2rem", "borderRadius": "12px"})
        ])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)