import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmstock.path_manager import PathManager  # adjust import to your structure
import logging
class HMMResultVisualization:
    def __init__(self, model_name: str, ticker: str, config: dict, base_dir  = None):
        """
        Args:
            model_name: Name of the model (e.g., "HMMModel")
            ticker: Stock ticker symbol (e.g., "AAPL")
            config: Plotting configuration dictionary
            base_dir: Base results directory
        """
        self.model_name = model_name
        self.ticker = ticker.replace("^", "")  # sanitize filename
        self.config = config
        # If no base_dir is provided, go 1 level up from app/ directory
        self.base_dir = Path(self.config['base_results_path'])
        self.path_manager = PathManager(self.base_dir)  # Now correct
        self.df = self._load_regime_states()
        self.transition_dfs = self._load_transition_matrices()
        self.available_layers = self._detect_layers()

    def _load_regime_states(self) -> pd.DataFrame:
        csv_file = self.path_manager.get_ticket_csv_file(self.model_name, self.ticker, f"regime_states.csv")
        if not csv_file.exists():
            raise FileNotFoundError(f"Regime states file not found at {csv_file}")
        return pd.read_csv(csv_file, index_col=0, parse_dates=True)

    def _load_transition_matrices(self) -> dict:
        transition_dict = {}
        # Try to load multiple layers, fallback to single
        i = 0
        while True:
            csv_file = self.path_manager.get_transition_matrix(
                self.model_name, self.ticker, f"{self.ticker}_transition_matrix_layer{i}.csv"
            )
            if not csv_file.exists():
                # Maybe they saved a simple "transition_matrix.csv" without layers
                if i == 0:
                    fallback = self.path_manager.get_ticket_csv_file(
                        self.model_name, self.ticker, f"{self.ticker}_transition_matrix.csv"
                    )
                    if fallback.exists():
                        transition_dict[0] = pd.read_csv(fallback, index_col=0)
                break
            transition_dict[i] = pd.read_csv(csv_file, index_col=0)
            i += 1
        return transition_dict

    def _detect_layers(self):
        """Automatically detect available regime state layers from DataFrame columns."""
        layers = []
        for col in self.df.columns:
            if col.startswith('regime_layer'):
                layer_num = int(col.replace('regime_layer', ''))
                layers.append(layer_num)
            if col == 'top_level_state' or col == 'sub_level_state':
                layers.append(col)
        return sorted(layers)

    def get_state_column(self, layer=0) -> str:
        if layer not in self.available_layers:
            raise ValueError(f"Layer {layer} not found. Available layers: {self.available_layers}")
        if isinstance(layer, str):
            return layer
        return f'regime_layer{layer}'
    
    def plot_normalized_return_means(self, layer=0) -> go.Figure:
        """Plot the mean of normalized_returns grouped by regime states."""
        state_col = self.get_state_column(layer)

        if 'normalized_returns' not in self.df.columns:
            raise ValueError("'normalized_returns' column is missing in the data.")

        grouped_means = self.df.groupby(state_col)['normalized_returns'].mean().reset_index()

        fig = px.bar(
            grouped_means,
            x=state_col,
            y='normalized_returns',
            title=f'Normalized Returns Mean per Regime (Layer {layer})',
            labels={'normalized_returns': 'Mean Normalized Return', state_col: 'Regime'},
        )
        fig.update_layout(xaxis_title='Regime', yaxis_title='Mean Normalized Return')
        return fig

    def plot_feature_means(self, layer=0) -> go.Figure:
        """Plot mean values of features (excluding normalized_returns) grouped by regime states."""
        state_col = self.get_state_column(layer)

        # Exclude normalized_returns
        numeric_cols = self.df.select_dtypes(include='number').columns
        feature_cols = [col for col in numeric_cols if col not in {state_col, 'normalized_returns'}]

        grouped_means = self.df.groupby(state_col)[feature_cols].mean()

        fig = px.bar(
            grouped_means.transpose(),
            barmode=self.config['mean_barplot']['barmode'],
            title=f'Feature Means per Regime (Layer {layer}) — Excluding Normalized Returns',
            labels={'value': 'Mean Value', 'index': 'Feature'},
        )
        fig.update_layout(xaxis_title='Feature', yaxis_title='Mean')
        return fig

    def plot_transition_matrix(self, layer=0) -> go.Figure:
        # if not self.transition_dfs or layer not in self.transition_dfs:
        #     return go.Figure()
        logging.info(self.transition_dfs[layer])
        transition_df = self.transition_dfs[layer]

        fig = go.Figure(data=go.Heatmap(
            z=transition_df.values,
            x=transition_df.columns,
            y=transition_df.index,
            colorscale=self.config['transition_matrix']['colorscale'],
            hovertemplate='From %{y} → %{x}: %{z:.2f}<extra></extra>'
        ))
        fig.update_layout(title=f'HMM Transition Matrix (Layer {layer})')
        return fig

    def plot_time_series_by_regime(self, layer=0) -> go.Figure:
        state_col = self.get_state_column(layer)
        color_map = {
            state: px.colors.qualitative.Plotly[state % len(px.colors.qualitative.Plotly)]
            for state in sorted(self.df[state_col].unique())
        }

        fig = px.scatter(
            self.df.reset_index(), x=self.df.index.name or 'Date', y='normalized_returns',
            color=self.df[state_col].map(color_map),
            title=f'Normalized Returns Over Time by Regime (Layer {layer})'
        )
        fig.update_traces(mode='lines+markers')
        return fig

    def plot_observation_distributions_by_state(self, layer=0) -> go.Figure:
        state_col = self.get_state_column(layer)
        feature_cols = [col for col in self.df.columns if col not in ['Date'] and not col.startswith('regime_state')]

        fig = make_subplots(
            rows=1, cols=len(feature_cols),
            subplot_titles=feature_cols,
            shared_yaxes=True
        )

        for i, feature in enumerate(feature_cols):
            for state in sorted(self.df[state_col].unique()):
                subset = self.df[self.df[state_col] == state]

                fig.add_trace(
                    go.Histogram(
                        x=subset[feature],
                        name=f"State {state}",
                        opacity=0.6,
                        nbinsx=50,
                        marker_color=px.colors.qualitative.Plotly[state % len(px.colors.qualitative.Plotly)],
                        showlegend=(i == 0)  # only show legend once
                    ),
                    row=1, col=i+1
                )

        fig.update_layout(
            title_text=f'Observation Distributions by Regime State (Layer {layer})',
            barmode='overlay',
            height=500,
            width=300 * len(feature_cols),
        )
        fig.update_traces(opacity=0.5)
        return fig

    def plot_states_vs_volatility(self, layer=0) -> go.Figure:
        """
        Plot HMM regime states against volatility indicators for a specific layer.
        
        Args:
            layer: The regime layer to plot (default=0)
        """
        config = self.config['volatility_plot']
        state_col = self.get_state_column(layer)

        states = self.df[state_col]
        aligned_index = self.df.index

        fig = go.Figure()

        # Plot volatility lines and thresholds
        for vol_col, color in config['vol_colors'].items():
            if vol_col in self.df.columns:
                vol_series = self.df[vol_col]
                fig.add_trace(go.Scatter(
                    x=aligned_index,
                    y=vol_series,
                    mode='lines',
                    name=f'{vol_col} Vol',
                    line=dict(color=color),
                    yaxis='y2'
                ))

                threshold = vol_series.quantile(config['quantile'])
                fig.add_trace(go.Scatter(
                    x=[aligned_index[0], aligned_index[-1]],
                    y=[threshold, threshold],
                    mode='lines',
                    name=f'{vol_col} {int(config["quantile"]*100)}th %tile',
                    line=dict(color=color, dash='dash'),
                    yaxis='y2',
                    showlegend=True
                ))

        # Plot HMM states as colored markers
        fig.add_trace(go.Scatter(
            x=aligned_index,
            y=states,
            mode='markers',
            marker=dict(
                color=states,
                colorscale=config['colorscale'],
                size=config['marker_size'],
                colorbar=dict(title="State")
            ),
            name=f'HMM State (Layer {layer})',
            yaxis='y1'
        ))

        fig.update_layout(
            title=f'HMM States vs. Volatility (Layer {layer}) with {int(config["quantile"]*100)}th Percentile Lines',
            xaxis_title='Date',
            yaxis=dict(
                title='HMM State',
                side='left',
                showgrid=True,
                tickmode='array',
                tickvals=list(sorted(states.unique())),
                ticktext=[str(s) for s in sorted(states.unique())]
            ),
            yaxis2=dict(
                title='Volatility',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            showlegend=True
        )
        return fig
    
    def plot_regime_capture_of_vol_quantiles(self, layer=0):
        """
        Plot regime capture rates of high-volatility quantile events for a specific layer.
        
        Args:
            layer: The regime layer to plot (default=0)
        """
        config = self.config['volatility_plot']
        quantile = config['quantile']
        state_col = self.get_state_column(layer)
        capture_rates = {}

        for vol_col in config['vol_colors']:
            if vol_col in self.df.columns:
                threshold = self.df[vol_col].quantile(quantile)
                high_vol = self.df[self.df[vol_col] >= threshold]
                captured = high_vol[state_col].value_counts(normalize=True)
                capture_rates[vol_col] = captured

        df_capture = pd.DataFrame(capture_rates).fillna(0)

        fig = px.bar(
            df_capture,
            barmode='stack',
            title=f'Regime Capture of Top {int(quantile*100)}% Volatility Events (Layer {layer})',
            labels={'value': 'Capture Rate', 'index': 'Regime State'}
        )
        fig.update_layout(
            xaxis_title='Regime State',
            yaxis_title='Proportion of High Volatility Captured'
        )
        return fig
    
    def plot_state_volatility_correlations(self, layer=0):
        """
        Plot correlation between volatility measures and regime state for a specific layer.
        
        Args:
            layer: The regime layer to plot (default=0)
        """
        correlations = {}
        state_col = self.get_state_column(layer)

        for vol_col in self.config['volatility_plot']['vol_colors']:
            if vol_col in self.df.columns:
                correlations[vol_col] = self.df[[vol_col, state_col]].corr().iloc[0, 1]

        fig = px.bar(
            x=list(correlations.keys()),
            y=list(correlations.values()),
            labels={'x': 'Volatility Measure', 'y': 'Correlation with Regime State'},
            title=f'Correlation Between Volatility and Regime State (Layer {layer})'
        )
        return fig
    
    def plot_regime_capture_of_vol_quantiles(self, layer=0):
        """
        Plot regime capture rates of high-volatility quantile events for a specific layer.
        
        Args:
            layer: The regime layer to plot (default=0)
        """
        config = self.config['volatility_plot']
        quantile = config['quantile']
        state_col = self.get_state_column(layer)
        capture_rates = {}

        for vol_col in config['vol_colors']:
            if vol_col in self.df.columns:
                threshold = self.df[vol_col].quantile(quantile)
                high_vol = self.df[self.df[vol_col] >= threshold]
                captured = high_vol[state_col].value_counts(normalize=True)
                capture_rates[vol_col] = captured

        df_capture = pd.DataFrame(capture_rates).fillna(0)

        fig = px.bar(
            df_capture,
            barmode='stack',
            title=f'Regime Capture of Top {int(quantile*100)}% Volatility Events (Layer {layer})',
            labels={'value': 'Capture Rate', 'index': 'Regime State'}
        )
        fig.update_layout(xaxis_title='Regime State', yaxis_title='Proportion of High Volatility Captured')
        return fig

    def available_layer_indices(self):
        """Return the list of detected available layers."""
        return self.available_layers
    

