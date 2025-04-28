import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots


class HMMResultVisualization:
    def __init__(self, df: pd.DataFrame, transition_dfs: dict, config: dict):
        """
        Args:
            df: DataFrame with observations and multiple regime state columns, e.g., 'regime_state_layer0', 'regime_state_layer1', etc.
            transition_dfs: Dict of {layer_index: transition_matrix_df}
            config: Plotting configuration
        """
        self.df = df
        self.transition_dfs = transition_dfs
        self.config = config
        self.available_layers = self._detect_layers()

    def _detect_layers(self):
        """Automatically detect available regime state layers from DataFrame columns."""
        layers = []
        for col in self.df.columns:
            if col.startswith('regime_state_layer'):
                layer_num = int(col.replace('regime_state_layer', ''))
                layers.append(layer_num)
        return sorted(layers)

    def get_state_column(self, layer=0):
        if layer not in self.available_layers:
            raise ValueError(f"Layer {layer} not found. Available layers: {self.available_layers}")
        return f'regime_state_layer{layer}'

    def plot_feature_means(self, layer=0):
        state_col = self.get_state_column(layer)
        grouped_means = self.df.groupby(state_col).mean()

        fig = px.bar(
            grouped_means.transpose(),
            barmode=self.config['mean_barplot']['barmode'],
            title=f'Feature Means per Regime (Layer {layer})',
            labels={'value': 'Mean Value', 'index': 'Feature'},
        )
        fig.update_layout(xaxis_title='Feature', yaxis_title='Mean')
        return fig

    def plot_transition_matrix(self, layer=0):
        if self.transition_dfs is None or layer not in self.transition_dfs:
            return go.Figure()

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

    def plot_time_series_by_regime(self, layer=0):
        state_col = self.get_state_column(layer)
        color_map = {
            state: px.colors.qualitative.Plotly[state % len(px.colors.qualitative.Plotly)]
            for state in sorted(self.df[state_col].unique())
        }

        fig = px.scatter(
            self.df.reset_index(), x='Date', y='normalized_returns',
            color=self.df[state_col].map(color_map),
            title=f'Normalized Returns Over Time by Regime (Layer {layer})'
        )
        fig.update_traces(mode='lines+markers')
        return fig

    def plot_observation_distributions_by_state(self, layer=0):
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

    def available_layer_indices(self):
        """Return the list of detected available layers."""
        return self.available_layers