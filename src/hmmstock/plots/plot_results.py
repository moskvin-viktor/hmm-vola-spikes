
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class HMMResultVisualization:
    def __init__(self, df: pd.DataFrame, transition_df: pd.DataFrame, config: dict):
        self.df = df
        self.transition_df = transition_df
        self.config = config

    def plot_feature_means_and_transition_matrix(self):
        # === 1. Mean feature barplot per regime ===
        grouped_means = self.df.groupby('regime_state').mean()

        fig1 = px.bar(
            grouped_means.transpose(),
            barmode=self.config['mean_barplot']['barmode'],
            title='Feature Means per Regime',
            labels={'value': 'Mean Value', 'index': 'Feature'},
        )
        fig1.update_layout(xaxis_title='Feature', yaxis_title='Mean')
        fig1.show()

        # === 2. Transition matrix heatmap ===
        if self.transition_df is not None:
            fig2 = go.Figure(data=go.Heatmap(
                z=self.transition_df.values,
                x=self.transition_df.columns,
                y=self.transition_df.index,
                colorscale=self.config['transition_matrix']['colorscale'],
                hovertemplate='From %{y} → %{x}: %{z:.2f}<extra></extra>'
            ))
            fig2.update_layout(title='HMM Transition Matrix')
            fig2.show()

        # === 3. Time series with regime coloring ===
        color_map = {
            state: px.colors.qualitative.Plotly[state % len(px.colors.qualitative.Plotly)]
            for state in self.df['regime_state'].unique()
        }

        fig3 = px.scatter(
            self.df.reset_index(), x='Date', y='normalized_returns',
            color=self.df['regime_state'].map(color_map),
            title='Normalized Returns Over Time by Regime'
        )
        fig3.update_traces(mode='lines+markers')
        fig3.show()

        # === 4. Regime frequency plot ===
        regime_counts = self.df['regime_state'].value_counts().sort_index()
        fig4 = px.bar(
            x=regime_counts.index.astype(str), y=regime_counts.values,
            labels={'x': 'Regime', 'y': 'Count'},
            title='Regime Frequency'
        )
        fig4.show()

    def plot_states_vs_volatility(self):
        config = self.config['volatility_plot']
        states = self.df['regime_state']
        aligned_index = self.df.index

        fig = go.Figure()

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

        fig.add_trace(go.Scatter(
            x=aligned_index,
            y=states,
            mode='markers',
            marker=dict(color=states, colorscale=config['colorscale'], size=config['marker_size']),
            name='HMM State',
            yaxis='y1'
        ))

        fig.update_layout(
            title=f'HMM States vs. Volatility (with {int(config["quantile"]*100)}th Percentile Lines)',
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

        fig.show()