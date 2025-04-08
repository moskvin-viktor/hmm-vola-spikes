
import plotly.graph_objects as go


def plot_states_vs_volatility(df, states, quantile=0.95):
    """Plot HMM states alongside volatility measures and quantile threshold lines."""
    if states is None or len(states) == 0:
        print("No states available.")
        return

    aligned_index = df.index[-len(states):]  # Align with states
    fig = go.Figure()

    # Colors for vol traces and quantile lines
    vol_config = {
        'vol_5': {'color': 'blue'},
        'vol_10': {'color': 'green'},
        'vol_20': {'color': 'red'}
    }

    # Add volatility lines + quantile thresholds if available
    for vol_col, cfg in vol_config.items():
        if vol_col in df.columns:
            vol_series = df.loc[aligned_index, vol_col]
            fig.add_trace(go.Scatter(
                x=aligned_index,
                y=vol_series,
                mode='lines',
                name=f'{vol_col} Vol',
                line=dict(color=cfg['color']),
                yaxis='y2'
            ))

            # Quantile threshold line
            threshold = vol_series.quantile(quantile)
            fig.add_trace(go.Scatter(
                x=[aligned_index[0], aligned_index[-1]],
                y=[threshold, threshold],
                mode='lines',
                name=f'{vol_col} {int(quantile*100)}th %tile',
                line=dict(color=cfg['color'], dash='dash'),
                yaxis='y2',
                showlegend=True
            ))

    # Add HMM state markers
    fig.add_trace(go.Scatter(
        x=aligned_index,
        y=states,
        mode='markers',
        marker=dict(color=states, colorscale='viridis', size=8),
        name='HMM State',
        yaxis='y1'
    ))

    fig.update_layout(
        title=f'HMM States vs. Volatility (with {int(quantile*100)}th Percentile Lines)',
        xaxis_title='Date',
        yaxis=dict(
            title='HMM State',
            side='left',
            showgrid=True,
            tickmode='array',
            tickvals=list(set(states)),
            ticktext=[str(s) for s in sorted(set(states))]
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
