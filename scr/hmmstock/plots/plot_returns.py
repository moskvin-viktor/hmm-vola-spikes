import plotly.graph_objects as go
import numpy as np

def plot_stock_analysis(data_dict, ticker, threshold_percentile=95):
    """
    Plots normalized returns and rolling volatilities for a given stock ticker.
    
    :param data_dict: Dictionary where keys are tickers, and values are DataFrames with returns and volatilities.
    :param ticker: Stock ticker symbol (string).
    :param threshold_percentile: Percentile threshold for defining volatility spikes (default: 95th percentile).
    """
    if ticker not in data_dict:
        print(f"Ticker {ticker} not found in the dataset.")
        return

    df = data_dict[ticker]

    if "normalized_returns" not in df.columns or not any(col.startswith("vol_") for col in df.columns):
        print(f"Data for {ticker} does not contain returns or volatility measures.")
        return

    fig1 = go.Figure()
    fig2 = go.Figure()

    # Plot Normalized Returns Histogram
    fig1.add_trace(go.Histogram(
        x=df["normalized_returns"],
        name="Histogram of Returns",
        opacity=0.75,
        marker=dict(color="blue"),
        nbinsx=50
    ))

    # Plot Normalized Returns Time Series
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df["normalized_returns"],
        mode='lines',
        name='Normalized Returns',
        line=dict(color='blue')
    ))

    # Plot All Rolling Volatilities
    spike_dates = []
    spike_values = []

    for col in df.columns:
        if col.startswith("vol_"):
            # Compute threshold for spikes
            threshold = np.percentile(df[col].dropna(), threshold_percentile)

            # Plot rolling volatility
            fig2.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(width=1)
            ))

            # Identify and highlight spikes
            spikes = df[df[col] > threshold]
            spike_dates.extend(spikes.index)
            spike_values.extend(spikes[col])

    # Highlight volatility spikes
    fig2.add_trace(go.Scatter(
        x=spike_dates,
        y=spike_values,
        mode='markers',
        name="Volatility Spikes",
        marker=dict(color='red', size=6, symbol='triangle-up')
    ))

    fig1.update_layout(
        title=f"Histogram of Normalized Returns for {ticker}",
        xaxis_title="Returns",
        yaxis_title="Frequency",
        showlegend=True
    )

    fig2.update_layout(
        title=f"Normalized Returns & Rolling Volatility for {ticker}",
        xaxis_title="Time",
        yaxis_title="Value",
        showlegend=True
    )

    fig1.show()
    fig2.show()
