# Data Used 


The `DataManager` class handles fetching, caching, and preprocessing of historical stock data and market volatility proxies. It provides cleaned, normalized data structures suitable for downstream modeling with HMMs or other regime detection techniques.


# Data Description

This document outlines the structure and processing pipeline of the input data used in the HMM-based stock regime detection model.

## Data Sources

Historical market data is retrieved using the [Yahoo Finance API](https://finance.yahoo.com/) via the `yfinance` Python package. The following assets are used:

- `AAPL`: Apple Inc.
- `MSFT`: Microsoft Corporation
- `AMZN`: Amazon.com, Inc.
- `^GSPC`: S&P 500 Index
- `^VIX`: CBOE Volatility Index (used as the market volatility proxy)

We used daily normalized log-returns from "2020-01-01" to "2024-12-31".

### Configuration Summary

The file ```data.yaml``` allows to configure the data processing.

```yaml
tickers:
  - AAPL
  - MSFT
  - ^GSPC
  - ^VIX
  - AMZN
period: "5y"
interval: "1d"
volatility_windows: [5, 10, 20]
date_filter:
  start: "2020-01-01"
  end: "2024-12-31"
market_proxy_processing:
  default_market_vola_proxy: "^VIX"
  type: "zscore"
  smoothing_window: 5
volatility_processing:
  normalize: true
  method: "zscore"
```

### Data Processing Pipeline (Returns)

Each ticker (excluding `^VIX`) undergoes the following transformations:

### 1. Log Returns

Log returns \( r_t \) are computed as:

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

where \( P_t \) is the adjusted close price at time \( t \).

### 2. Normalized Returns

The mean return \( \mu \) over the time series is computed:

$$
\mu = \frac{1}{T} \sum_{t=1}^T r_t
$$

Then returns are mean-centered:

$$
\tilde{r}_t = r_t - \mu
$$

### 3. Rolling Volatility

For each rolling window size \( w ∈ {5, 10, 20} \), the standard deviation of returns is computed:

$$
\sigma_t^{(w)} = \sqrt{\frac{1}{w} \sum_{i=t-w+1}^t \left(\tilde{r}_i\right)^2}
$$

### 4. Volatility Normalization

Each rolling volatility time series is normalized using the z-score method:

$$
z_t = \frac{\sigma_t - \mu_{\sigma}}{\sigma_{\sigma}}
$$

where \( \mu_{\sigma} \) and \( \sigma_{\sigma} \) are the mean and standard deviation of the volatility time series.

### 5. Market Volatility Proxy Transformation

The market proxy \( V_t \) (i.e., `^VIX`) undergoes transformation using:

#### Smoothing:

A moving average over the specified window \( s \):

$$
\bar{V}_t = \frac{1}{s} \sum_{i=t-s+1}^t V_i
$$

#### Normalization:

The smoothed series is standardized:

$$
z_t^{(V)} = \frac{\bar{V}_t - \mu_V}{\sigma_V}
$$

where \( \mu_V \) and \( \sigma_V \) are the mean and standard deviation of the smoothed volatility proxy \( V_t \).

## Output Structure

For each ticker (excluding the proxy itself), the final output is a pandas `DataFrame` with the following columns:

- `normalized_returns`: Mean-centered log returns
- `vol_5`, `vol_10`, `vol_20`: Normalized rolling volatilities over 5, 10, and 20-day windows
- `market_vola`: Transformed market volatility proxy

All missing values due to windowing and lag operations are removed via `.dropna()`.

## Idea 

Idea is to clean the volatility as much as possible as well as use the `VIX` index as the proxy of the market volatility, this should in theory improve the fit of the model. 


# Reference

::: hmmstock.DataManager