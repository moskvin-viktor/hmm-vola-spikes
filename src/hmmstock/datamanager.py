import yfinance as yf
import numpy as np
import pandas as pd
import os
import pickle

class StockReturnModel:
    def __init__(self, tickers, period="5y", interval="1d", volatility_windows=[5, 10, 20]):
        """
        :param tickers: List of stock tickers.
        :param period: Data period (default: "5y").
        :param interval: Data interval (default: "1d" for daily returns).
        :param volatility_windows: List of rolling window sizes for volatility (default: [5,10,20] days).
        """
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.volatility_windows = volatility_windows

        self.data = self._fetch_data()
        self.returns = self._compute_daily_returns()
        self.mu, self.normalized_returns = self._normalize_returns()
        self.rolling_volatility = self._compute_rolling_volatility()
        self.output = self._generate_output()

    def _fetch_data(self):
        """Fetch historical stock price data with caching."""
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(BASE_DIR, "data_cache", "cached_stock_data.pkl")

        # Load cached data if available
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                if set(self.tickers).issubset(set(cached_data.columns)):
                    print("Loaded cached stock data.")
                    return cached_data[self.tickers]

        print("Fetching new stock data...")
        data = yf.download(self.tickers, period=self.period, interval=self.interval)["Close"]

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

        return data

    def _compute_daily_returns(self):
        """Compute daily log returns for all tickers."""
        return np.log(self.data / self.data.shift(1)).dropna()

    def _normalize_returns(self):
        """Normalize returns to have zero mean."""
        mu = self.returns.mean()
        return mu, self.returns - mu

    def _compute_rolling_volatility(self):
        """Compute rolling volatility for multiple window sizes."""
        rolling_vol = {ticker: pd.DataFrame(index=self.returns.index) for ticker in self.tickers}
        for window in self.volatility_windows:
            for ticker in self.tickers:
                rolling_vol[ticker][f"vol_{window}"] = self.returns[ticker].rolling(window=window).std()
        return rolling_vol

    def _generate_output(self):
        """Return a dictionary with tickers as keys and DataFrames as values."""
        output = {}
        for ticker in self.tickers:
            df = pd.DataFrame(index=self.returns.index)
            df["normalized_returns"] = self.normalized_returns[ticker]
            for window in self.volatility_windows:
                df[f"vol_{window}"] = self.rolling_volatility[ticker][f"vol_{window}"]
            output[ticker] = df
        return output

    def get_data(self):
        """Return the final dictionary of processed data, keyed by ticker."""
        return self.output
    

def default_split(X, ratio=0.5):
    split_idx = int(len(X) * ratio)
    return X[:split_idx], X[split_idx:]