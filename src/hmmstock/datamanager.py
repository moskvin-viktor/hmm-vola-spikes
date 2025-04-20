import yfinance as yf
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def default_split(X, split_cfg=None):
    '''Splitting on train/test 
    TODO: Should be moved inside the class and handled by the class
    '''
    if split_cfg is None:
        split_cfg = {"train_ratio": 0.8, "validation_ratio": 0.2, "shuffle": False}

    train_ratio = split_cfg.get("train_ratio", 0.8)
    validation_ratio = split_cfg.get("validation_ratio", 0.2)
    shuffle = split_cfg.get("shuffle", False)

    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    train_end = int(train_ratio * n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:]

    return X[train_idx], X[val_idx]

class DataManager:
    def __init__(self, config):
        self.tickers = config["tickers"]
        self.period = config["period"]
        self.interval = config["interval"]
        self.volatility_windows = config["volatility_windows"]
        self.market_proxy = config["default_market_vola_proxy"]
        self.cache_file = Path(__file__).parent / "data_cache" / "cached_stock_data.pkl"

        self.all_tickers = list(set(self.tickers + [self.market_proxy]))
        self.data = self._fetch_data()

        self.returns = self._compute_daily_returns()
        self.mu, self.normalized_returns = self._normalize_returns()
        self.rolling_volatility = self._compute_rolling_volatility()

        self.market_returns = self.returns[self.market_proxy]  # used for injection only
        self.output = self._generate_output()

    # Inside DataManager
    def _fetch_data(self):
        '''Fetch stock data from Yahoo Finance and cache it locally.'''
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)
                if set(self.all_tickers).issubset(set(cached_data.columns)):
                    logger.info("Loaded cached stock data.")
                    return cached_data[self.all_tickers]

        logger.info("Fetching new stock data from Yahoo Finance...")
        data = yf.download(self.all_tickers, period=self.period, interval=self.interval)["Close"]
        self.cache_file.parent.mkdir(exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(data, f)
        return data

    def _compute_daily_returns(self):
        return np.log(self.data / self.data.shift(1)).dropna()

    def _normalize_returns(self):
        mu = self.returns.mean()
        return mu, self.returns - mu

    def _compute_rolling_volatility(self):
        vol_dict = {ticker: pd.DataFrame(index=self.returns.index) for ticker in self.tickers}
        for ticker in self.tickers:
            for window in self.volatility_windows:
                vol_dict[ticker][f"vol_{window}"] = self.returns[ticker].rolling(window).std()
        return vol_dict

    def _generate_output(self):
        output = {}
        for ticker in self.tickers:
            df = pd.DataFrame(index=self.returns.index)
            df["normalized_returns"] = self.normalized_returns[ticker]

            for window in self.volatility_windows:
                df[f"vol_{window}"] = self.rolling_volatility[ticker][f"vol_{window}"]

            # Inject market-level volatility proxy (VIX returns)
            df["market_vola"] = self.market_returns

            output[ticker] = df.dropna()
        return output

    def get_data(self):
        return self.output