import yfinance as yf
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
import logging

from .market_vola_proxy_calcs import MarketVolatilityProxyCalculations
from .volatility_normalizer import VolatilityNormalizer
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
    '''Class to manage stock data fetching, processing, and caching.'''
    def __init__(self, config):
        self.config = config
        self._load_config()

        self.data = self._fetch_data()
        self.returns = self._compute_daily_returns()
        self.mu, self.normalized_returns = self._normalize_returns()
        self.rolling_volatility = self._compute_rolling_volatility()
        self.output = self._generate_output()

    def _load_config(self):
        self.tickers = self.config["tickers"]
        self.period = self.config["period"]
        self.interval = self.config["interval"]
        self.volatility_windows = self.config["volatility_windows"]
        self.market_proxy_conf = self.config["market_proxy_processing"]
        self.market_proxy = self.market_proxy_conf["default_market_vola_proxy"]
        self.vola_norm_method = self.config.get("volatility_processing", {}).get("normalize_method", "zscore")
        self.date_range = self.config.get("date_filter", {"start": None, "end": None})
        self.cache_file = Path(__file__).parent / "data_cache" / "cached_stock_data.pkl"
        self.all_tickers = list(set(self.tickers + [self.market_proxy]))

    def _fetch_data(self):
        '''Fetch stock data from Yahoo Finance and cache it locally.'''
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                cached_data = pickle.load(f)
                if set(self.all_tickers).issubset(set(cached_data.columns)):
                    logger.info("Loaded cached stock data.")
                    return self._apply_date_filter(cached_data[self.all_tickers])

        logger.info("Fetching new stock data from Yahoo Finance...")
        data = yf.download(self.all_tickers, period=self.period, interval=self.interval)["Close"]
        self.cache_file.parent.mkdir(exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(data, f)
        return self._apply_date_filter(data)

    def _apply_date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        start = self.date_range.get("start")
        end = self.date_range.get("end")
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def _compute_daily_returns(self):
        return np.log(self.data / self.data.shift(1)).dropna()

    def _normalize_returns(self):
        mu = self.returns.mean()
        return mu, self.returns - mu

    def _compute_rolling_volatility(self):
        vol_dict = {ticker: pd.DataFrame(index=self.returns.index) for ticker in self.tickers}
        for ticker in self.tickers:
            for window in self.volatility_windows:
                vol = self.returns[ticker].rolling(window).std()
                norm_vol = VolatilityNormalizer(self.vola_norm_method).normalize(vol)
                vol_dict[ticker][f"vol_{window}"] = norm_vol
        return vol_dict

    def _generate_output(self):
        proxy_series = self.data[self.market_proxy]
        processed_proxy = MarketVolatilityProxyCalculations(
            proxy_series, self.market_proxy_conf
        ).process()

        output = {}
        for ticker in self.tickers:
            df = pd.DataFrame(index=self.returns.index)
            df["normalized_returns"] = self.normalized_returns[ticker]

            for window in self.volatility_windows:
                df[f"vol_{window}"] = self.rolling_volatility[ticker][f"vol_{window}"]

            df["market_vola"] = processed_proxy
            output[ticker] = df.dropna()

        return output

    def get_data(self):
        return self.output