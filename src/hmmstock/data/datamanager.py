import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from typing import Any

from .market_vola_proxy_calcs import MarketVolatilityProxyCalculations
from .volatility_normalizer import VolatilityNormalizer

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def default_split(X: Any, split_cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Splitting on train/test
    TODO: Should be moved inside the class and handled by the class
    """
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
    """
    Class to manage stock data fetching, preprocessing, and caching.

    This class retrieves data from Yahoo Finance, processes it through a chained pipeline
    of transformations (date filtering, log returns, normalization, volatility calculation),
    and integrates a processed market volatility proxy. Configuration parameters are stored
    as private attributes.

    Attributes:
        _config (dict): Configuration object loaded from YAML/OmegaConf.
        output (pd.DataFrame): Final structured data after pipeline processing.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the DataManager with configuration and orchestrate the data processing pipeline.

        Args:
            config (dict): Contains tickers, _period, _interval, volatility settings,
                           market proxy config, and optional date filter.
        """
        self._config = config
        self._load_config()

        # Fetch raw data
        raw_data = self._fetch_data()

        # Process market volatility proxy separately
        proxy_series = raw_data[self._market_proxy]
        processed_proxy = MarketVolatilityProxyCalculations(
            proxy_series, self._market_proxy_conf
        ).process()

        # Chain the main pipeline methods, starting with filtering
        processed_pipeline_data = (
            raw_data[self._tickers]  # Start with the relevant tickers from raw_data
            .pipe(self._apply_date_filter)  # Integrate date filtering into the pipeline
            .pipe(self._compute_daily_returns)
            .pipe(self._normalize_returns)
            .pipe(self._compute_rolling_volatility)
            .pipe(self._ensure_float64_types)  # Ensure float64 types in the pipeline
        )

        # Merge processed market proxy into the final DataFrame
        final_output_df = processed_pipeline_data.copy()
        final_output_df["market_vola"] = processed_proxy.reindex(final_output_df.index)

        self.output = final_output_df.dropna()

    def _load_config(self) -> None:
        """Parses and stores configuration parameters as private attributes."""
        self._tickers = self._config["tickers"]
        self._period = self._config["period"]
        self._interval = self._config["interval"]
        self._volatility_windows = self._config["volatility_windows"]
        self._market_proxy_conf = self._config["market_proxy_processing"]
        self._market_proxy = self._market_proxy_conf["default_market_vola_proxy"]
        self._vola_norm_method = self._config.get("volatility_processing", {}).get(
            "normalize_method", "zscore"
        )
        self._date_range = self._config.get("date_filter", {"start": None, "end": None})
        self._cache_file = (
            Path(__file__).parent / "data_cache" / "cached_stock_data.pkl"
        )
        self._all_tickers = list(set(self._tickers + [self._market_proxy]))

    def _fetch_data(self) -> pd.DataFrame:
        """
        Orchestrates data loading, attempting to load from cache first, then downloading if necessary.

        Returns:
            pd.DataFrame: Close price data for tickers.
        """
        cached_data = self._load_cached_data()
        if cached_data is not None:
            return pd.DataFrame(cached_data["Close"])

        logger.info("Fetching new stock data from Yahoo Finance...")
        raw_data = yf.download(
            self._all_tickers, period=self._period, interval=self._interval
        )
        if raw_data is None or raw_data.empty:
            raise ValueError("No data has been collected and no cache found! Aborting!")
        data = raw_data["Close"]

        self._save_data_to_cache(raw_data)  # Save raw_data, not just "Close" prices

        return pd.DataFrame(data)

    def _apply_date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies start and end date filters to the DataFrame using DatetimeIndex slicing.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        start = self._date_range.get("start")
        end = self._date_range.get("end")

        # Ensure the index is a DatetimeIndex for slicing
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Use direct slicing for DatetimeIndex
        if start and end:
            return df.loc[start:end]
        elif start:
            return df.loc[start:]
        elif end:
            return df.loc[:end]
        return df

    def _compute_daily_returns(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes daily log returns from the input DataFrame.

        Args:
            data_df (pd.DataFrame): DataFrame containing stock close prices.

        Returns:
            pd.DataFrame: Log returns.
        """
        returns_raw = data_df / data_df.shift(1)
        log_returns = returns_raw.transform(np.log)
        return log_returns.dropna()

    def _normalize_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean-centers returns.

        Args:
            returns_df (pd.DataFrame): DataFrame containing daily log returns.

        Returns:
            pd.DataFrame: Mean-centered returns.
        """
        if returns_df.empty:
            raise ValueError("No data has been collected! Aborting!")

        mu = returns_df.mean()  # Calculate mu locally
        return returns_df.sub(mu)  # Use mu immediately

    def _compute_rolling_volatility(
        self, normalized_returns_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes and normalizes rolling volatility and adds it to the input DataFrame.

        Args:
            normalized_returns_df (pd.DataFrame): DataFrame with normalized returns.

        Returns:
            pd.DataFrame: DataFrame with normalized returns and rolling volatility columns.
        """
        if normalized_returns_df.empty:
            logger.warning(
                "Normalized returns DataFrame is empty. Cannot compute rolling volatility. Returning empty DataFrame."
            )
            return pd.DataFrame()

        output_df = normalized_returns_df.copy()  # Reintroduce copy

        for ticker in self._tickers:
            for window in self._volatility_windows:
                # Ensure the ticker column exists before computing volatility
                if ticker in output_df.columns:
                    vol = output_df[ticker].rolling(window).std()
                    norm_vol = VolatilityNormalizer(self._vola_norm_method).normalize(
                        vol
                    )
                    output_df[f"{ticker}_vol_{window}"] = norm_vol
                else:
                    logger.warning(
                        f"Ticker '{ticker}' not found in normalized returns DataFrame. Skipping volatility calculation for this ticker."
                    )
        return output_df.dropna()  # Reintroduce dropna

    def _ensure_float64_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures all numerical columns in the DataFrame are of float64 type.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with numerical columns cast to float64.
        """
        numerical_cols = df.select_dtypes(include=np.number).columns
        df[numerical_cols] = df[numerical_cols].astype(np.float64)
        return df

    def get_data(self) -> pd.DataFrame:
        """
        Returns the processed dataset as a single DataFrame.

        Returns:
            pd.DataFrame: DataFrame with returns, volatility, and proxy for all tickers.
        """
        return self.output

    def _load_cached_data(self) -> pd.DataFrame | None:
        """
        Attempts to load data from the cache.

        Returns:
            pd.DataFrame | None: Cached data if successful and valid, otherwise None.
        """
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    if set(self._all_tickers).issubset(
                        set(cached_data.columns.get_level_values(1))
                    ):
                        logger.info("Loaded cached stock data.")
                        return cached_data
                    else:
                        logger.info(
                            "Cached data does not contain all required tickers. Fetching new data."
                        )
            except (
                FileNotFoundError,
                EOFError,
                Exception,
            ) as e:  # Catch specific errors
                logger.warning(f"Error loading cached data: {e}. Fetching new data.")

    def _save_data_to_cache(self, data: pd.DataFrame) -> None:
        """
        Saves the provided DataFrame to the cache.

        Args:
            data (pd.DataFrame): The DataFrame to save.
        """
        self._cache_file.parent.mkdir(exist_ok=True)
        with open(self._cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info("Data saved to cache.")
