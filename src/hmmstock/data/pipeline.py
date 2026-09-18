import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, DateFilter
from .market_vola_proxy_calcs import MarketVolatilityProxyCalculations
from .price_cache import PriceCache
from .volatility_normalizer import VolatilityNormalizer
from .yfinance_client import YFinanceClient

logger = logging.getLogger(__name__)

DEFAULT_CACHE_FILE = Path(__file__).parent / "data_cache" / "cached_stock_data.pkl"


def _apply_date_filter(prices: pd.DataFrame, date_filter: DateFilter) -> pd.DataFrame:
    if date_filter.start:
        prices = prices[prices.index >= pd.to_datetime(date_filter.start)]
    if date_filter.end:
        prices = prices[prices.index <= pd.to_datetime(date_filter.end)]
    return prices


def fetch_prices(
    order: DataConfig, client: YFinanceClient, cache: PriceCache
) -> pd.DataFrame:
    """Close prices for `order.all_tickers`, from cache if available."""
    cached = cache.load()
    if cached is not None and set(order.all_tickers).issubset(set(cached.columns)):
        logger.info("Loaded cached stock data.")
        prices = cached[order.all_tickers]
    else:
        logger.info("Fetching new stock data from Yahoo Finance...")
        prices = client.get_close_prices(
            order.all_tickers, order.period, order.interval
        )
        cache.save(prices)
    return _apply_date_filter(prices, order.date_filter)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns of close prices."""
    return np.log(prices / prices.shift(1)).dropna()


def normalize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Mean-centers returns per ticker."""
    return returns - returns.mean()


def compute_rolling_volatility(
    returns: pd.DataFrame, order: DataConfig
) -> dict[str, pd.DataFrame]:
    """Rolling std of returns per ticker/window, normalized. Ticker -> DataFrame."""
    normalizer = VolatilityNormalizer(order.volatility_processing.method)
    volatility = {}
    for ticker in order.tickers:
        vol_df = pd.DataFrame(index=returns.index)
        for window in order.volatility_windows:
            raw_vol = returns[ticker].rolling(window).std()
            vol_df[f"vol_{window}"] = normalizer.normalize(raw_vol)
        volatility[ticker] = vol_df
    return volatility


def compute_market_proxy(prices: pd.DataFrame, order: DataConfig) -> pd.Series:
    """Processed market volatility proxy series (e.g. VIX z-score)."""
    proxy_conf = order.market_proxy_processing
    proxy_series = prices[proxy_conf.default_market_vola_proxy]
    return MarketVolatilityProxyCalculations(
        proxy_series, proxy_conf.model_dump()
    ).process()


def assemble_output(
    normalized_returns: pd.DataFrame,
    rolling_volatility: dict[str, pd.DataFrame],
    market_proxy: pd.Series,
    order: DataConfig,
) -> dict[str, pd.DataFrame]:
    """Combines normalized returns, volatility, and market proxy per ticker."""
    output = {}
    for ticker in order.tickers:
        df = pd.DataFrame(index=normalized_returns.index)
        df["normalized_returns"] = normalized_returns[ticker]
        for window in order.volatility_windows:
            df[f"vol_{window}"] = rolling_volatility[ticker][f"vol_{window}"]
        df["market_vola"] = market_proxy
        output[ticker] = df.dropna()
    return output


def run_pipeline(
    order: DataConfig,
    client: YFinanceClient | None = None,
    cache: PriceCache | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetches and prepares per-ticker training data for `order`.

    Stages: fetch_prices -> compute_returns -> normalize_returns ->
    compute_rolling_volatility / compute_market_proxy -> assemble_output.
    Each stage is a pure function of its inputs; only fetch_prices touches
    the network/disk, through the injected client/cache.
    """
    client = client or YFinanceClient()
    cache = cache or PriceCache(DEFAULT_CACHE_FILE)

    prices = fetch_prices(order, client, cache)
    returns = compute_returns(prices)
    normalized_returns = normalize_returns(returns)
    rolling_volatility = compute_rolling_volatility(returns, order)
    market_proxy = compute_market_proxy(prices, order)

    return assemble_output(normalized_returns, rolling_volatility, market_proxy, order)
