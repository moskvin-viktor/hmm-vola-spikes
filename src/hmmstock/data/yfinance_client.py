from collections.abc import Callable

import pandas as pd
import yfinance as yf


class YFinanceClient:
    """Thin wrapper around yfinance's download API.

    The actual network call is injected as `fetch_fn`, so callers can
    substitute a fake in tests instead of hitting Yahoo Finance.
    """

    def __init__(self, fetch_fn: Callable[..., pd.DataFrame] = yf.download):
        self._fetch_fn = fetch_fn

    def get_close_prices(
        self, tickers: list[str], period: str, interval: str
    ) -> pd.DataFrame:
        """Fetch close prices for `tickers` as a wide DataFrame (columns = tickers)."""
        raw = self._fetch_fn(tickers, period=period, interval=interval)
        return raw["Close"]
