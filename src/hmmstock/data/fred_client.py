from collections.abc import Callable
from io import StringIO

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _fetch_series(series_id: str) -> pd.DataFrame:
    response = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), na_values=".")
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


class FredClient:
    """Thin wrapper around FRED's public CSV download endpoint
    (https://fred.stlouisfed.org/graph/fredgraph.csv) -- no API key
    required, unlike FRED's JSON API. The fetch call is injected as
    `fetch_series`, so tests can substitute a fake instead of hitting
    the network.

    Matches YFinanceClient.get_close_prices's shape, so it's a drop-in
    `client` for the data pipeline: `run_pipeline(order, client=FredClient())`.
    Pass FRED series IDs as tickers instead of Yahoo tickers (e.g.
    "VIXCLS" for VIX close, "DGS10" for the 10-year Treasury yield,
    "SP500" for the S&P 500) to bring in macro series yfinance doesn't
    have.
    """

    def __init__(self, fetch_series: Callable[[str], pd.DataFrame] = _fetch_series):
        self._fetch_series = fetch_series

    def get_close_prices(
        self, tickers: list[str], period: str, interval: str
    ) -> pd.DataFrame:
        """Fetch each series and align into one wide DataFrame (columns =
        tickers, outer-joined on date). period/interval are unused: FRED
        series come back at their own native frequency, and the pipeline's
        date_filter narrows the range after fetching, same as for
        YFinanceClient. Series on different native frequencies (e.g. a
        daily one mixed with a monthly one) will have NaN gaps -- downstream
        compute_returns()'s dropna() will thin those rows out.
        """
        frames = [self._fetch_series(series_id) for series_id in tickers]
        return pd.concat(frames, axis=1, join="outer", sort=False).sort_index()
