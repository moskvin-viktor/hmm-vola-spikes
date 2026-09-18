import pandas as pd

from hmmstock.data.yfinance_client import YFinanceClient


def _fake_download(tickers, period, interval):
    dates = pd.date_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product(
        [["Close", "Open"], tickers], names=["Price", "Ticker"]
    )
    data = [[i + j for j in range(len(columns))] for i in range(3)]
    return pd.DataFrame(data, index=dates, columns=columns)


def test_get_close_prices_extracts_close_level():
    client = YFinanceClient(fetch_fn=_fake_download)

    result = client.get_close_prices(["AAPL", "MSFT"], period="5y", interval="1d")

    assert list(result.columns) == ["AAPL", "MSFT"]
    assert len(result) == 3


def test_get_close_prices_passes_through_args():
    seen = {}

    def fetch_fn(tickers, period, interval):
        seen["tickers"] = tickers
        seen["period"] = period
        seen["interval"] = interval
        return _fake_download(tickers, period, interval)

    client = YFinanceClient(fetch_fn=fetch_fn)
    client.get_close_prices(["AAPL"], period="1y", interval="1wk")

    assert seen == {"tickers": ["AAPL"], "period": "1y", "interval": "1wk"}
