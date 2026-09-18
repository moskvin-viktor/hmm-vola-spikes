import numpy as np
import pandas as pd
import pytest

from hmmstock.data.config import DataConfig
from hmmstock.data.pipeline import (
    assemble_output,
    compute_market_proxy,
    compute_returns,
    compute_rolling_volatility,
    fetch_prices,
    normalize_returns,
    run_pipeline,
)
from hmmstock.data.price_cache import PriceCache


class FakeClient:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.calls = 0

    def get_close_prices(self, tickers, period, interval):
        self.calls += 1
        return self.prices[tickers]


def _synthetic_prices(tickers, n=200, seed=0):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {t: 100 + np.cumsum(rng.normal(0, 1, n)) for t in tickers}, index=dates
    )


def _order(**overrides):
    kwargs = dict(
        tickers=["AAPL", "MSFT"],
        period="1y",
        interval="1d",
        volatility_windows=[5, 10, 20],
        market_proxy_processing={"default_market_vola_proxy": "VIX"},
    )
    kwargs.update(overrides)
    return DataConfig(**kwargs)


# -- fetch_prices --------------------------------------------------------


def test_fetch_prices_calls_client_and_populates_cache_when_empty(tmp_path):
    order = _order()
    prices = _synthetic_prices(order.all_tickers)
    client = FakeClient(prices)
    cache = PriceCache(tmp_path / "cache.pkl")

    result = fetch_prices(order, client, cache)

    assert client.calls == 1
    assert cache.load() is not None
    assert list(result.columns) == order.all_tickers


def test_fetch_prices_uses_cache_on_second_call(tmp_path):
    order = _order()
    prices = _synthetic_prices(order.all_tickers)
    client = FakeClient(prices)
    cache = PriceCache(tmp_path / "cache.pkl")

    fetch_prices(order, client, cache)
    fetch_prices(order, client, cache)

    assert client.calls == 1


def test_fetch_prices_applies_date_filter(tmp_path):
    order = _order(date_filter={"start": "2020-06-01", "end": "2020-06-30"})
    prices = _synthetic_prices(order.all_tickers)
    client = FakeClient(prices)
    cache = PriceCache(tmp_path / "cache.pkl")

    result = fetch_prices(order, client, cache)

    assert result.index.min() >= pd.Timestamp("2020-06-01")
    assert result.index.max() <= pd.Timestamp("2020-06-30")


# -- pure transform stages ------------------------------------------------


def test_compute_returns_is_log_returns():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 121.0]}, index=pd.date_range("2024-01-01", periods=3)
    )

    returns = compute_returns(prices)

    expected = np.log(prices / prices.shift(1)).dropna()
    pd.testing.assert_frame_equal(returns, expected)


def test_normalize_returns_mean_centers():
    returns = pd.DataFrame({"AAPL": [0.1, 0.2, 0.3]})

    normalized = normalize_returns(returns)

    assert normalized["AAPL"].mean() == pytest.approx(0, abs=1e-9)


def test_compute_rolling_volatility_uses_configured_method():
    order = _order(
        tickers=["AAPL"],
        volatility_windows=[2],
        volatility_processing={"method": "minmax"},
    )
    returns = pd.DataFrame({"AAPL": [0.1, -0.2, 0.3, -0.1, 0.05, 0.2]})

    volatility = compute_rolling_volatility(returns, order)

    col = volatility["AAPL"]["vol_2"].dropna()
    assert col.min() == pytest.approx(0.0)
    assert col.max() == pytest.approx(1.0)


def test_compute_market_proxy_uses_configured_ticker():
    order = _order(
        tickers=["AAPL"], market_proxy_processing={"default_market_vola_proxy": "VIX"}
    )
    prices = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0], "VIX": [10.0, 20.0, 30.0]})

    proxy = compute_market_proxy(prices, order)

    assert proxy.tolist() == [10.0, 20.0, 30.0]


def test_assemble_output_combines_columns_and_drops_nan_rows():
    idx = pd.date_range("2024-01-01", periods=3)
    order = _order(tickers=["AAPL"], volatility_windows=[2])
    normalized_returns = pd.DataFrame({"AAPL": [0.1, 0.2, 0.3]}, index=idx)
    rolling_volatility = {
        "AAPL": pd.DataFrame({"vol_2": [np.nan, 0.5, 0.6]}, index=idx)
    }
    market_proxy = pd.Series([1.0, 2.0, 3.0], index=idx)

    output = assemble_output(
        normalized_returns, rolling_volatility, market_proxy, order
    )

    df = output["AAPL"]
    assert list(df.columns) == ["normalized_returns", "vol_2", "market_vola"]
    assert len(df) == 2


# -- end-to-end -------------------------------------------------------------


def test_run_pipeline_end_to_end(tmp_path):
    order = _order()
    prices = _synthetic_prices(order.all_tickers)
    client = FakeClient(prices)
    cache = PriceCache(tmp_path / "cache.pkl")

    data = run_pipeline(order, client=client, cache=cache)

    assert set(data.keys()) == {"AAPL", "MSFT"}
    for df in data.values():
        assert list(df.columns) == [
            "normalized_returns",
            "vol_5",
            "vol_10",
            "vol_20",
            "market_vola",
        ]
        assert len(df) > 0
        assert not df.isna().any().any()
