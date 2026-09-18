"""End-to-end test against the real yfinance API.

Hits the network for real -- excluded from the default `hatch run test`
(see [tool.pytest.ini_options] addopts). Run explicitly with
`hatch run test-e2e`.
"""

import pytest

from hmmstock.data.config import DataConfig
from hmmstock.data.pipeline import run_pipeline
from hmmstock.data.price_cache import PriceCache
from hmmstock.data.yfinance_client import YFinanceClient

pytestmark = pytest.mark.e2e

TICKERS = ["AAPL", "MSFT", "^GSPC", "AMZN"]


def test_run_pipeline_against_live_yfinance(tmp_path):
    order = DataConfig(
        tickers=TICKERS,
        period="1y",
        interval="1d",
        volatility_windows=[5, 10, 20],
        market_proxy_processing={"default_market_vola_proxy": "^VIX"},
    )
    assert len(order.tickers) == 4

    client = YFinanceClient()  # real yf.download, no fetch_fn override
    cache = PriceCache(tmp_path / "e2e_cache.pkl")  # isolated from the real cache

    data = run_pipeline(order, client=client, cache=cache)

    assert set(data.keys()) == set(order.tickers)
    for ticker, df in data.items():
        assert len(df) > 50, f"{ticker}: unexpectedly few rows ({len(df)})"
        assert not df.isna().any().any(), f"{ticker}: unexpected NaNs in pipeline output"

        # Control variance: a flat/stale/bad ticker would silently
        # zscore-normalize into garbage instead of failing loudly, so
        # assert the real signal actually varies.
        assert df["normalized_returns"].std() > 0, f"{ticker}: zero-variance returns"
        for window in order.volatility_windows:
            col = f"vol_{window}"
            assert df[col].std() > 0, f"{ticker}: zero-variance {col}"
