import numpy as np
import pandas as pd
import pytest

from hmmstock.data.market_vola_proxy_calcs import MarketVolatilityProxyCalculations


def _series():
    return pd.Series([10.0, 12.0, 11.0, 13.0, 14.0])


def test_raw_returns_series_unchanged():
    result = MarketVolatilityProxyCalculations(_series(), {"type": "raw"}).process()

    pd.testing.assert_series_equal(result, _series())


def test_smoothed_applies_rolling_mean():
    result = MarketVolatilityProxyCalculations(
        _series(), {"type": "smoothed", "smoothing_window": 2}
    ).process()

    pd.testing.assert_series_equal(result, _series().rolling(2).mean())


def test_returns_is_log_returns():
    result = MarketVolatilityProxyCalculations(_series(), {"type": "returns"}).process()

    expected = np.log(_series() / _series().shift(1)).dropna()
    pd.testing.assert_series_equal(result, expected)


def test_zscore_normalizes():
    result = MarketVolatilityProxyCalculations(_series(), {"type": "zscore"}).process()

    s = _series()
    expected = (s - s.mean()) / s.std()
    pd.testing.assert_series_equal(result, expected)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        MarketVolatilityProxyCalculations(_series(), {"type": "bogus"}).process()
