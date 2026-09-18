import numpy as np
import pandas as pd
import pytest

from hmmstock.data.volatility_normalizer import VolatilityNormalizer


def test_zscore_normalizes_to_zero_mean_unit_std():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = VolatilityNormalizer("zscore").normalize(series)

    assert result.mean() == pytest.approx(0, abs=1e-9)
    assert result.std() == pytest.approx(1, abs=1e-9)


def test_minmax_bounds_to_zero_one():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = VolatilityNormalizer("minmax").normalize(series)

    assert result.min() == 0
    assert result.max() == 1


def test_log_matches_np_log1p():
    series = pd.Series([0.0, 1.0, 2.0])

    result = VolatilityNormalizer("log").normalize(series)

    pd.testing.assert_series_equal(result, np.log1p(series))


def test_none_is_passthrough():
    series = pd.Series([1.0, 2.0])

    result = VolatilityNormalizer("none").normalize(series)

    pd.testing.assert_series_equal(result, series)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        VolatilityNormalizer("bogus").normalize(pd.Series([1.0]))
