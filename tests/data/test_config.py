import pytest
from pydantic import ValidationError

from hmmstock.data.config import DataConfig


def _base_kwargs(**overrides):
    kwargs = dict(
        tickers=["AAPL", "MSFT"],
        period="5y",
        interval="1d",
        volatility_windows=[5, 10, 20],
        market_proxy_processing={"default_market_vola_proxy": "^VIX"},
    )
    kwargs.update(overrides)
    return kwargs


def test_all_tickers_includes_market_proxy():
    order = DataConfig(**_base_kwargs())

    assert order.all_tickers == ["AAPL", "MSFT", "^VIX"]


def test_all_tickers_dedupes_proxy_already_in_tickers():
    order = DataConfig(
        **_base_kwargs(
            tickers=["AAPL", "^VIX"],
            market_proxy_processing={"default_market_vola_proxy": "^VIX"},
        )
    )

    assert order.all_tickers == ["AAPL", "^VIX"]


def test_defaults_applied_for_optional_sections():
    order = DataConfig(**_base_kwargs())

    assert order.date_filter.start is None
    assert order.volatility_processing.method == "zscore"


def test_is_immutable():
    order = DataConfig(**_base_kwargs())

    with pytest.raises(ValidationError):
        order.tickers = ["GOOG"]


def test_missing_required_field_raises():
    kwargs = _base_kwargs()
    del kwargs["tickers"]

    with pytest.raises(ValidationError):
        DataConfig(**kwargs)
