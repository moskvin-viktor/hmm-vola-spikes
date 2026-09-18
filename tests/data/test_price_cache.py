import pandas as pd

from hmmstock.data.price_cache import PriceCache


def test_load_returns_none_when_missing(tmp_path):
    cache = PriceCache(tmp_path / "missing.pkl")

    assert cache.load() is None


def test_save_then_load_roundtrips(tmp_path):
    cache = PriceCache(tmp_path / "nested" / "prices.pkl")
    df = pd.DataFrame({"AAPL": [1.0, 2.0]})

    cache.save(df)
    loaded = cache.load()

    pd.testing.assert_frame_equal(loaded, df)
