import pandas as pd

from hmmstock.data.fred_client import FredClient


def _fake_series(series_id: str) -> pd.DataFrame:
    starts = {"VIXCLS": "2024-01-01", "DGS10": "2024-01-02"}
    dates = pd.date_range(starts[series_id], periods=3)
    return pd.DataFrame({series_id: [1.0, 2.0, 3.0]}, index=dates)


def test_get_close_prices_returns_one_column_per_series():
    client = FredClient(fetch_series=_fake_series)

    result = client.get_close_prices(["VIXCLS", "DGS10"], period="5y", interval="1d")

    assert list(result.columns) == ["VIXCLS", "DGS10"]


def test_get_close_prices_outer_joins_on_date():
    client = FredClient(fetch_series=_fake_series)

    result = client.get_close_prices(["VIXCLS", "DGS10"], period="5y", interval="1d")

    # VIXCLS starts 01-01, DGS10 starts 01-02 -> union of dates, gap as NaN
    assert len(result) == 4
    assert pd.isna(result.loc["2024-01-01", "DGS10"])
    assert result.loc["2024-01-02", "DGS10"] == 1.0


def test_get_close_prices_passes_through_series_ids():
    seen = []

    def fetch_series(series_id):
        seen.append(series_id)
        return pd.DataFrame(
            {series_id: [1.0]}, index=pd.date_range("2024-01-01", periods=1)
        )

    client = FredClient(fetch_series=fetch_series)
    client.get_close_prices(["VIXCLS", "DGS10"], period="1y", interval="1d")

    assert seen == ["VIXCLS", "DGS10"]
