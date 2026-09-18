"""End-to-end test against the real FRED CSV endpoint.

Hits the network for real -- excluded from the default `hatch run test`
(see [tool.pytest.ini_options] addopts). Run explicitly with
`hatch run test-e2e`.
"""

import pytest

from hmmstock.data.fred_client import FredClient

pytestmark = pytest.mark.e2e


def test_get_close_prices_against_live_fred():
    client = FredClient()

    result = client.get_close_prices(["VIXCLS", "DGS10"], period="5y", interval="1d")

    assert list(result.columns) == ["VIXCLS", "DGS10"]
    assert len(result) > 100
    for col in result.columns:
        assert result[col].dropna().std() > 0, f"{col}: zero-variance series"
