from hmmstock.model import sanitize_ticker


def test_sanitize_ticker_strips_caret():
    assert sanitize_ticker("^GSPC") == "GSPC"


def test_sanitize_ticker_replaces_slash():
    assert sanitize_ticker("BRK/B") == "BRK_B"


def test_sanitize_ticker_leaves_plain_ticker():
    assert sanitize_ticker("AAPL") == "AAPL"
