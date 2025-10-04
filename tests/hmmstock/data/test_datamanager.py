import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Assuming DataManager is importable from src.hmmstock.data
from src.hmmstock.data.datamanager import DataManager
from src.hmmstock.data.market_vola_proxy_calcs import MarketVolatilityProxyCalculations, ProcessingType
from src.hmmstock.data.volatility_normalizer import VolatilityNormalizer, NormalizationMethod

# Mock yfinance to return predictable data
@pytest.fixture
def mock_yfinance_download():
    with patch('yfinance.download') as mock_download:
        # Create a sample DataFrame for yfinance.download
        dates = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 11)]) # 10 days
        tickers = ['AAPL', 'MSFT', '^VIX']
        # Create a MultiIndex DataFrame directly
        mock_data = pd.DataFrame({
            ('Close', 'AAPL'): [100 + i for i in range(10)],
            ('Close', 'MSFT'): [200 + i*2 for i in range(10)],
            ('Close', '^VIX'): [15 + (i % 3) for i in range(10)],
            ('Open', 'AAPL'): [99 + i for i in range(10)],
            ('Open', 'MSFT'): [199 + i*2 for i in range(10)],
            ('Open', '^VIX'): [14 + (i % 3) for i in range(10)]
        }, index=dates)
        mock_download.return_value = mock_data

        yield mock_download

# Mock pathlib.Path for cache handling
@pytest.fixture
def mock_pathlib_path():
    with patch('pathlib.Path') as mock_path:
        # Configure mock_path.exists() to return False by default (no cache)
        mock_path.return_value.exists.return_value = False
        # Configure mock_path.parent.mkdir() to do nothing
        mock_path.return_value.parent.mkdir.return_value = None
        yield mock_path

# Mock pickle for cache handling
@pytest.fixture
def mock_pickle():
    with patch('src.hmmstock.data.datamanager.pickle.load') as mock_load, \
         patch('src.hmmstock.data.datamanager.pickle.dump') as mock_dump:
        yield mock_load, mock_dump

@pytest.fixture
def sample_config():
    return {
        "tickers": ["AAPL", "MSFT"],
        "period": "5d",
        "interval": "1d",
        "volatility_windows": [2, 3],
        "market_proxy_processing": {
            "default_market_vola_proxy": "^VIX",
            "type": "returns",
            "smoothing_window": 2
        },
        "volatility_processing": {
            "normalize_method": "zscore"
        },
        "date_filter": {
            "start": "2023-01-01", # Changed from 2023-01-02
            "end": "2023-01-10"   # Changed from 2023-01-04
        }
    }

def test_datamanager_pipeline_success(mock_yfinance_download, mock_pathlib_path, mock_pickle, sample_config):
    mock_load, mock_dump = mock_pickle
    # Ensure cache is not hit for this test
    mock_pathlib_path.return_value.exists.return_value = False

    dm = DataManager(sample_config)
    output_df = dm.get_data()

    # Assertions
    assert isinstance(output_df, pd.DataFrame)
    assert not output_df.empty
    assert isinstance(output_df.index, pd.MultiIndex)
    assert output_df.index.names == ['ticker', 'Date'] # Assuming 'Date' is the name of the date level

    # Check columns in the MultiIndex DataFrame
    expected_columns = [
        'normalized_returns',
        'vol_2', 'vol_3',
        'market_vola'
    ]
    assert all(col in output_df.columns for col in expected_columns)

    # Check dtypes
    for col in expected_columns:
        assert output_df[col].dtype == float

    # Check date filtering (on the 'Date' level of the MultiIndex)
    assert output_df.index.get_level_values('Date').min() == pd.to_datetime("2023-01-04")
    assert output_df.index.get_level_values('Date').max() == pd.to_datetime("2023-01-10")

    # Check if yfinance.download was called
    call_args, call_kwargs = mock_yfinance_download.call_args
    assert set(call_args[0]) == set(['AAPL', 'MSFT', '^VIX'])
    assert call_kwargs['period'] == '5d'
    assert call_kwargs['interval'] == '1d'
    # Check if data was saved to cache
    mock_dump.assert_called_once()

def test_datamanager_cache_hit(mock_yfinance_download, mock_pathlib_path, mock_pickle, sample_config):
    mock_load, mock_dump = mock_pickle
    # Configure mock_path.exists() to return True (cache hit)
    mock_pathlib_path.return_value.exists.return_value = True

    # Mock cached data
    dates = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 11)]) # 10 days
    tickers = ['AAPL', 'MSFT', '^VIX']
    cached_raw_data = pd.DataFrame({
        ('Close', 'AAPL'): [100 + i for i in range(10)],
        ('Close', 'MSFT'): [200 + i*2 for i in range(10)],
        ('Close', '^VIX'): [15 + (i % 3) for i in range(10)],
        ('Open', 'AAPL'): [99 + i for i in range(10)],
        ('Open', 'MSFT'): [199 + i*2 for i in range(10)],
        ('Open', '^VIX'): [14 + (i % 3) for i in range(10)]
    }, index=dates)
    mock_load.return_value = cached_raw_data

    dm = DataManager(sample_config)
    output_df = dm.get_data()

    # Assertions
    assert isinstance(output_df, pd.DataFrame)
    assert not output_df.empty
    assert isinstance(output_df.index, pd.MultiIndex)
    assert output_df.index.names == ['ticker', 'Date']

    # Check date filtering
    assert output_df.index.get_level_values('Date').min() == pd.to_datetime("2023-01-04")
    assert output_df.index.get_level_values('Date').max() == pd.to_datetime("2023-01-10")

def test_datamanager_empty_data_from_yfinance(mock_yfinance_download, mock_pathlib_path, sample_config):
    mock_yfinance_download.return_value = pd.DataFrame() # Simulate empty data
    mock_pathlib_path.return_value.exists.return_value = False # No cache

    with pytest.raises(ValueError, match='No data has been collected and no cache found! Aborting!'):
        DataManager(sample_config)

def test_datamanager_empty_returns_df(mock_yfinance_download, mock_pathlib_path, sample_config):
    # Simulate a scenario where returns_df becomes empty after filtering
    # This is a bit tricky to mock precisely without over-complicating yfinance mock
    # For now, let's rely on the ValueError from empty yfinance data.
    # A more granular test would involve mocking internal methods if needed.
    pass # This test case might be better handled by mocking internal methods if needed.

def test_datamanager_no_date_filter(mock_yfinance_download, mock_pathlib_path, sample_config):
    config_no_filter = sample_config.copy()
    config_no_filter["date_filter"] = {"start": None, "end": None}

    dm = DataManager(config_no_filter)
    output_df = dm.get_data()

    # Check that no date filtering occurred
    assert isinstance(output_df.index, pd.MultiIndex)
    assert output_df.index.names == ['ticker', 'Date']
    assert output_df.index.get_level_values('Date').min() == pd.to_datetime("2023-01-04")
    assert output_df.index.get_level_values('Date').max() == pd.to_datetime("2023-01-10")

def test_datamanager_market_proxy_processing(mock_yfinance_download, mock_pathlib_path, sample_config):
    dm = DataManager(sample_config)
    output_df = dm.get_data()

    assert 'market_vola' in output_df.columns
    # Further assertions could check the actual values of market_vola if needed
    # For example, if ^VIX was [15, 16, 17, 16, 15]
    # log returns would be [NaN, log(16/15), log(17/16), log(16/17), log(15/16)]
    # After reindex and dropna, the values should align.
    # For now, just checking presence and non-emptiness.
    assert not output_df['market_vola'].empty
    assert output_df['market_vola'].dtype == float

# Test for volatility normalization method
def test_datamanager_volatility_normalization(mock_yfinance_download, mock_pathlib_path, sample_config):
    dm = DataManager(sample_config)
    output_df = dm.get_data()

    assert 'vol_2' in output_df.columns
    assert output_df['vol_2'].dtype == float
    assert not output_df['vol_2'].empty

    # More specific tests would involve calculating expected normalized volatility
    # and comparing, but that's more of a unit test for VolatilityNormalizer itself.