import os 
import pickle
import pandas as pd
import yfinance as yf

class IVDataManager:
    """
    A class to fetch and cache implied volatility (IV) data from options chains.
    Currently fetches mean IV from nearest expiry calls for each ticker.
    """
    def __init__(self, tickers, cache_dir="data_cache"):
        """
        :param tickers: List of stock tickers.
        :param cache_dir: Directory to store cached IV data.
        """
        self.tickers = tickers
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "cached_iv_data.pkl")
        self.iv_data = self._fetch_iv_data()

    def _fetch_iv_data(self):
        """Fetch and cache implied volatility data."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Loaded cached IV data.")
                return pickle.load(f)

        print("Fetching implied volatility data...")
        iv_df = pd.DataFrame()

        for ticker in self.tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                expiries = yf_ticker.options
                if not expiries:
                    continue
                nearest_expiry = expiries[0]
                opt_chain = yf_ticker.option_chain(nearest_expiry)
                calls = opt_chain.calls

                iv_mean = calls['impliedVolatility'].mean()
                iv_df.at[nearest_expiry, ticker] = iv_mean
            except Exception as e:
                print(f"Error fetching IV for {ticker}: {e}")

        with open(self.cache_file, "wb") as f:
            pickle.dump(iv_df, f)

        return iv_df

    def get_iv_data(self):
        """Return the DataFrame of implied volatilities."""
        return self.iv_data
    

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "MSFT"]
    iv_data_manager = IVDataManager(tickers)
    iv_data = iv_data_manager.get_iv_data()
    print(iv_data)