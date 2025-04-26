import numpy as np
import pandas as pd
from omegaconf import OmegaConf

class MarketVolatilityProxyCalculations:
    def __init__(self, series: pd.Series, config: dict):
        self.series = series.dropna()
        self.config = config
        self.type = config.get("type", "raw")
        self.smoothing_window = config.get("smoothing_window", 5)

    def process(self):
        """
        Process the series based on the specified type.
        - "raw": Returns the original series.
        - "smoothed": Returns the smoothed series using a rolling mean.
        - "returns": Returns the log returns of the series.
        - "zscore": Returns the normalized series (z-score).
        """
        # Ensure the series is not empty after dropping NaN values
        if self.type == "raw":
            return self.series

        elif self.type == "smoothed":
            return self.series.rolling(self.smoothing_window).mean()

        elif self.type == "returns":
            return np.log(self.series / self.series.shift(1)).dropna()

        elif self.type == "zscore":
            return (self.series - self.series.mean()) / (self.series.std())

        else:
            raise ValueError(f"Unknown processing type: {self.type}")