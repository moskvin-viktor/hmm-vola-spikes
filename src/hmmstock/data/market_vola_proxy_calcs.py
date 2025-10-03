import numpy as np
import pandas as pd
from enum import Enum


class ProcessingType(Enum):
    RAW = "raw"
    SMOOTHED = "smoothed"
    RETURNS = "returns"
    ZSCORE = "zscore"


class MarketVolatilityProxyCalculations:
    def __init__(self, series: pd.Series, config: dict):
        self.series = series.dropna()
        self.config = config
        self.type = ProcessingType(config.get("type", "raw"))
        self.smoothing_window = config.get("smoothing_window", 5)

    def process(self):
        """
        Process the series based on the specified type.
        - "raw": Returns the original series.
        - "smoothed": Returns the smoothed series using a rolling mean.
        - "returns": Returns the log returns of the series.
        - "zscore": Returns the normalized series (z-score).
        """
        if self.series.empty:
            return pd.Series(dtype=float)  # Return empty Series if input is empty

        if self.type == ProcessingType.RAW:
            return self.series

        elif self.type == ProcessingType.SMOOTHED:
            return self.series.rolling(self.smoothing_window).mean()

        elif self.type == ProcessingType.RETURNS:
            return (self.series / self.series.shift(1)).transform(np.log).dropna()

        elif self.type == ProcessingType.ZSCORE:
            return (self.series - self.series.mean()) / (self.series.std())

        else:
            raise ValueError(f"Unknown processing type: {self.type}")
