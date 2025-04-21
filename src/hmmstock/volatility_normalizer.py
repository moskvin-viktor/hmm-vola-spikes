import pandas as pd
import numpy as np

class VolatilityNormalizer:
    def __init__(self, method: str = "zscore"):
        self.method = method

    def normalize(self, series: pd.Series) -> pd.Series:
        if self.method == "zscore":
            return (series - series.mean()) / series.std()
        elif self.method == "minmax":
            return (series - series.min()) / (series.max() - series.min())
        elif self.method == "log":
            return np.log1p(series)
        elif self.method == "none":
            return series
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")