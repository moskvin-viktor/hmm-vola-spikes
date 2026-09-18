from typing import cast

import numpy as np
import pandas as pd


class VolatilityNormalizer:
    def __init__(self, method: str = "zscore"):
        self.method = method

    def normalize(self, series: pd.Series) -> pd.Series:
        if self.method == "zscore":
            return (series - series.mean()) / series.std()
        elif self.method == "minmax":
            return (series - series.min()) / (series.max() - series.min())
        elif self.method == "log":
            return cast(pd.Series, np.log1p(series))
        elif self.method == "none":
            return series
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
