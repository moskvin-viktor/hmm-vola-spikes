import pandas as pd
import numpy as np
from enum import Enum


class NormalizationMethod(Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"
    LOG = "log"
    NONE = "none"


class VolatilityNormalizer:
    def __init__(self, method: str = "zscore"):
        self.method = NormalizationMethod(method)

    def normalize(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(dtype=float)  # Return empty Series if input is empty

        if self.method == NormalizationMethod.ZSCORE:
            return (series - series.mean()) / series.std()
        elif self.method == NormalizationMethod.MINMAX:
            s_min = series.min()
            s_max = series.max()
            if s_max == s_min:  # Avoid division by zero
                return pd.Series(0.0, index=series.index)
            return (series - s_min) / (s_max - s_min)
        elif self.method == NormalizationMethod.LOG:
            return series.transform(np.log1p)
        elif self.method == NormalizationMethod.NONE:
            return series
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
