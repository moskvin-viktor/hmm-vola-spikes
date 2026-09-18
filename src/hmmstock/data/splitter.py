from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import TimeSeriesSplit


class SplitConfig(BaseModel):
    """Train/CV/test split order (from `config/model/default.yaml`'s
    `split:` section).

    test_size: fraction of the full series reserved as a final holdout,
        chronologically last, never used for model selection. Default 0.15.
    n_splits: number of expanding-window walk-forward CV folds over the
        remaining (non-test) data, used to select hyperparameters.
        Default 5.
    """

    model_config = ConfigDict(frozen=True)

    test_size: float = 0.15
    n_splits: int = 5


def train_test_holdout(
    X: np.ndarray, config: SplitConfig = SplitConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological holdout: the first (1 - test_size) fraction of X for
    training/CV, the last test_size fraction as an untouched test set.
    Never shuffles -- X is a time series, and a shuffled holdout would let
    training data from after the "test" period leak into selection."""
    n = len(X)
    split = int(n * (1 - config.test_size))
    return X[:split], X[split:]


def walk_forward_splits(
    X: np.ndarray, n_splits: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward CV folds, via sklearn's
    TimeSeriesSplit: each fold trains on all data before a cutoff and
    validates on the chunk immediately after it, with the cutoff moving
    forward each fold. A fold never validates on anything chronologically
    older than its own training data -- unlike shuffled k-fold, which
    would leak future rows into training."""
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        yield X[train_idx], X[val_idx]
