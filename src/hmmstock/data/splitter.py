from collections.abc import Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import TimeSeriesSplit


class SplitConfig(BaseModel):
    """Walk-forward CV order (from `config/model/default.yaml`'s `split:`
    section).

    n_splits: number of expanding-window walk-forward CV folds, used to
        select hyperparameters. Default 5.
    """

    model_config = ConfigDict(frozen=True)

    n_splits: int = 5


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


def adaptive_n_splits(n_samples: int, requested: int, min_per_fold: int = 10) -> int:
    """Caps `requested` folds down so each fold gets roughly at least
    `min_per_fold` samples on average, for data too scarce to support the
    full requested fold count (e.g. HierarchicalHMMModel's per-regime
    sub-partitions). Never goes below 2 -- sklearn's TimeSeriesSplit
    requires at least that many."""
    return max(2, min(requested, n_samples // min_per_fold))
