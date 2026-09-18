import numpy as np
import pytest
from pydantic import ValidationError

from hmmstock.data.splitter import SplitConfig, adaptive_n_splits, walk_forward_splits


def test_walk_forward_splits_never_validates_on_data_older_than_training():
    X = np.arange(60).reshape(60, 1)

    for X_train, X_val in walk_forward_splits(X, n_splits=3):
        assert X_train.max() < X_val.min()


def test_walk_forward_splits_yields_n_splits_folds():
    X = np.arange(60).reshape(60, 1)

    folds = list(walk_forward_splits(X, n_splits=4))

    assert len(folds) == 4


def test_walk_forward_splits_training_set_expands_each_fold():
    X = np.arange(60).reshape(60, 1)

    train_sizes = [len(X_train) for X_train, _ in walk_forward_splits(X, n_splits=3)]

    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)  # strictly expanding


def test_split_config_is_immutable():
    config = SplitConfig()

    with pytest.raises(ValidationError):
        config.n_splits = 10


def test_adaptive_n_splits_caps_down_for_scarce_data():
    assert adaptive_n_splits(n_samples=30, requested=5, min_per_fold=10) == 3


def test_adaptive_n_splits_never_goes_below_two():
    assert adaptive_n_splits(n_samples=10, requested=5, min_per_fold=10) == 2


def test_adaptive_n_splits_never_exceeds_requested():
    assert adaptive_n_splits(n_samples=10_000, requested=5, min_per_fold=10) == 5
