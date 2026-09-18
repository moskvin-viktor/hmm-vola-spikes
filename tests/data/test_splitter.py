import numpy as np
import pytest
from pydantic import ValidationError

from hmmstock.data.splitter import SplitConfig, train_test_holdout, walk_forward_splits


def test_train_test_holdout_default_ratio_is_chronological():
    X = np.arange(100).reshape(100, 1)

    X_trainval, X_test = train_test_holdout(X)

    assert len(X_trainval) == 85
    assert len(X_test) == 15
    assert X_trainval.flatten().tolist() == list(range(85))
    assert X_test.flatten().tolist() == list(range(85, 100))


def test_train_test_holdout_reads_test_size_from_config():
    X = np.arange(100).reshape(100, 1)

    X_trainval, X_test = train_test_holdout(X, config=SplitConfig(test_size=0.2))

    assert len(X_trainval) == 80
    assert len(X_test) == 20


def test_train_test_holdout_never_shuffles():
    # X_test must always be the chronologically LAST rows -- a shuffled
    # holdout would let training data from after the "test" period leak
    # into what's supposed to be an untouched, chronologically later slice.
    X = np.arange(20).reshape(20, 1)

    _, X_test = train_test_holdout(X, config=SplitConfig(test_size=0.5))

    assert X_test.flatten().tolist() == list(range(10, 20))


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
        config.test_size = 0.5
