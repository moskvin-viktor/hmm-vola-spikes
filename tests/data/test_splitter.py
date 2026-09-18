import numpy as np
import pytest
from pydantic import ValidationError

from hmmstock.data.splitter import SplitConfig, train_val_split


def test_default_ratio_and_no_shuffle_is_time_ordered():
    X = np.arange(10).reshape(10, 1)

    X_train, X_val = train_val_split(X)

    assert len(X_train) == 8
    assert len(X_val) == 2
    assert X_train.flatten().tolist() == list(range(8))
    assert X_val.flatten().tolist() == [8, 9]


def test_reads_train_size_from_config():
    X = np.arange(10).reshape(10, 1)

    X_train, X_val = train_val_split(X, config=SplitConfig(train_size=0.5))

    assert len(X_train) == 5
    assert len(X_val) == 5


def test_shuffle_preserves_row_membership():
    X = np.arange(10).reshape(10, 1)

    X_train, X_val = train_val_split(
        X, config=SplitConfig(train_size=0.5, shuffle=True)
    )

    combined = sorted(np.concatenate([X_train, X_val]).flatten().tolist())
    assert combined == list(range(10))


def test_config_is_immutable():
    config = SplitConfig()

    with pytest.raises(ValidationError):
        config.train_size = 0.5
