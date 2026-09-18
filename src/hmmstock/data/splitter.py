import numpy as np
from pydantic import BaseModel, ConfigDict


class SplitConfig(BaseModel):
    """Train/validation split order (from `config/model.yaml`'s `split:` section).

    train_size: fraction of rows kept for training. Default 0.8.
    shuffle: shuffle row order before splitting. Default False (rows are
        time-ordered, so shuffling would leak future data into training).
    """

    model_config = ConfigDict(frozen=True)

    train_size: float = 0.8
    shuffle: bool = False


def train_val_split(
    X: np.ndarray, config: SplitConfig = SplitConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Splits `X` into (train, validation) arrays per `config`."""
    n = len(X)
    indices = np.random.permutation(n) if config.shuffle else np.arange(n)
    train_end = int(config.train_size * n)
    return X[indices[:train_end]], X[indices[train_end:]]
