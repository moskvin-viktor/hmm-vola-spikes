from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections.abc import Callable

from hmmlearn import hmm


class MarkovModel(ABC):
    """Abstract base class for Markov models."""

    is_layered: bool = False
    best_score: float = -np.inf
    normalized_ll: float | None = None
    entropy: float | None = None
    random_state: int | None = None
    model = None

    @abstractmethod
    def fit(
        self, splitter: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    ) -> hmm.GaussianHMM | None:
        pass

    @abstractmethod
    def predict_states(self) -> np.ndarray | pd.DataFrame | None:
        pass
