from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pandas as pd
from hmmlearn import hmm
from pydantic import BaseModel


class RegimeModel(ABC):
    """Common interface for the HMM/LayeredHMM/HierarchicalHMM regime models.

    Subclasses store their trained GaussianHMM estimator(s) however suits
    them internally, but must expose predictions and transition matrices
    through this uniform interface so RegimeModelManager never needs to
    know which concrete model it's holding.
    """

    config_cls: ClassVar[type[BaseModel]]

    name: str
    X: np.ndarray
    best_score: float

    @abstractmethod
    def __init__(
        self, name: str, X: np.ndarray, config: BaseModel, evaluation_metric
    ): ...

    @abstractmethod
    def fit(self, splitter: Callable) -> hmm.GaussianHMM | None:
        """Fits the model. Returns the best/final trained GaussianHMM, or
        None if fitting failed (e.g. not enough data)."""

    @abstractmethod
    def predict_states(self) -> np.ndarray | pd.DataFrame | None:
        """Predicted regime state(s) for `self.X`, or None if unfitted."""

    @abstractmethod
    def transition_matrices(self) -> list[pd.DataFrame]:
        """One transition-matrix DataFrame per trained HMM layer."""

    @staticmethod
    def _relabel_states_by_volatility(
        original_states: np.ndarray, model: hmm.GaussianHMM, X: np.ndarray
    ) -> np.ndarray:
        """Relabels states 0..n-1 in order of increasing observation volatility."""
        state_vols = []
        for state in range(model.n_components):
            state_obs = X[original_states == state]
            vol = np.std(state_obs)
            state_vols.append((state, vol))

        sorted_states = sorted(state_vols, key=lambda x: x[1])
        state_map = {old: new for new, (old, _) in enumerate(sorted_states)}
        return np.vectorize(state_map.get)(original_states)

    @staticmethod
    def _transition_matrix_df(model: hmm.GaussianHMM, layer_idx: int) -> pd.DataFrame:
        labels = [f"VS{layer_idx}_{i}" for i in range(model.n_components)]
        return pd.DataFrame(model.transmat_, index=labels, columns=labels)
