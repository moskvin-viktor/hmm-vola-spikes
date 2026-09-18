from abc import ABC, abstractmethod
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
    cv_score: float  # walk-forward CV score fit() used to select hyperparameters

    @abstractmethod
    def __init__(
        self, name: str, X: np.ndarray, config: BaseModel, evaluation_metric
    ): ...

    @abstractmethod
    def fit(self, n_splits: int) -> hmm.GaussianHMM | None:
        """Fits the model. Hyperparameters are selected by `n_splits`-fold
        walk-forward CV over all of `self.X` (never validates on data
        older than its own training slice); `self.cv_score` ends up
        holding the winning config's average CV score. The deployed model
        is then refit on all of `self.X` for maximal information. Returns
        the deployed GaussianHMM, or None if fitting failed (e.g. not
        enough data)."""

    @abstractmethod
    def predict_states(self) -> np.ndarray | pd.DataFrame | None:
        """Predicted regime state(s) for `self.X`, or None if unfitted."""

    @abstractmethod
    def transition_matrices(self) -> list[pd.DataFrame]:
        """One transition-matrix DataFrame per trained HMM layer."""

    @staticmethod
    def _volatility_rank_map(model: hmm.GaussianHMM) -> dict[int, int]:
        """Maps each of `model`'s raw state indices to its rank by total
        variance (trace of that state's fitted covariance matrix),
        ascending -- 0 = lowest-variance state. Based on the model's own
        fitted covariance, not raw observation dispersion: computing
        np.std() over an observation slice with more than one feature
        column would flatten unrelated feature units (e.g. returns and
        several differently-scaled rolling-volatility windows) into one
        meaningless scalar.
        """
        variances = [
            (state, float(np.trace(np.atleast_2d(model.covars_[state]))))
            for state in range(model.n_components)
        ]
        ranked = sorted(variances, key=lambda x: x[1])
        return {old: new for new, (old, _) in enumerate(ranked)}

    @classmethod
    def _relabel_states_by_volatility(
        cls, original_states: np.ndarray, model: hmm.GaussianHMM
    ) -> np.ndarray:
        """Relabels states 0..n-1 in order of increasing total variance."""
        state_map = cls._volatility_rank_map(model)
        return np.vectorize(state_map.get)(original_states)

    @staticmethod
    def _transition_matrix_df(model: hmm.GaussianHMM, layer_idx: int) -> pd.DataFrame:
        labels = [f"VS{layer_idx}_{i}" for i in range(model.n_components)]
        return pd.DataFrame(model.transmat_, index=labels, columns=labels)
