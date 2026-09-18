import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .base import RegimeModel
from .config import HMMConfig
from .trainer import fit_best_gaussian_hmm

logger = logging.getLogger(__name__)


class HMMModel(RegimeModel):
    """Standard single-layer Gaussian HMM, via hmmlearn."""

    config_cls = HMMConfig

    def __init__(self, name: str, X: np.ndarray, config: HMMConfig, evaluation_metric):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.layer: hmm.GaussianHMM | None = None
        self.best_score = -np.inf

    def fit(self, splitter: Callable) -> hmm.GaussianHMM | None:
        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train. Skipping.")
            return None

        np.random.seed(self.cfg.random_seed)
        X_train, X_validate = splitter(self.X)

        best_model, best_score = fit_best_gaussian_hmm(
            X_train,
            X_validate,
            component_range=range(2, self.cfg.max_components + 1),
            n_fits=self.cfg.n_fits,
            covariance_type=self.cfg.covariance_type,
            init_params=self.cfg.init_params,
            tol=self.cfg.tol,
            evaluation_metric=self.evaluation_metric,
            log_prefix=f"[{self.name}] ",
        )

        if best_score > self.best_score:
            self.best_score = best_score

        self.layer = best_model
        return best_model

    def predict_states(self) -> np.ndarray | None:
        if self.layer is None:
            return None
        raw_states = self.layer.predict(self.X)
        return self._relabel_states_by_volatility(raw_states, self.layer, self.X)

    def transition_matrices(self) -> list[pd.DataFrame]:
        if self.layer is None:
            return []
        return [self._transition_matrix_df(self.layer, layer_idx=0)]
