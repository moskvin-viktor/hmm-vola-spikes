import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .base import RegimeModel
from .config import HMMConfig
from .trainer import refit_gaussian_hmm, score_on_test_holdout, select_best_gaussian_hmm

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
        self.cv_score = -np.inf

    def fit(self, splitter: Callable, n_splits: int) -> hmm.GaussianHMM | None:
        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train. Skipping.")
            return None

        np.random.seed(self.cfg.random_seed)
        X_trainval, X_test = splitter(self.X)

        if len(X_trainval) < 20:
            logger.warning(f"[{self.name}] Not enough trainval data after holdout.")
            return None

        best_n, best_seed, cv_score = select_best_gaussian_hmm(
            X_trainval,
            component_range=range(2, self.cfg.max_components + 1),
            n_fits=self.cfg.n_fits,
            n_splits=n_splits,
            covariance_type=self.cfg.covariance_type,
            init_params=self.cfg.init_params,
            n_iter=self.cfg.n_iter,
            tol=self.cfg.tol,
            evaluation_metric=self.evaluation_metric,
            log_prefix=f"[{self.name}] ",
        )

        if best_n is None or best_seed is None:
            logger.warning(f"[{self.name}] CV found no viable model.")
            return None

        self.cv_score = cv_score
        test_score = score_on_test_holdout(
            X_trainval,
            X_test,
            cv_score,
            n_components=best_n,
            seed=best_seed,
            covariance_type=self.cfg.covariance_type,
            init_params=self.cfg.init_params,
            n_iter=self.cfg.n_iter,
            tol=self.cfg.tol,
            evaluation_metric=self.evaluation_metric,
            log_prefix=f"[{self.name}] ",
        )
        if test_score > self.best_score:
            self.best_score = test_score

        # Deployed model: refit the winning config on all data (train + CV + test).
        final_model = refit_gaussian_hmm(
            self.X,
            n_components=best_n,
            seed=best_seed,
            covariance_type=self.cfg.covariance_type,
            init_params=self.cfg.init_params,
            n_iter=self.cfg.n_iter,
            tol=self.cfg.tol,
        )
        self.layer = final_model
        return final_model

    def predict_states(self) -> np.ndarray | None:
        if self.layer is None:
            return None
        raw_states = self.layer.predict(self.X)
        return self._relabel_states_by_volatility(raw_states, self.layer, self.X)

    def transition_matrices(self) -> list[pd.DataFrame]:
        if self.layer is None:
            return []
        return [self._transition_matrix_df(self.layer, layer_idx=0)]
