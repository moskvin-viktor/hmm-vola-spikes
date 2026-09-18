import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .base import RegimeModel
from .config import HierarchicalHMMConfig
from .trainer import fit_best_gaussian_hmm

logger = logging.getLogger(__name__)


class HierarchicalHMMModel(RegimeModel):
    """Two-level HMM: a top-level HMM governs transitions between
    high-level regimes, and a separate sub-HMM is trained per top-level
    regime to model local dynamics within it.

    Note: sub-HMMs are trained and scored on the same partition of data
    (no held-out split within a regime) -- there typically isn't enough
    data per regime for a further split. transition_matrices() currently
    only exposes the top-level matrix; no per-sub-HMM matrices are saved.
    """

    config_cls = HierarchicalHMMConfig

    def __init__(
        self, name: str, X: np.ndarray, config: HierarchicalHMMConfig, evaluation_metric
    ):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.top: hmm.GaussianHMM | None = None
        self.sub_models: dict[int, hmm.GaussianHMM] = {}
        self.best_score = -np.inf

    def fit(self, splitter: Callable) -> hmm.GaussianHMM | None:
        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train")
            return None

        np.random.seed(self.cfg.random_seed)
        X_train, X_validate = splitter(self.X)

        logger.info(f"[{self.name}] Training Top-level HMM")
        top_cfg = self.cfg.top_layer
        best_top, _ = fit_best_gaussian_hmm(
            X_train,
            X_validate,
            component_range=range(top_cfg.min_components, top_cfg.max_components + 1),
            n_fits=self.cfg.n_fits,
            covariance_type=top_cfg.covariance_type,
            init_params=top_cfg.init_params,
            tol=self.cfg.tol,
            evaluation_metric=self.evaluation_metric,
            log_prefix=f"[{self.name}] Top ",
        )

        if best_top is None:
            logger.error(f"[{self.name}] No Top-level model could be trained")
            return None

        self.top = best_top
        top_states = self.top.predict(self.X)

        sub_cfg = self.cfg.sub_layer
        for top_state in np.unique(top_states):
            logger.info(f"[{self.name}] Training Sub-HMM for Top State {top_state}")
            sub_X = self.X[top_states == top_state]

            if len(sub_X) < 10:
                logger.warning(
                    f"[{self.name}] Not enough samples for Sub-HMM in Top State {top_state}"
                )
                continue

            best_sub, best_sub_score = fit_best_gaussian_hmm(
                sub_X,
                sub_X,
                component_range=range(
                    sub_cfg.min_components, sub_cfg.max_components + 1
                ),
                n_fits=self.cfg.n_fits,
                covariance_type=sub_cfg.covariance_type,
                init_params=sub_cfg.init_params,
                tol=self.cfg.tol,
                evaluation_metric=self.evaluation_metric,
                log_prefix=f"[{self.name}] Sub {top_state} ",
            )

            if best_sub is not None:
                self.sub_models[top_state] = best_sub
                if best_sub_score > self.best_score:
                    self.best_score = best_sub_score
            else:
                logger.error(
                    f"[{self.name}] No Sub-HMM could be trained for Top State {top_state}"
                )

        return self.top

    def predict_states(self) -> pd.DataFrame | None:
        if self.top is None or not self.sub_models:
            return None

        top_states = self.top.predict(self.X)
        sub_states = []

        for idx, top_state in enumerate(top_states):
            sub_model = self.sub_models.get(top_state)
            if sub_model is None:
                sub_states.append(np.nan)
            else:
                sub_states.append(sub_model.predict(self.X[idx : idx + 1])[0])

        return pd.DataFrame(
            {"top_level_state": top_states, "sub_level_state": sub_states},
            index=pd.RangeIndex(len(self.X)),
        )

    def transition_matrices(self) -> list[pd.DataFrame]:
        if self.top is None:
            return []
        return [self._transition_matrix_df(self.top, layer_idx=0)]
