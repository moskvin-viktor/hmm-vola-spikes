import logging

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .base import RegimeModel
from .config import HierarchicalHMMConfig
from .trainer import refit_gaussian_hmm, select_best_gaussian_hmm

logger = logging.getLogger(__name__)


class HierarchicalHMMModel(RegimeModel):
    """Two-level HMM: a top-level HMM governs transitions between
    high-level regimes, and a separate sub-HMM is trained per top-level
    regime to model local dynamics within it.

    The top-level HMM gets the same walk-forward CV treatment as
    HMMModel. Sub-HMMs are trained on partitions of an already-small
    dataset (must have >=10 samples); select_best_gaussian_hmm caps
    n_splits down for them automatically (adaptive_n_splits) when there
    isn't enough data to support the requested fold count -- still
    genuine walk-forward CV, just fewer/bigger folds.

    Both top-level and sub-level states are relabeled by volatility (see
    RegimeModel._relabel_states_by_volatility). Sub-level ranks are
    computed per sub-HMM, so "sub-level_state 0" always means "the
    lowest-variance sub-state within that particular top-level regime" --
    not a globally comparable index across different regimes'
    independently fitted sub-HMMs; group by top_level_state before
    comparing sub_level_state across rows.

    transition_matrices() currently only exposes the top-level matrix; no
    per-sub-HMM matrices are saved.
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
        self.cv_score = -np.inf

    def fit(self, n_splits: int) -> hmm.GaussianHMM | None:
        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train")
            return None

        np.random.seed(self.cfg.random_seed)

        logger.info(f"[{self.name}] Training Top-level HMM")
        top_cfg = self.cfg.top_layer
        best_n, best_seed, cv_score = select_best_gaussian_hmm(
            self.X,
            component_range=range(top_cfg.min_components, top_cfg.max_components + 1),
            n_fits=self.cfg.n_fits,
            n_splits=n_splits,
            covariance_type=top_cfg.covariance_type,
            init_params=top_cfg.init_params,
            n_iter=top_cfg.n_iter,
            tol=self.cfg.tol,
            evaluation_metric=self.evaluation_metric,
            log_prefix=f"[{self.name}] Top ",
        )

        if best_n is None or best_seed is None:
            logger.error(f"[{self.name}] No Top-level model could be trained")
            return None

        self.cv_score = cv_score

        # Deployed top-level model: refit the winning config on all data.
        self.top = refit_gaussian_hmm(
            self.X,
            n_components=best_n,
            seed=best_seed,
            covariance_type=top_cfg.covariance_type,
            init_params=top_cfg.init_params,
            n_iter=top_cfg.n_iter,
            tol=self.cfg.tol,
        )
        if self.top is None:
            logger.error(f"[{self.name}] Top-level final refit failed")
            return None

        top_states = self._relabel_states_by_volatility(
            self.top.predict(self.X), self.top
        )

        sub_cfg = self.cfg.sub_layer
        for top_state in np.unique(top_states):
            log_prefix = f"[{self.name}] Sub {top_state} "
            logger.info(f"[{self.name}] Training Sub-HMM for Top State {top_state}")
            sub_X = self.X[top_states == top_state]

            if len(sub_X) < 10:
                logger.warning(
                    f"[{self.name}] Not enough samples for Sub-HMM in Top State {top_state}"
                )
                continue

            best_sub_n, best_sub_seed, sub_score = select_best_gaussian_hmm(
                sub_X,
                component_range=range(
                    sub_cfg.min_components, sub_cfg.max_components + 1
                ),
                n_fits=self.cfg.n_fits,
                n_splits=n_splits,
                covariance_type=sub_cfg.covariance_type,
                init_params=sub_cfg.init_params,
                n_iter=sub_cfg.n_iter,
                tol=self.cfg.tol,
                evaluation_metric=self.evaluation_metric,
                log_prefix=log_prefix,
            )

            if best_sub_n is None or best_sub_seed is None:
                logger.error(
                    f"[{self.name}] No Sub-HMM could be trained for Top State {top_state}"
                )
                continue

            # Deployed sub-model: refit the winning config on all of this
            # regime's data.
            final_sub_model = refit_gaussian_hmm(
                sub_X,
                n_components=best_sub_n,
                seed=best_sub_seed,
                covariance_type=sub_cfg.covariance_type,
                init_params=sub_cfg.init_params,
                n_iter=sub_cfg.n_iter,
                tol=self.cfg.tol,
            )
            if final_sub_model is None:
                logger.error(f"{log_prefix}Final refit failed")
                continue

            self.sub_models[top_state] = final_sub_model
            if sub_score > self.cv_score:
                self.cv_score = sub_score

        return self.top

    def predict_states(self) -> pd.DataFrame | None:
        if self.top is None or not self.sub_models:
            return None

        top_states = self._relabel_states_by_volatility(
            self.top.predict(self.X), self.top
        )
        sub_relabel_maps = {
            top_state: self._volatility_rank_map(sub_model)
            for top_state, sub_model in self.sub_models.items()
        }

        sub_states = []
        for idx, top_state in enumerate(top_states):
            sub_model = self.sub_models.get(top_state)
            if sub_model is None:
                sub_states.append(np.nan)
            else:
                raw_sub_state = sub_model.predict(self.X[idx : idx + 1])[0]
                sub_states.append(sub_relabel_maps[top_state][raw_sub_state])

        return pd.DataFrame(
            {"top_level_state": top_states, "sub_level_state": sub_states},
            index=pd.RangeIndex(len(self.X)),
        )

    def transition_matrices(self) -> list[pd.DataFrame]:
        if self.top is None:
            return []
        return [self._transition_matrix_df(self.top, layer_idx=0)]
