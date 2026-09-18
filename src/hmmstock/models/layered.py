import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .base import RegimeModel
from .config import LayeredHMMConfig
from .trainer import refit_gaussian_hmm, score_on_test_holdout, select_best_gaussian_hmm

logger = logging.getLogger(__name__)


class LayeredHMMModel(RegimeModel):
    """Stacks GaussianHMMs sequentially: each layer is trained on the
    original features plus the posterior state probabilities of the
    previous layer, letting later layers learn increasingly abstract
    regimes."""

    config_cls = LayeredHMMConfig

    def __init__(
        self, name: str, X: np.ndarray, config: LayeredHMMConfig, evaluation_metric
    ):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.layers: list[hmm.GaussianHMM] = []
        self.best_score = -np.inf
        self.cv_score = -np.inf

    def fit(self, splitter: Callable, n_splits: int) -> hmm.GaussianHMM | None:
        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train")
            return None

        original_X = self.X.copy()
        current_X = original_X
        np.random.seed(self.cfg.random_seed)
        final_model = None

        for layer_idx, layer_cfg in enumerate(self.cfg.layers):
            log_prefix = f"[{self.name}] Layer {layer_idx + 1} "
            logger.info(f"[{self.name}] Training Layer {layer_idx + 1}")

            X_trainval, X_test = splitter(current_X)
            if len(X_trainval) < 20:
                logger.error(f"{log_prefix}Not enough trainval data after holdout.")
                return None

            best_n, best_seed, cv_score = select_best_gaussian_hmm(
                X_trainval,
                component_range=range(
                    layer_cfg.min_components, layer_cfg.max_components + 1
                ),
                n_fits=self.cfg.n_fits,
                n_splits=n_splits,
                covariance_type=layer_cfg.covariance_type,
                init_params=layer_cfg.init_params,
                n_iter=layer_cfg.n_iter,
                tol=self.cfg.tol,
                evaluation_metric=self.evaluation_metric,
                log_prefix=log_prefix,
            )

            if best_n is None or best_seed is None:
                logger.error(f"{log_prefix}CV found no viable model.")
                return None

            if cv_score > self.cv_score:
                self.cv_score = cv_score

            test_score = score_on_test_holdout(
                X_trainval,
                X_test,
                cv_score,
                n_components=best_n,
                seed=best_seed,
                covariance_type=layer_cfg.covariance_type,
                init_params=layer_cfg.init_params,
                n_iter=layer_cfg.n_iter,
                tol=self.cfg.tol,
                evaluation_metric=self.evaluation_metric,
                log_prefix=log_prefix,
            )
            if test_score > self.best_score:
                self.best_score = test_score

            # Deployed layer: refit the winning config on all of this
            # layer's current_X (train + CV + test).
            layer_model = refit_gaussian_hmm(
                current_X,
                n_components=best_n,
                seed=best_seed,
                covariance_type=layer_cfg.covariance_type,
                init_params=layer_cfg.init_params,
                n_iter=layer_cfg.n_iter,
                tol=self.cfg.tol,
            )
            if layer_model is None:
                logger.error(f"{log_prefix}Final refit failed.")
                return None

            self.layers.append(layer_model)
            final_model = layer_model

            posterior = layer_model.predict_proba(current_X)
            current_X = np.hstack([original_X, posterior])

        return final_model

    def predict_states(self) -> pd.DataFrame | None:
        if not self.layers:
            return None

        original_X = self.X
        current_X = original_X
        all_layer_states = {}
        relabeled_states = None

        for idx, model in enumerate(self.layers):
            raw_states = model.predict(current_X)
            relabeled_states = self._relabel_states_by_volatility(
                raw_states, model, current_X
            )
            all_layer_states[f"regime_layer{idx}"] = relabeled_states

            if idx < len(self.layers) - 1:
                posterior = model.predict_proba(current_X)
                current_X = np.hstack([original_X, posterior])

        assert relabeled_states is not None  # self.layers is non-empty
        return pd.DataFrame(
            all_layer_states, index=pd.RangeIndex(len(self.X))[-len(relabeled_states) :]
        )

    def transition_matrices(self) -> list[pd.DataFrame]:
        return [
            self._transition_matrix_df(model, layer_idx)
            for layer_idx, model in enumerate(self.layers)
        ]
