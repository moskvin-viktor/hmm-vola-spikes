import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .base import RegimeModel
from .config import LayeredHMMConfig
from .trainer import fit_best_gaussian_hmm

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

    def fit(self, splitter: Callable) -> hmm.GaussianHMM | None:
        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train")
            return None

        original_X = self.X.copy()
        current_X = original_X
        np.random.seed(self.cfg.random_seed)
        best_model = None

        for layer_idx, layer_cfg in enumerate(self.cfg.layers):
            logger.info(f"[{self.name}] Training Layer {layer_idx + 1}")
            X_train, X_validate = splitter(current_X)

            best_model, best_score = fit_best_gaussian_hmm(
                X_train,
                X_validate,
                component_range=range(
                    layer_cfg.min_components, layer_cfg.max_components + 1
                ),
                n_fits=self.cfg.n_fits,
                covariance_type=layer_cfg.covariance_type,
                init_params=layer_cfg.init_params,
                tol=self.cfg.tol,
                evaluation_metric=self.evaluation_metric,
                log_prefix=f"[{self.name}] Layer {layer_idx + 1} ",
            )

            if best_model is None:
                logger.error(
                    f"[{self.name}] No model could be trained for Layer {layer_idx + 1}"
                )
                return None

            if best_score > self.best_score:
                self.best_score = best_score

            self.layers.append(best_model)

            posterior = best_model.predict_proba(current_X)
            current_X = np.hstack([original_X, posterior])

        return best_model

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
