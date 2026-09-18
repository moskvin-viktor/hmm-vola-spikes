import logging
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .artifact_store import ArtifactStore, ArtifactVersion
from .data.splitter import SplitConfig, train_val_split
from .metrics import LogLikelihoodWithEntropy
from .models import HMMModel, RegimeModel

logger = logging.getLogger(__name__)


def sanitize_ticker(ticker: str) -> str:
    """Sanitize ticker symbol by removing/replacing unwanted characters."""
    return ticker.replace("^", "").replace("/", "_")


class RegimeModelManager:
    """Trains one RegimeModel per ticker and writes a new versioned
    artifact set for the run.

    Model construction/training/prediction is delegated to the
    RegimeModel subclass (model_class); every call to train_all() writes
    a fresh artifacts/{model_name}/version_N/ via ArtifactStore -- no
    run ever overwrites a previous one.
    """

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        cfg: DictConfig,
        evaluation_metric=None,
        train_test_splitter=None,
        model_class: type[RegimeModel] = HMMModel,
    ):
        self.cfg = cfg

        self.data_dict = {
            sanitize_ticker(ticker): df for ticker, df in data_dict.items()
        }
        self.original_ticker_map = {
            sanitize_ticker(ticker): ticker for ticker in data_dict.keys()
        }

        self.evaluation_metric = (
            evaluation_metric()
            if evaluation_metric
            else (
                LogLikelihoodWithEntropy(
                    entropy_weight=self.cfg.get("entropy_weight", 3)
                )
            )
        )

        split_node = self.cfg.get("split")
        split_cfg = (
            cast(dict, OmegaConf.to_container(split_node, resolve=True))
            if split_node
            else {}
        )
        self.splitter = train_test_splitter or partial(
            train_val_split, config=SplitConfig(**split_cfg)
        )

        self.model_class = model_class
        self.model_name = model_class.__name__

        self.models: dict[str, RegimeModel] = {}
        self.states: dict[str, np.ndarray | pd.DataFrame | None] = {}
        self.state_labeled_data: dict[str, pd.DataFrame] = {}

        self.artifact_store = ArtifactStore(Path("artifacts"), self.model_name)

    def _build_model_config(self):
        raw = OmegaConf.to_container(self.cfg[self.model_name], resolve=True)
        return self.model_class.config_cls.model_validate(raw)

    def train_all(self) -> ArtifactVersion:
        """Train models for all tickers, writing a new artifact version."""
        version = self.artifact_store.new_version()
        version.write_config(self.cfg[self.model_name])

        config = self._build_model_config()
        for ticker, df in self.data_dict.items():
            original_ticker = self.original_ticker_map[ticker]

            X = df.to_numpy()
            model = self.model_class(ticker, X, config, self.evaluation_metric)
            fitted_model = model.fit(self.splitter)

            if not fitted_model:
                logger.warning(
                    f"No model fitted for {original_ticker} due to insufficient data or errors."
                )
                version.record_metric(ticker, fitted=False)
                continue

            logger.info(
                f"Training completed for {original_ticker} with {fitted_model.n_components} components."
            )

            self.models[ticker] = model
            self.states[ticker] = model.predict_states()
            version.record_metric(
                ticker,
                fitted=True,
                best_score=model.best_score,
                n_components=fitted_model.n_components,
            )
            version.write_model(ticker, model)

        version.flush_metrics()
        self.generate_state_labeled_data(version)

        for ticker in self.data_dict:
            self.write_transition_matrices(ticker, version)

        logger.info(f"Wrote {self.model_name} artifacts to {version.path}")
        return version

    def _get_states(self) -> dict[str, pd.DataFrame]:
        """Get predicted states for all tickers."""
        state_dict = {}
        for ticker, model_instance in self.models.items():
            states = model_instance.predict_states()
            if states is None:
                continue

            if isinstance(states, pd.DataFrame):
                state_df = states.copy()
            else:
                state_df = pd.DataFrame(
                    {"regime_layer0": states},
                    index=self.data_dict[ticker].index[-len(states) :],
                )

            state_df.index = self.data_dict[ticker].index[-len(state_df) :]
            state_dict[ticker] = state_df

        return state_dict

    def generate_state_labeled_data(self, version: ArtifactVersion):
        """Generate and save labeled datasets for this run's version."""
        state_dict = self._get_states()

        for ticker, df in self.data_dict.items():
            if ticker not in state_dict:
                continue

            merged_df = df.merge(
                state_dict[ticker], left_index=True, right_index=True, how="left"
            )
            self.state_labeled_data[ticker] = merged_df

            csv_path = version.write_regime_states(ticker, merged_df)
            logger.info(
                f"Saved labeled regime data for {self.original_ticker_map[ticker]} to {csv_path}"
            )

    def write_transition_matrices(self, ticker: str, version: ArtifactVersion):
        """Compute and save transition matrices for a ticker's version."""
        ticker = sanitize_ticker(ticker)
        model_instance = self.models.get(ticker)

        if not model_instance:
            logger.warning(
                f"No model found for {self.original_ticker_map.get(ticker, ticker)}."
            )
            return

        matrices = model_instance.transition_matrices()
        if not matrices:
            logger.warning(
                f"No model(s) available for {self.original_ticker_map.get(ticker, ticker)}."
            )
            return

        paths = version.write_transition_matrices(ticker, matrices)
        for layer_idx, path in enumerate(paths):
            logger.info(
                f"Saved transition matrix for {self.original_ticker_map[ticker]} (Layer {layer_idx}) to {path}"
            )
