import logging
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .data.splitter import SplitConfig, train_val_split
from .metrics import LogLikelihoodWithEntropy
from .model_store import ModelStore
from .models import HMMModel, RegimeModel
from .results_writer import ResultsWriter

logger = logging.getLogger(__name__)


def sanitize_ticker(ticker: str) -> str:
    """Sanitize ticker symbol by removing/replacing unwanted characters."""
    return ticker.replace("^", "").replace("/", "_")


class RegimeModelManager:
    """Trains one RegimeModel per ticker and writes its results to disk.

    Model construction/training/prediction is delegated to the
    RegimeModel subclass (model_class); persistence goes through
    ModelStore, CSV/transition-matrix export through ResultsWriter --
    this class only orchestrates the per-ticker loop.
    """

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        config_path: str,
        evaluation_metric=None,
        train_test_splitter=None,
        model_class: type[RegimeModel] = HMMModel,
    ):
        self.cfg = cast(DictConfig, OmegaConf.load(config_path))

        self.data_dict = {
            sanitize_ticker(ticker): df for ticker, df in data_dict.items()
        }
        self.original_ticker_map = {
            sanitize_ticker(ticker): ticker for ticker in data_dict.keys()
        }

        self.evaluation_metric = (
            evaluation_metric() if evaluation_metric else LogLikelihoodWithEntropy()
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

        results_dir = Path(f"results/{self.model_name}")
        self.store = ModelStore(
            results_dir / "saved_models", f"{self.model_name}_hmm.pkl"
        )
        self.writer = ResultsWriter(
            csv_dir=results_dir / "csvs",
            transition_matrices_dir=results_dir / "transition_matrices",
        )

    def _build_model_config(self):
        raw = OmegaConf.to_container(self.cfg[self.model_name], resolve=True)
        return self.model_class.config_cls.model_validate(raw)

    def train_all(self):
        """Train or load models for all tickers."""
        self.models = self.store.load()

        if self.models:
            logger.info(f"Loaded saved models for {len(self.models)} tickers.")
            for ticker, model in self.models.items():
                self.states[ticker] = model.predict_states()
                print(f"{model.best_score} for {self.model_name} and {ticker}.")
        else:
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
                    continue

                logger.info(
                    f"Training completed for {original_ticker} with {fitted_model.n_components} components."
                )

                self.models[ticker] = model
                self.states[ticker] = model.predict_states()
                print(f"Trained models for {len(self.models)} tickers.")
                print(f"{model.best_score} for {self.model_name} model and {ticker}.")

        self.store.save(self.models)
        self.generate_state_labeled_data()

        for ticker in self.data_dict:
            self.write_transition_matrices(ticker)

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

    def generate_state_labeled_data(self):
        """Generate and save labeled datasets."""
        state_dict = self._get_states()

        for ticker, df in self.data_dict.items():
            if ticker not in state_dict:
                continue

            merged_df = df.merge(
                state_dict[ticker], left_index=True, right_index=True, how="left"
            )
            self.state_labeled_data[ticker] = merged_df

            csv_path = self.writer.write_regime_states(ticker, merged_df)
            logger.info(
                f"Saved labeled regime data for {self.original_ticker_map[ticker]} to {csv_path}"
            )

    def write_transition_matrices(self, ticker: str):
        """Compute and save transition matrices for a ticker."""
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

        paths = self.writer.write_transition_matrices(ticker, matrices)
        for layer_idx, path in enumerate(paths):
            logger.info(
                f"Saved transition matrix for {self.original_ticker_map[ticker]} (Layer {layer_idx}) to {path}"
            )
