from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np
import pandas as pd
from .metrics import LogLikelihoodWithEntropy
from .data.datamanager import default_split
from pathlib import Path
import logging
import os
import joblib
from functools import partial
from typing import Any
from .models.factory import MarkovModelEnum, MarkovModelFactory
from .models.markov_model import MarkovModel
from .db_manager import DatabaseManager
from .result_processor import ModelResultProcessor

# Set up logging
LOGGING_DIR = "results/logs"
os.makedirs(LOGGING_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOGGING_DIR, "hmm_model.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sanitize_ticker(ticker: str) -> str:
    """Sanitize ticker symbol by removing/replacing unwanted characters."""
    return ticker.replace("^", "").replace("/", "_")


class RegimeModelManager:
    """A class to manage the training and evaluation of HMM models for multiple tickers."""

    def __init__(
        self,
        multiindexed_df: pd.DataFrame,
        config_path: str,
        model_type: MarkovModelEnum,
        evaluation_metric=None,
        train_test_splitter=None,
    ):
        cfg = OmegaConf.load(config_path)
        if not isinstance(cfg, DictConfig):
            raise TypeError("Configuration file must be a dictionary.")
        self._cfg: DictConfig = cfg
        self._data = multiindexed_df
        self._tickers = self._data.index.get_level_values("ticker").unique().tolist()

        self._evaluation_metric = (
            evaluation_metric() if evaluation_metric else LogLikelihoodWithEntropy()
        )

        self._splitter_func = train_test_splitter or default_split
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        split_cfg_from_yaml = (
            cfg_dict.get("split", {}) if isinstance(cfg_dict, dict) else {}
        )
        split_cfg = {
            "train_ratio": split_cfg_from_yaml.get("train_size", 0.8),
            "shuffle": split_cfg_from_yaml.get("shuffle", False),
        }
        self._splitter = partial(self._splitter_func, split_cfg=split_cfg)

        self._model_type = model_type
        self._model_name = self._model_type.value

        self._models: dict[str, MarkovModel] = {}

        self._db_manager = DatabaseManager()

    def train_all(self):
        """Train or load HMM models for all tickers."""

        for ticker in self._tickers:
            logger.info(f"Starting training for {ticker}.")
            df: pd.DataFrame = self._data.loc[ticker]

            X = df.to_numpy()
            model_config = self._cfg.get(self._model_name)
            model_config_dict = OmegaConf.to_container(model_config, resolve=True)

            if not isinstance(model_config_dict, dict):
                logger.error(
                    f"Model config for {self._model_name} is not a dictionary."
                )
                continue

            model = MarkovModelFactory.create_model(
                model_type=self._model_type,
                name=ticker,
                X=X,
                config=model_config_dict,
                evaluation_metric=self._evaluation_metric,
            )
            fitted_model = model.fit(self._splitter)

            if not fitted_model:
                logger.warning(
                    f"No model fitted for {ticker} due to insufficient data or errors."
                )
                continue

            logger.info(
                f"Training completed for {ticker} with {fitted_model.n_components} components."
            )

            self._models[ticker] = model

            # Process and save results
            processor = ModelResultProcessor(model, df)
            state_labeled_data = processor.get_state_labeled_data()
            if not state_labeled_data.empty:
                logger.info(f"Saving regime states for {ticker}.")
                self._db_manager.save_regime_states(ticker, state_labeled_data)
                logger.info(f"Finished saving regime states for {ticker}.")

            transition_matrices = processor.get_transition_matrices()
            for i, matrix in enumerate(transition_matrices):
                logger.info(f"Saving transition matrix for {ticker}, layer {i}.")
                self._db_manager.save_transition_matrix(ticker, i, matrix)
                logger.info(
                    f"Finished saving transition matrix for {ticker}, layer {i}."
                )

            logger.info(f"Saving model result for {ticker}.")
            self._db_manager.save_model_result(ticker, model, self._model_name)
            logger.info(f"Finished saving model result for {ticker}.")

    # def expected_steps_before_change(self, ticker: str) -> pd.Series | None:
    #     """Compute expected steps before switching state."""
    #     ticker = sanitize_ticker(ticker)
    #     model_instance = self._models.get(ticker)

    #     if not model_instance or not model_instance.model:
    #         logger.warning(f"No model for {ticker}.")
    #         return None

    #     transmat = model_instance.model.transmat_
    #     diag = np.diag(transmat)
    #     expected_steps = 1 / (1 - diag + 1e-10)

    #     return pd.Series(
    #         expected_steps,
    #         index=[f"VS{i}" for i in range(model_instance.model.n_components)],
    #         name="ExpectedStepsInState",
    #     )

    # def load_data(self, ticker: str) -> dict[str, pd.DataFrame] | None:
    #     """Loads processed data for a given ticker from the database."""
    #     return self._db_manager.load_data(ticker)
