from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from .metrics import LogLikelihoodWithEntropy
from .data.datamanager import default_split
import joblib
import logging
import os
from .hmm_model import HMMModel
from typing import Dict, Type
# Set up logging
logging_dir = "results/logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logging_dir, "hmm_model.log"),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def sanitize_ticker(ticker: str) -> str:
    """Sanitizes a ticker symbol by removing or replacing unwanted characters."""
    return ticker.replace("^", "").replace("/", "_")


class RegimeModelManager:
    def __init__(self, data_dict: Dict[str, pd.DataFrame], config_path: str,
                 evaluation_metric=None, train_test_splitter=None,
                 model_class: Type[HMMModel] = HMMModel):
        self.cfg = OmegaConf.load(config_path)

        # Sanitize tickers once at the start
        self.data_dict = {sanitize_ticker(ticker): df for ticker, df in data_dict.items()}
        self.original_ticker_map = {sanitize_ticker(ticker): ticker for ticker in data_dict.keys()}

        self.evaluation_metric = evaluation_metric() if evaluation_metric else LogLikelihoodWithEntropy()
        self.splitter = train_test_splitter or default_split
        self.model_class = model_class

        self.models: Dict[str, HMMModel] = {}
        self.states: Dict[str, np.ndarray] = {}
        self.state_labeled_data: Dict[str, pd.DataFrame] = {}

        self.model_name = self.model_class.__name__
        self.results_dir = f"results_{self.model_name}"
        os.makedirs(os.path.join(self.results_dir, "csv"), exist_ok=True)

    def train_all(self):
        """Trains the model for all tickers in the dataset."""
        for ticker, df in self.data_dict.items():
            original_ticker = self.original_ticker_map[ticker]  # Retrieve original for logging
            
            # logger.info(f"Training {self.model_name} for {original_ticker} (1 to {self.cfg[self.model_name].max_components} states)...")
            
            X = df.to_numpy()
            model_instance = self.model_class(ticker, X, self.cfg[self.model_name], self.evaluation_metric)
            fitted_model = model_instance.fit(self.splitter)
            self.models[ticker] = model_instance

            if fitted_model:
                self.states[ticker] = model_instance.predict_states()
                logger.info(f"Training completed for {original_ticker} with {fitted_model.n_components} components.")
                # Save the model
                model_instance.save_model()
            else:
                logger.warning(f"No model fitted for {original_ticker} due to insufficient data or training errors.")

        self.generate_state_labeled_data()

    def _get_states(self) -> pd.DataFrame:
        """Returns a DataFrame with state sequences for each ticker."""
        state_dict = {}

        for ticker, model_instance in self.models.items():
            states = model_instance.predict_states()

            if states is None:
                continue

            if isinstance(states, pd.DataFrame):
                # Layered model (multi-columns: regime_layer0, regime_layer1, ...)
                state_df = states.copy()
            else:
                # Classic model (single array of states)
                state_df = pd.DataFrame({f"regime_layer0": states},
                                        index=self.data_dict[ticker].index[-len(states):])

            state_df.index = self.data_dict[ticker].index[-len(state_df):]
            state_dict[ticker] = state_df

        return state_dict

    def generate_state_labeled_data(self):
        """Generates and saves state-labeled datasets for all tickers."""
        state_dict = self._get_states()

        for ticker, df in self.data_dict.items():
            if ticker in state_dict:
                state_info = state_dict[ticker]
                merged_df = df.merge(state_info, left_index=True, right_index=True, how="left")
                self.state_labeled_data[ticker] = merged_df

                # NEW: Create a subfolder per ticker
                ticker_dir = os.path.join(self.results_dir, "csv", ticker)
                os.makedirs(ticker_dir, exist_ok=True)

                csv_path = os.path.join(ticker_dir, "regime_states.csv")
                merged_df.to_csv(csv_path)
                logger.info(f"Saved labeled regime data for {self.original_ticker_map[ticker]} to {csv_path}")

    def get_transition_matrix(self, ticker: str):
        """Computes and saves the transition matrix (one per layer if layered)."""
        ticker = sanitize_ticker(ticker)
        model_instance = self.models.get(ticker)

        if not model_instance:
            logger.warning(f"No model found for {self.original_ticker_map.get(ticker, ticker)}.")
            return

        if getattr(model_instance, "is_layered", True):
            models = model_instance.models
        else:
            models = [model_instance.model]

        if not models or any(m is None for m in models):
            logger.warning(f"No model(s) available for {self.original_ticker_map.get(ticker, ticker)}.")
            return

        for layer_idx, model in enumerate(models):
            trans_df = pd.DataFrame(
                model.transmat_,
                index=[f"VS{layer_idx}_{i}" for i in range(model.n_components)],
                columns=[f"VS{layer_idx}_{i}" for i in range(model.n_components)]
            )

            csv_path = os.path.join(self.results_dir, "csv", f"{ticker}_transition_matrix_layer{layer_idx}.csv")
            trans_df.to_csv(csv_path)
            logger.info(f"Saved transition matrix for {self.original_ticker_map[ticker]} (Layer {layer_idx}) to {csv_path}")

    def expected_steps_before_change(self, ticker: str) -> pd.Series:
        """Computes expected steps before changing states for a given model."""
        ticker = sanitize_ticker(ticker)
        model_instance = self.models.get(ticker)

        if not model_instance or not model_instance.model:
            logger.warning(f"No model for {self.original_ticker_map.get(ticker, ticker)}.")
            return None

        transmat = model_instance.model.transmat_
        diag = np.diag(transmat)
        expected_steps = 1 / (1 - diag + 1e-10)  # Avoid division by zero

        return pd.Series(expected_steps, index=[f"VS{i}" for i in range(model_instance.model.n_components)],
                         name="ExpectedStepsInState")
