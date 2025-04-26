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
    return ticker.replace("^", "").replace("/", "_")


class RegimeModelManager:
    def __init__(self, data_dict: Dict[str, pd.DataFrame], config_path: str,
                 evaluation_metric=None, train_test_splitter=None,
                 model_class: Type[HMMModel] = HMMModel):
        self.cfg = OmegaConf.load(config_path)

        self.data_dict = data_dict
        self.evaluation_metric = evaluation_metric() if evaluation_metric else LogLikelihoodWithEntropy()
        self.splitter = train_test_splitter or default_split
        self.model_class = model_class

        self.state_labeled_data = {}
        self.models: Dict[str, HMMModel] = {}
        self.states = {}

        self.model_name = self.model_class.__name__
        self.results_dir = f"results_{self.model_name}"
        os.makedirs(os.path.join(self.results_dir, "csv"), exist_ok=True)

    def _compute_log_returns(self, df):
        if "normalized_returns" in df.columns:
            return df["normalized_returns"].dropna().values.reshape(-1, 1)
        else:
            raise ValueError("Returns column missing in DataFrame.")

    def train_all(self):
        for ticker, df in self.data_dict.items():
            cleaned_ticker = sanitize_ticker(ticker)
            logger.info(f"Training {self.model_name} for {ticker} (1 to {self.cfg.max_components} states)...")
            try:
                X = self._compute_log_returns(df)
                model_instance = self.model_class(cleaned_ticker, X, self.cfg, self.evaluation_metric)
                fitted_model = model_instance.fit(self.splitter, self.cfg.n_fits, self.cfg.random_seed)
                self.models[cleaned_ticker] = model_instance

                if fitted_model:
                    states = model_instance.predict_states()
                    self.states[cleaned_ticker] = states
                    logger.info(f"Training completed for {ticker} with {fitted_model.n_components} components.")
                else:
                    logger.warning(f"No model fitted for {ticker} due to insufficient data or training errors.")
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
        self.generate_state_labeled_data()

    def _get_states(self):
        state_dict = {}
        for ticker, states in self.states.items():
            if states is not None:
                aligned_index = self.data_dict[ticker].index[-len(states):]
                state_dict[ticker] = pd.Series(states, index=aligned_index)
        return pd.DataFrame(state_dict)

    def generate_state_labeled_data(self):
        states_df = self._get_states()

        for ticker, df in self.data_dict.items():
            cleaned_ticker = sanitize_ticker(ticker)
            if cleaned_ticker in states_df:
                state_series = pd.Series(states_df[cleaned_ticker], index=df.index[-len(df):])
                aligned_states = state_series.rename("regime_state").to_frame()
                merged = df.merge(aligned_states, left_index=True, right_index=True, how="left")
                self.state_labeled_data[cleaned_ticker] = merged
                merged.to_csv(os.path.join(self.results_dir, "csv", f"{cleaned_ticker}_regime_states.csv"))
                logger.info(f"Saved labeled data for {cleaned_ticker} to {self.results_dir}/csv/{cleaned_ticker}_regime_states.csv")

    def get_transition_matrix(self, ticker):
        cleaned_ticker = sanitize_ticker(ticker)
        model_instance = self.models.get(cleaned_ticker)
        if model_instance and model_instance.model:
            trans_df = pd.DataFrame(
                model_instance.model.transmat_,
                index=[f"VS{i}" for i in range(model_instance.model.n_components)],
                columns=[f"VS{i}" for i in range(model_instance.model.n_components)]
            )
            csv_path = os.path.join(self.results_dir, "csv", f"{cleaned_ticker}_transition_matrix.csv")
            trans_df.to_csv(csv_path)
            logger.info(f"Saved transition matrix for {cleaned_ticker} to {csv_path}")
        else:
            logger.warning(f"No model found for {ticker}.")

    def expected_steps_before_change(self, ticker):
        cleaned_ticker = sanitize_ticker(ticker)
        model_instance = self.models.get(cleaned_ticker)
        if not model_instance or not model_instance.model:
            logger.warning(f"No model for {ticker}.")
            return None

        transmat = model_instance.model.transmat_
        diag = np.diag(transmat)
        expected_steps = 1 / (1 - diag + 1e-10)
        return pd.Series(expected_steps, index=[f"VS{i}" for i in range(model_instance.model.n_components)],
                         name="ExpectedStepsInState")
