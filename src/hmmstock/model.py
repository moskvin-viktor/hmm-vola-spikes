from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from .metrics import LogLikelihoodWithEntropy, BICMetric
from .data.datamanager import default_split
from .path_manager import PathManager
from pathlib import Path    
import logging
import os
import joblib
from .hmm_model import HMMModel
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
    """Sanitize ticker symbol by removing/replacing unwanted characters."""
    return ticker.replace("^", "").replace("/", "_")


class RegimeModelManager:
    '''A class to manage the training and evaluation of HMM models for multiple tickers.
    It handles the training, evaluation, and saving of models, as well as the generation of state-labeled data.
    The class is designed to work with a dictionary of dataframes, where each dataframe corresponds to a ticker.
    The class also provides methods to compute transition matrices and expected steps before state changes.
    '''
    def __init__(self, data_dict: dict[str, pd.DataFrame], config_path: str,
                 evaluation_metric=None, train_test_splitter=None,
                 model_class: type[HMMModel] = HMMModel):
        self.cfg = OmegaConf.load(config_path)

        self.data_dict = {sanitize_ticker(ticker): df for ticker, df in data_dict.items()}
        self.original_ticker_map = {sanitize_ticker(ticker): ticker for ticker in data_dict.keys()}

        self.evaluation_metric = evaluation_metric() if evaluation_metric else LogLikelihoodWithEntropy()
        self.splitter = train_test_splitter or default_split
        self.model_class = model_class
        self.name = self.model_class.__name__

        self.models: dict[str, HMMModel] = {}
        self.states: dict[str, np.ndarray] = {}
        self.state_labeled_data: dict[str, pd.DataFrame] = {}

        self.model_name = self.model_class.__name__
        self.results_dir = Path(f"results/{self.model_name}")
        self.csv_dir = self.results_dir / "csvs"
        self.logs_dir = self.results_dir / "logs"
        self.saved_models_dir = self.results_dir / "saved_models"
        self.transition_matrices_dir = self.results_dir / "transition_matrices"

        # Create folders
        for folder in [self.csv_dir, self.logs_dir, self.saved_models_dir, self.transition_matrices_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def train_all(self):
        """Train or load HMM models for all tickers."""
        model_path = self.saved_models_dir
        self.load_model(model_path)
        if self.models:
            logger.info(f"Loaded saved models for {len(self.models)} tickers.")
            for ticker, model in self.models.items():
                original_ticker = self.original_ticker_map[ticker]
                self.states[ticker] = model.predict_states()
                print(f"{model.best_score} for {self.model_name} and {ticker}.")
                
        else:
            for ticker, df in self.data_dict.items():
                original_ticker = self.original_ticker_map[ticker]
                

                X = df.to_numpy()
                model = self.model_class(ticker, X, self.cfg[self.model_name], self.evaluation_metric)
                fitted_model = model.fit(self.splitter)

                if not fitted_model:
                    logger.warning(f"No model fitted for {original_ticker} due to insufficient data or errors.")
                    continue

                logger.info(f"Training completed for {original_ticker} with {fitted_model.n_components} components.")

                # Register model and states
                self.models[ticker] = model
                self.states[ticker] = model.predict_states()
                print(f"Trained models for {len(self.models)} tickers.")
                print(f"{model.best_score} for {self.model_name} model and {ticker}.")
        self.save_model(model_path)
        self.generate_state_labeled_data()

        # Compute transition matrices for all tickers
        for ticker in self.data_dict:
            self.get_transition_matrix(ticker)

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
                    {f"regime_layer0": states},
                    index=self.data_dict[ticker].index[-len(states):]
                )

            state_df.index = self.data_dict[ticker].index[-len(state_df):]
            state_dict[ticker] = state_df

        return state_dict

    def generate_state_labeled_data(self):
        """Generate and save labeled datasets."""
        state_dict = self._get_states()

        for ticker, df in self.data_dict.items():
            if ticker not in state_dict:
                continue

            state_info = state_dict[ticker]
            merged_df = df.merge(state_info, left_index=True, right_index=True, how="left")
            self.state_labeled_data[ticker] = merged_df

            ticker_dir = self.csv_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            csv_path = ticker_dir / "regime_states.csv"
            merged_df.to_csv(csv_path)
            logger.info(f"Saved labeled regime data for {self.original_ticker_map[ticker]} to {csv_path}")

    def get_transition_matrix(self, ticker: str):
        """Compute and save transition matrix for a ticker."""
        ticker = sanitize_ticker(ticker)
        model_instance = self.models.get(ticker)

        if not model_instance:
            logger.warning(f"No model found for {self.original_ticker_map.get(ticker, ticker)}.")
            return

        models = model_instance.models if getattr(model_instance, "is_layered", True) else [model_instance.model]

        if not models or any(m is None for m in models):
            logger.warning(f"No model(s) available for {self.original_ticker_map.get(ticker, ticker)}.")
            return

        for layer_idx, model in enumerate(models):
            trans_df = pd.DataFrame(
                model.transmat_,
                index=[f"VS{layer_idx}_{i}" for i in range(model.n_components)],
                columns=[f"VS{layer_idx}_{i}" for i in range(model.n_components)]
            )

            self.transition_matrices_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.transition_matrices_dir / f"{ticker}_transition_matrix_layer{layer_idx}.csv"
            trans_df.to_csv(csv_path)
            logger.info(f"Saved transition matrix for {self.original_ticker_map[ticker]} (Layer {layer_idx}) to {csv_path}")

    def expected_steps_before_change(self, ticker: str) -> pd.Series:
        """Compute expected steps before switching state."""
        ticker = sanitize_ticker(ticker)
        model_instance = self.models.get(ticker)

        if not model_instance or not model_instance.model:
            logger.warning(f"No model for {self.original_ticker_map.get(ticker, ticker)}.")
            return None

        transmat = model_instance.model.transmat_
        diag = np.diag(transmat)
        expected_steps = 1 / (1 - diag + 1e-10)

        return pd.Series(
            expected_steps,
            index=[f"VS{i}" for i in range(model_instance.model.n_components)],
            name="ExpectedStepsInState"
        )
    
    def save_model(self, path: Path):
        """Save model(s) to the specified path."""
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.models, path / f"{self.name}_hmm.pkl")

    def load_model(self, path: Path):
        """Load model(s) from the specified path."""
        try:
            self.models = joblib.load(path / f"{self.name}_hmm.pkl")
        except FileNotFoundError:
            return {}