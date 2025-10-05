import numpy as np
import logging
import os
from hmmlearn import hmm
from omegaconf import OmegaConf
from .markov_model import MarkovModel

from typing_extensions import override
from collections.abc import Callable

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


class HMMModel(MarkovModel):
    """A class to encapsulate the HMM model training and prediction process.
    We use calssical HMM from hmmlearn library.
    The model is trained on the volatility of the log returns of the stock prices and a market proxy volatiltiy
    """

    is_layered: bool = False
    name: str = "HMMModel"

    def __init__(self, X: np.ndarray, config: dict, evaluation_metric) -> None:

        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.model = None

        self.normalized_ll = None
        self.entropy = None
        self.random_state = None

    @override
    def fit(
        self, splitter: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    ) -> hmm.GaussianHMM | None:
        """Fit the HMM model to the data using the specified splitter and number of fits.
        The function returns the best model based on the evaluation metric or None if fitting fails.
        """
        if len(self.X) < 20:
            logger.warning(f"[{HMMModel.name}] Not enough data to train. Skipping.")
            return None

        best_overall_score = -np.inf
        best_model = None

        X_train, X_validate = splitter(self.X)
        np.random.seed(self.cfg["random_seed"])
        print(X_train.shape, X_validate.shape)
        for n_components in range(2, self.cfg["max_components"] + 1):
            for idx in range(self.cfg["n_fits"]):
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.cfg["covariance_type"],
                    random_state=idx,
                    init_params=self.cfg["init_params"],
                    n_iter=self.cfg["n_fits"],
                    tol=self.cfg["tol"],
                )
                try:
                    model.fit(X_train)
                    log_likelihood = model.score(X_validate)
                    states = model.predict(X_validate)
                    state_counts = np.bincount(states, minlength=model.n_components)
                    probs = state_counts / state_counts.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    normalized_ll = log_likelihood / len(X_validate)

                    base_score = self.evaluation_metric.evaluate(model, X_validate)

                    if base_score > best_overall_score:
                        logger.info(
                            f"[{HMMModel.name}] Components: {n_components}, Seed: {idx}, "
                            f"Norm LL: {normalized_ll:.4f}, Entropy: {entropy:.4f}, "
                            f"Score: {base_score:.4f}"
                        )
                        best_model = model
                        best_overall_score = base_score
                        self.normalized_ll = normalized_ll
                        self.entropy = entropy
                        self.random_state = idx
                except Exception as e:
                    logger.warning(
                        f"[{HMMModel.name}] HMM training failed with {n_components} components "
                        f"and seed {idx}: {e}"
                    )

        if best_overall_score > self.best_score:
            self.best_score = best_overall_score
        self.model = best_model
        return self.model

    @override
    def predict_states(self):
        if not self.model:
            return None
        raw_states = self.model.predict(self.X)
        return self._relabel_states_by_volatility(raw_states)

    def _relabel_states_by_volatility(self, original_states):
        if self.model is None:
            logger.warning("Cannot relabel states: model is not fitted.")
            return None
        state_vols = []
        for state in range(self.model.n_components):
            state_obs = self.X[original_states == state]
            vol = np.std(state_obs)
            state_vols.append((state, vol))

        sorted_states = sorted(state_vols, key=lambda x: x[1])
        state_map = {old: new for new, (old, _) in enumerate(sorted_states)}
        return np.vectorize(state_map.get)(original_states)
