import numpy as np
import logging
import os
from hmmlearn import hmm

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


class HMMModel:
    '''A class to encapsulate the HMM model training and prediction process.
    We use calssical HMM from hmmlearn library.
    The model is trained on the volatility of the log returns of the stock prices and a market proxy volatiltiy
    '''
    def __init__(self, name: str, X: np.ndarray, config, evaluation_metric):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.model = None
        self.is_layered = False
        self.best_score = -np.inf

    def fit(self, splitter : callable) -> None | hmm.GaussianHMM:
        '''Fit the HMM model to the data using the specified splitter and number of fits.
        The configuration is used to set the number of components, covariance type, and other parameters.
        The function will return the best model based on the evaluation metric.
        The function will return None if the model cannot be trained.
        The function will return the best model if the model is trained successfully.
        '''
        if len(self.X) < 20:
            return None

        best_overall_score = -np.inf
        best_model = None

        X_train, X_validate = splitter(self.X)
        np.random.seed(self.cfg.random_seed)
        for n_components in range(2, self.cfg.max_components + 1):
            for idx in range(self.cfg.n_fits):
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.cfg.covariance_type,
                    random_state=idx,
                    init_params=self.cfg.init_params,
                    n_iter=self.cfg.n_fits,
                    tol=self.cfg.tol
                )
                try:
                    model.fit(X_train)
                    base_score = self.evaluation_metric.evaluate(model, X_validate)
                    if base_score > best_overall_score:
                        best_model = model
                        best_overall_score = base_score
                except Exception as e:
                    logger.warning(f"[{self.name}] HMM training failed with {n_components} components: {e}")
        if best_overall_score > self.best_score:
            self.best_score = best_overall_score
        self.model = best_model
        return best_model

    def predict_states(self):
        if not self.model:
            return None
        raw_states = self.model.predict(self.X)
        return self._relabel_states_by_volatility(raw_states)

    def _relabel_states_by_volatility(self, original_states):
        state_vols = []
        for state in range(self.model.n_components):
            state_obs = self.X[original_states == state]
            vol = np.std(state_obs)
            state_vols.append((state, vol))

        sorted_states = sorted(state_vols, key=lambda x: x[1])
        state_map = {old: new for new, (old, _) in enumerate(sorted_states)}
        return np.vectorize(state_map.get)(original_states)