import numpy as np
import logging
import os
import joblib
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
    def __init__(self, name: str, X: np.ndarray, config, evaluation_metric):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.model = None

    def fit(self, splitter : callable[np.array], n_fits : int, random_seed : int):
        if len(self.X) < 20:
            return None

        best_overall_score = -np.inf
        best_model = None

        X_train, X_validate = splitter(self.X)
        np.random.seed(random_seed)

        for n_components in range(2, self.cfg.max_components + 1):
            for idx in range(n_fits):
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.cfg.hmm_config.covariance_type,
                    random_state=idx,
                    init_params=self.cfg.hmm_config.init_params,
                    n_iter=self.cfg.hmm_config.n_iter,
                    tol=self.cfg.hmm_config.tol
                )
                try:
                    model.fit(X_train)
                    base_score = self.evaluation_metric.evaluate(model, X_validate)
                    if base_score > best_overall_score:
                        best_model = model
                        best_overall_score = base_score
                except Exception as e:
                    logger.warning(f"[{self.name}] HMM training failed with {n_components} components: {e}")

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

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, f"{self.name}_hmm.pkl"))

    def load_model(self, path):
        self.model = joblib.load(os.path.join(path, f"{self.name}_hmm.pkl"))