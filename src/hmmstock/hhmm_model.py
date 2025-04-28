import numpy as np
import logging
import os
import joblib
from hmmlearn import hmm
import pandas as pd
#  Set up logging
logging_dir = "results/logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logging_dir, "hmm_model.log"),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HierarchicalHMMModel:
    '''
    Hierarchical HMM model: top-level HMM decides regime, sub-HMMs model observations inside regimes.
    '''
    def __init__(self, name: str, X: np.ndarray, config, evaluation_metric):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.top_model = None
        self.sub_models = {}
        self.is_layered = False
        self.path = os.path.join(f"results_{self.name}", "models", f"{self.name}_hierarchical_hmm")

    def fit(self, splitter: callable):
        '''Fit a Hierarchical HMM model.'''

        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train")
            return None

        np.random.seed(self.cfg.random_seed)
        X_train, X_validate = splitter(self.X)

        # 1. Fit the top-level HMM
        logger.info(f"[{self.name}] Training Top-level HMM")
        best_score = -np.inf
        best_top_model = None

        for n_components in range(self.cfg.top_layer.min_components, self.cfg.top_layer.max_components + 1):
            for fit_idx in range(self.cfg.n_fits):
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.cfg.top_layer.covariance_type,
                    random_state=fit_idx,
                    init_params=self.cfg.top_layer.init_params,
                    n_iter=self.cfg.n_fits,
                    tol=self.cfg.tol
                )
                try:
                    model.fit(X_train)
                    score = self.evaluation_metric.evaluate(model, X_validate)
                    if score > best_score:
                        best_score = score
                        best_top_model = model
                except Exception as e:
                    logger.warning(f"[{self.name}] Top-level HMM training failed: {e}")

        if best_top_model is None:
            logger.error(f"[{self.name}] No Top-level model could be trained")
            return None

        self.top_model = best_top_model

        # 2. Assign data points to top-level states
        top_states = self.top_model.predict(self.X)

        # 3. Train sub-HMMs for each top-level state
        for top_state in np.unique(top_states):
            logger.info(f"[{self.name}] Training Sub-HMM for Top State {top_state}")
            sub_X = self.X[top_states == top_state]

            if len(sub_X) < 10:
                logger.warning(f"[{self.name}] Not enough samples for Sub-HMM in Top State {top_state}")
                continue

            best_sub_score = -np.inf
            best_sub_model = None

            for n_components in range(self.cfg.sub_layer.min_components, self.cfg.sub_layer.max_components + 1):
                for fit_idx in range(self.cfg.n_fits):
                    sub_model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=self.cfg.sub_layer.covariance_type,
                        random_state=fit_idx,
                        init_params=self.cfg.sub_layer.init_params,
                        n_iter=self.cfg.n_fits,
                        tol=self.cfg.tol
                    )
                    try:
                        sub_model.fit(sub_X)
                        sub_score = self.evaluation_metric.evaluate(sub_model, sub_X)
                        if sub_score > best_sub_score:
                            best_sub_score = sub_score
                            best_sub_model = sub_model
                    except Exception as e:
                        logger.warning(f"[{self.name}] Sub-HMM training failed for Top State {top_state}: {e}")

            if best_sub_model:
                self.sub_models[top_state] = best_sub_model
            else:
                logger.error(f"[{self.name}] No Sub-HMM could be trained for Top State {top_state}")

        return self.top_model

    def predict_states(self):
        '''Predict top-level and sub-level regimes.'''
        if self.top_model is None or not self.sub_models:
            return None

        top_states = self.top_model.predict(self.X)
        all_states = {"top_level_state": top_states}
        sub_states = []

        for idx, top_state in enumerate(top_states):
            sub_model = self.sub_models.get(top_state)

            if sub_model is None:
                sub_states.append(np.nan)  # If no submodel available, fill with NaN
            else:
                # Predict based on each individual observation
                obs = self.X[idx:idx+1]
                sub_state = sub_model.predict(obs)[0]
                sub_states.append(sub_state)

        all_states["sub_level_state"] = sub_states
        return pd.DataFrame(all_states, index=pd.RangeIndex(len(self.X)))