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
class LayeredHMMModel:
    '''
    Layered HMM model: trains multiple HMMs sequentially.
    Layer 1 on raw features, Layer 2 on posterior probabilities from Layer 1, etc.
    '''
    def __init__(self, name: str, X: np.ndarray, config, evaluation_metric):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.models = []
        self.is_layered = True
        self.path = os.path.join(f"results_{self.name}", "models", f"{self.name}_layered_hmm")

    def fit(self, splitter: callable):
        '''Fit a Layered HMM model using the provided data splitter.'''

        if len(self.X) < 20:
            logger.warning(f"[{self.name}] Not enough data to train")
            return None

        current_X = self.X
        np.random.seed(self.cfg.random_seed)

        for layer_idx in range(self.cfg.num_layers):
            logger.info(f"[{self.name}] Training Layer {layer_idx+1}")

            best_overall_score = -np.inf
            best_model = None

            layer_cfg = self.cfg.layers[layer_idx]
            X_train, X_validate = splitter(current_X)

            for n_components in range(layer_cfg.min_components, layer_cfg.max_components + 1):
                for fit_idx in range(self.cfg.n_fits):
                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=layer_cfg.covariance_type,
                        random_state=fit_idx,
                        init_params=layer_cfg.init_params,
                        n_iter=self.cfg.n_fits, 
                        tol=self.cfg.tol  
                    )
                    try:
                        model.fit(X_train)
                        score = self.evaluation_metric.evaluate(model, X_validate)
                        if score > best_overall_score:
                            best_model = model
                            best_overall_score = score
                    except Exception as e:
                        logger.warning(f"[{self.name}] Layer {layer_idx+1} training failed: {e}")

            if best_model is None:
                logger.error(f"[{self.name}] No model could be trained for Layer {layer_idx+1}")
                return None

            self.models.append(best_model)
            current_X = best_model.predict_proba(current_X)

        return best_model

    def predict_states(self):
        '''Predict regimes for each layer.'''
        if not self.models:
            return None

        current_X = self.X
        all_layer_states = {}

        for idx, model in enumerate(self.models):
            raw_states = model.predict(current_X)
            relabeled_states = self._relabel_states_by_volatility(raw_states, model, current_X)
            all_layer_states[f"regime_layer{idx}"] = relabeled_states

            # For next layer input
            if idx < len(self.models) - 1:
                current_X = model.predict_proba(current_X)

        # Return as a DataFrame
        return pd.DataFrame(all_layer_states, index=pd.RangeIndex(len(self.X))[-len(relabeled_states):])

    def _relabel_states_by_volatility(self, original_states, model, X_layer):
        '''Relabel states based on volatility: from low volatility (0) to high (n-1).'''
        state_vols = []
        for state in range(model.n_components):
            state_obs = X_layer[original_states == state]
            vol = np.std(state_obs)
            state_vols.append((state, vol))

        sorted_states = sorted(state_vols, key=lambda x: x[1])
        state_map = {old: new for new, (old, _) in enumerate(sorted_states)}
        return np.vectorize(state_map.get)(original_states)