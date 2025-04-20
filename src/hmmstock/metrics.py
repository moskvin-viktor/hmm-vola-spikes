import numpy as np
from omegaconf import OmegaConf



class EvaluationMetric:
    def evaluate(self, model, X_validate):
        raise NotImplementedError("Must implement `evaluate()` in subclass.")
    

class LogLikelihoodWithEntropy(EvaluationMetric):
    def __init__(self, config_path = "config/model.yaml"):
        self.cfg = OmegaConf.load(config_path)
        self.entropy_weight = self.cfg.get("entropy_weight", 3)

    def evaluate(self, model, X_validate):
        score = model.score(X_validate)
        states = model.predict(X_validate)
        state_counts = np.bincount(states, minlength=model.n_components)
        probs = state_counts / state_counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return score + self.entropy_weight * entropy
    
class BICMetric(EvaluationMetric):
    def evaluate(self, model, X_validate):
        log_likelihood = model.score(X_validate)
        n_samples, n_features = X_validate.shape
        n_components = model.n_components

        # Number of parameters:
        # Means: n_components * n_features
        # Covariances: n_components * n_features * (n_features + 1) / 2
        # Transition matrix: n_components * (n_components - 1) [excluding last column of each row]
        # Initial state probs: n_components - 1
        n_params = (
            n_components * n_features +  # means
            n_components * n_features * (n_features + 1) / 2 +  # full cov matrix
            n_components * (n_components - 1) +  # transition probabilities
            (n_components - 1)  # initial state probabilities
        )

        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        return -bic  # negate so higher is better (like log-likelihood)