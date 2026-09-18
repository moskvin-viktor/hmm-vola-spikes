import numpy as np


class EvaluationMetric:
    def evaluate(self, model, X_train, X_validate):
        raise NotImplementedError("Must implement `evaluate()` in subclass.")


class LogLikelihoodWithEntropy(EvaluationMetric):
    def __init__(self, entropy_weight: float = 3):
        self.entropy_weight = entropy_weight

    def evaluate(self, model, X_train, X_validate):
        log_likelihood = model.score(X_validate)
        n_frames = len(X_validate)

        # Normalize log-likelihood by sequence length
        normalized_ll = log_likelihood / n_frames

        # Entropy of the state usage
        states = model.predict(X_validate)
        state_counts = np.bincount(states, minlength=model.n_components)
        probs = state_counts / state_counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return normalized_ll + self.entropy_weight * entropy


class BICMetric(EvaluationMetric):
    def evaluate(self, model, X_train, X_validate):
        # BIC penalizes likelihood against training-set fit, not held-out
        # data -- that's the whole point (it approximates held-out
        # performance from training data alone, without needing a
        # validation set), so it's scored on X_train. hmmlearn's model.bic()
        # follows the standard convention (lower is better); every
        # selection loop here does `if score > best_score`, so negate it.
        return -model.bic(X_train)


METRIC_CLASSES = {
    "LogLikelihoodWithEntropy": LogLikelihoodWithEntropy,
    "BICMetric": BICMetric,
}


def build_evaluation_metric(cfg) -> EvaluationMetric:
    """Builds the evaluation metric named by `cfg["evaluation_metric"]`
    (default "LogLikelihoodWithEntropy"), reading entropy_weight from the
    same cfg when applicable."""
    name = cfg.get("evaluation_metric", "LogLikelihoodWithEntropy")
    if name not in METRIC_CLASSES:
        raise ValueError(
            f"Unknown evaluation_metric {name!r}; expected one of {list(METRIC_CLASSES)}"
        )
    if name == "LogLikelihoodWithEntropy":
        return LogLikelihoodWithEntropy(entropy_weight=cfg.get("entropy_weight", 3))
    return METRIC_CLASSES[name]()
