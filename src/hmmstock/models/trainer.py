import logging

import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)


def fit_best_gaussian_hmm(
    X_train: np.ndarray,
    X_validate: np.ndarray,
    *,
    component_range: range,
    n_fits: int,
    covariance_type: str,
    init_params: str,
    tol: float,
    evaluation_metric,
    log_prefix: str = "",
) -> tuple[hmm.GaussianHMM | None, float]:
    """Grid search over n_components x random restarts.

    Fits a GaussianHMM for every (n_components, seed) combination in
    `component_range` x `range(n_fits)`, scores each on `X_validate` via
    `evaluation_metric`, and returns the best-scoring model. A fit that
    raises (e.g. singular covariance for a given seed) is skipped, not
    fatal to the search.

    Returns (best_model, best_score); best_model is None and best_score is
    -inf if every fit failed.
    """
    best_model = None
    best_score = -np.inf

    for n_components in component_range:
        for seed in range(n_fits):
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=seed,
                init_params=init_params,
                n_iter=n_fits,
                tol=tol,
            )
            try:
                model.fit(X_train)
                score = evaluation_metric.evaluate(model, X_validate)
                if score > best_score:
                    best_model = model
                    best_score = score
                    print(
                        f"{log_prefix}components={n_components} seed={seed} "
                        f"score={score:.4f}"
                    )
            except Exception as e:
                logger.warning(
                    f"{log_prefix}fit failed (components={n_components}, "
                    f"seed={seed}): {e}"
                )

    return best_model, best_score
