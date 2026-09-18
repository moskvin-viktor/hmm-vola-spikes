import logging

import numpy as np
from hmmlearn import hmm

from hmmstock.data.splitter import adaptive_n_splits, walk_forward_splits

logger = logging.getLogger(__name__)


def cv_score_gaussian_hmm(
    X: np.ndarray,
    *,
    n_splits: int,
    n_components: int,
    seed: int,
    covariance_type: str,
    init_params: str,
    n_iter: int,
    tol: float,
    evaluation_metric,
    log_prefix: str = "",
) -> float:
    """Average `evaluation_metric` score for one (n_components, seed)
    config across `n_splits` walk-forward CV folds of X: refit fresh on
    each fold's own training slice, score on that fold's validation slice
    (chronologically after it, never before). Returns -inf if every
    fold's fit failed.
    """
    scores = []
    for X_train, X_val in walk_forward_splits(X, n_splits):
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=seed,
            init_params=init_params,
            n_iter=n_iter,
            tol=tol,
        )
        try:
            model.fit(X_train)
            scores.append(evaluation_metric.evaluate(model, X_train, X_val))
        except Exception as e:
            logger.warning(
                f"{log_prefix}CV fold failed (components={n_components}, seed={seed}): {e}"
            )

    return float(np.mean(scores)) if scores else -np.inf


def select_best_gaussian_hmm(
    X: np.ndarray,
    *,
    component_range: range,
    n_fits: int,
    n_splits: int,
    covariance_type: str,
    init_params: str,
    n_iter: int,
    tol: float,
    evaluation_metric,
    log_prefix: str = "",
) -> tuple[int | None, int | None, float]:
    """Grid search over n_components x random-seed restarts, each scored
    by cv_score_gaussian_hmm. `n_splits` is capped down (adaptive_n_splits)
    when X is too small to support it -- sklearn's TimeSeriesSplit raises
    outright if n_splits is too large for the data, rather than just
    scoring badly, so this is a correctness guard, not just a quality one.
    Returns (best_n_components, best_seed, best_cv_score); (None, None,
    -inf) if every candidate failed on every fold.
    """
    best_n_components, best_seed, best_score = None, None, -np.inf
    n_splits = adaptive_n_splits(len(X), n_splits)

    for n_components in component_range:
        for seed in range(n_fits):
            score = cv_score_gaussian_hmm(
                X,
                n_splits=n_splits,
                n_components=n_components,
                seed=seed,
                covariance_type=covariance_type,
                init_params=init_params,
                n_iter=n_iter,
                tol=tol,
                evaluation_metric=evaluation_metric,
                log_prefix=log_prefix,
            )
            if score > best_score:
                best_n_components, best_seed, best_score = n_components, seed, score
                print(
                    f"{log_prefix}components={n_components} seed={seed} "
                    f"cv_score={score:.4f}"
                )

    return best_n_components, best_seed, best_score


def refit_gaussian_hmm(
    X: np.ndarray,
    *,
    n_components: int,
    seed: int,
    covariance_type: str,
    init_params: str,
    n_iter: int,
    tol: float,
) -> hmm.GaussianHMM | None:
    """Fits one GaussianHMM with a specific (n_components, seed) config on
    all of X. Used to materialize a CV winner as the model that actually
    gets deployed, refit on all available data for maximal information."""
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        init_params=init_params,
        n_iter=n_iter,
        tol=tol,
    )
    try:
        model.fit(X)
        return model
    except Exception as e:
        logger.warning(f"Refit failed (components={n_components}, seed={seed}): {e}")
        return None
