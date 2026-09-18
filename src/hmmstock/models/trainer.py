import logging

import numpy as np
from hmmlearn import hmm

from hmmstock.data.splitter import walk_forward_splits

logger = logging.getLogger(__name__)


def cv_score_gaussian_hmm(
    X_trainval: np.ndarray,
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
    config across `n_splits` walk-forward CV folds of X_trainval: refit
    fresh on each fold's own training slice, score on that fold's
    validation slice (chronologically after it, never before). Returns
    -inf if every fold's fit failed.
    """
    scores = []
    for X_train, X_val in walk_forward_splits(X_trainval, n_splits):
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
    X_trainval: np.ndarray,
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
    by cv_score_gaussian_hmm. Returns (best_n_components, best_seed,
    best_cv_score); (None, None, -inf) if every candidate failed on every
    fold (e.g. too little data for n_splits folds).
    """
    best_n_components, best_seed, best_score = None, None, -np.inf

    for n_components in component_range:
        for seed in range(n_fits):
            score = cv_score_gaussian_hmm(
                X_trainval,
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


def select_best_gaussian_hmm_holdout(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    component_range: range,
    n_fits: int,
    covariance_type: str,
    init_params: str,
    n_iter: int,
    tol: float,
    evaluation_metric,
    log_prefix: str = "",
) -> tuple[int | None, int | None, float]:
    """Grid search scored by a single train/test split rather than
    multi-fold CV -- for data too scarce for walk-forward CV (sklearn's
    TimeSeriesSplit requires n_splits >= 2, which needs more rows than a
    small partition may have; used for HierarchicalHMMModel's per-regime
    sub-HMMs). Falls back to scoring in-sample on X_train if X_test is
    empty, so a config with no test rows available still gets *a* score
    rather than being unscorable.
    """
    best_n_components, best_seed, best_score = None, None, -np.inf
    eval_data = X_test if len(X_test) > 0 else X_train

    for n_components in component_range:
        for seed in range(n_fits):
            model = refit_gaussian_hmm(
                X_train,
                n_components=n_components,
                seed=seed,
                covariance_type=covariance_type,
                init_params=init_params,
                n_iter=n_iter,
                tol=tol,
            )
            if model is None:
                continue
            try:
                score = evaluation_metric.evaluate(model, X_train, eval_data)
            except Exception as e:
                logger.warning(
                    f"{log_prefix}scoring failed (components={n_components}, "
                    f"seed={seed}): {e}"
                )
                continue
            if score > best_score:
                best_n_components, best_seed, best_score = n_components, seed, score
                print(
                    f"{log_prefix}components={n_components} seed={seed} "
                    f"score={score:.4f}"
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
    all of X. Used to materialize a CV winner as an actual fitted model:
    once on train+CV data for an honest test-holdout score, again on all
    data (train+CV+test) for the model that actually gets deployed."""
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


def score_on_test_holdout(
    X_trainval: np.ndarray,
    X_test: np.ndarray,
    cv_score: float,
    *,
    n_components: int,
    seed: int,
    covariance_type: str,
    init_params: str,
    n_iter: int,
    tol: float,
    evaluation_metric,
    log_prefix: str = "",
) -> float:
    """Honest score of a CV-selected (n_components, seed) config on the
    untouched test holdout: refits on X_trainval only, scores on X_test.
    Falls back to `cv_score` if there's no test data or the holdout refit
    fails -- so a model always has *some* score, even in edge cases.
    """
    if len(X_test) == 0:
        return cv_score

    trainval_model = refit_gaussian_hmm(
        X_trainval,
        n_components=n_components,
        seed=seed,
        covariance_type=covariance_type,
        init_params=init_params,
        n_iter=n_iter,
        tol=tol,
    )
    if trainval_model is None:
        return cv_score

    try:
        return evaluation_metric.evaluate(trainval_model, X_trainval, X_test)
    except Exception as e:
        logger.warning(f"{log_prefix}Test-holdout scoring failed: {e}")
        return cv_score
