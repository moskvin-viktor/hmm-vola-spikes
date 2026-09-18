import numpy as np

from hmmstock.models.trainer import (
    cv_score_gaussian_hmm,
    refit_gaussian_hmm,
    score_on_test_holdout,
    select_best_gaussian_hmm,
    select_best_gaussian_hmm_holdout,
)


class _ScoreByComponents:
    """Deterministic fake metric: score == n_components, so the winner of
    a grid search is predictable regardless of actual EM convergence."""

    def __init__(self, fail_for=frozenset()):
        self.calls = 0
        self.fail_for = fail_for

    def evaluate(self, model, X_train, X_validate):
        self.calls += 1
        if model.n_components in self.fail_for:
            raise RuntimeError("forced failure")
        return model.n_components


def _synthetic_X(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


COMMON_KW = dict(covariance_type="full", init_params="stmc", n_iter=10, tol=1e-2)


# -- cv_score_gaussian_hmm --------------------------------------------------


def test_cv_score_averages_across_folds():
    X = _synthetic_X()
    metric = _ScoreByComponents()

    score = cv_score_gaussian_hmm(
        X, n_splits=3, n_components=2, seed=0, evaluation_metric=metric, **COMMON_KW
    )

    assert score == 2  # fake metric always returns n_components
    assert metric.calls == 3  # one evaluate() per fold


def test_cv_score_is_neg_inf_when_every_fold_fails():
    X = _synthetic_X()
    metric = _ScoreByComponents(fail_for={2})

    score = cv_score_gaussian_hmm(
        X, n_splits=3, n_components=2, seed=0, evaluation_metric=metric, **COMMON_KW
    )

    assert score == float("-inf")


# -- select_best_gaussian_hmm (walk-forward CV grid search) -----------------


def test_select_best_picks_highest_cv_score():
    X = _synthetic_X()
    metric = _ScoreByComponents()

    best_n, best_seed, best_score = select_best_gaussian_hmm(
        X,
        component_range=range(2, 4),
        n_fits=2,
        n_splits=3,
        evaluation_metric=metric,
        **COMMON_KW,
    )

    assert best_n == 3  # highest score under the fake metric
    assert best_seed is not None
    assert best_score == 3


def test_select_best_a_failing_component_does_not_abort_the_search():
    X = _synthetic_X()
    metric = _ScoreByComponents(fail_for={3})  # the would-be winner fails every fold

    best_n, best_seed, best_score = select_best_gaussian_hmm(
        X,
        component_range=range(2, 4),
        n_fits=2,
        n_splits=3,
        evaluation_metric=metric,
        **COMMON_KW,
    )

    assert best_n == 2  # only surviving candidate
    assert best_score == 2


def test_select_best_returns_none_when_everything_fails():
    X = _synthetic_X()
    metric = _ScoreByComponents(fail_for={2, 3})

    best_n, best_seed, best_score = select_best_gaussian_hmm(
        X,
        component_range=range(2, 4),
        n_fits=2,
        n_splits=3,
        evaluation_metric=metric,
        **COMMON_KW,
    )

    assert (best_n, best_seed, best_score) == (None, None, float("-inf"))


# -- select_best_gaussian_hmm_holdout (single split, scarce-data path) ------


def test_select_best_holdout_picks_highest_score():
    X = _synthetic_X(n=40)
    X_train, X_test = X[:30], X[30:]
    metric = _ScoreByComponents()

    best_n, best_seed, best_score = select_best_gaussian_hmm_holdout(
        X_train,
        X_test,
        component_range=range(2, 4),
        n_fits=2,
        evaluation_metric=metric,
        **COMMON_KW,
    )

    assert best_n == 3
    assert best_score == 3


def test_select_best_holdout_falls_back_to_in_sample_when_test_empty():
    X_train = _synthetic_X(n=30)
    metric = _ScoreByComponents()

    best_n, best_seed, best_score = select_best_gaussian_hmm_holdout(
        X_train,
        np.empty((0, 2)),
        component_range=range(2, 3),
        n_fits=1,
        evaluation_metric=metric,
        **COMMON_KW,
    )

    assert best_n == 2  # still finds a candidate, scored on X_train itself
    assert metric.calls == 1


# -- refit_gaussian_hmm ------------------------------------------------------


def test_refit_returns_a_fitted_model():
    X = _synthetic_X()

    model = refit_gaussian_hmm(X, n_components=2, seed=0, **COMMON_KW)

    assert model is not None
    assert model.n_components == 2


def test_refit_returns_none_on_failure():
    # More components than samples -> hmmlearn can't fit, raises.
    X = _synthetic_X(n=3)

    model = refit_gaussian_hmm(X, n_components=10, seed=0, **COMMON_KW)

    assert model is None


# -- score_on_test_holdout ---------------------------------------------------


def test_score_on_test_holdout_scores_a_fresh_refit_on_test_data():
    X = _synthetic_X(n=60)
    X_trainval, X_test = X[:50], X[50:]
    metric = _ScoreByComponents()

    score = score_on_test_holdout(
        X_trainval,
        X_test,
        cv_score=-999,
        n_components=2,
        seed=0,
        evaluation_metric=metric,
        **COMMON_KW,
    )

    assert score == 2  # fake metric returns n_components, not the cv_score fallback


def test_score_on_test_holdout_falls_back_to_cv_score_when_test_empty():
    X_trainval = _synthetic_X(n=50)
    metric = _ScoreByComponents()

    score = score_on_test_holdout(
        X_trainval,
        np.empty((0, 2)),
        cv_score=-42,
        n_components=2,
        seed=0,
        evaluation_metric=metric,
        **COMMON_KW,
    )

    assert score == -42
    assert metric.calls == 0  # never even tried to evaluate
