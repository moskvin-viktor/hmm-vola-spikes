import numpy as np

from hmmstock.models.trainer import fit_best_gaussian_hmm


class _ScoreByComponents:
    """Deterministic fake metric: score == n_components, so the winner of
    the grid search is predictable regardless of actual EM convergence.
    Optionally raises for specific component counts, to test that a
    failing fit doesn't abort the rest of the search."""

    def __init__(self, fail_for=frozenset()):
        self.calls = 0
        self.fail_for = fail_for

    def evaluate(self, model, X_validate):
        self.calls += 1
        if model.n_components in self.fail_for:
            raise RuntimeError("forced failure")
        return model.n_components


def _synthetic_X(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


def test_picks_the_highest_scoring_model():
    X = _synthetic_X()
    metric = _ScoreByComponents()

    best_model, best_score = fit_best_gaussian_hmm(
        X,
        X,
        component_range=range(2, 4),  # 2, 3
        n_fits=2,
        covariance_type="full",
        init_params="stmc",
        tol=1e-2,
        evaluation_metric=metric,
    )

    assert best_model is not None
    assert best_model.n_components == 3  # highest score under the fake metric
    assert best_score == 3


def test_exhausts_the_full_grid():
    X = _synthetic_X()
    metric = _ScoreByComponents()

    fit_best_gaussian_hmm(
        X,
        X,
        component_range=range(2, 5),  # 2, 3, 4
        n_fits=3,
        covariance_type="full",
        init_params="stmc",
        tol=1e-2,
        evaluation_metric=metric,
    )

    assert metric.calls == 3 * 3  # component_range size * n_fits


def test_a_failing_component_count_does_not_abort_the_search():
    X = _synthetic_X()
    metric = _ScoreByComponents(fail_for={3})  # the would-be winner fails

    best_model, best_score = fit_best_gaussian_hmm(
        X,
        X,
        component_range=range(2, 4),  # 2, 3
        n_fits=2,
        covariance_type="full",
        init_params="stmc",
        tol=1e-2,
        evaluation_metric=metric,
    )

    assert best_model is not None
    assert best_model.n_components == 2  # only surviving candidate
    assert best_score == 2


def test_returns_none_and_neg_inf_when_every_fit_fails():
    X = _synthetic_X()
    metric = _ScoreByComponents(fail_for={2, 3})

    best_model, best_score = fit_best_gaussian_hmm(
        X,
        X,
        component_range=range(2, 4),
        n_fits=2,
        covariance_type="full",
        init_params="stmc",
        tol=1e-2,
        evaluation_metric=metric,
    )

    assert best_model is None
    assert best_score == float("-inf")
