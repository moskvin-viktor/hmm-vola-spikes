import numpy as np

from hmmstock.models.config import HMMConfig
from hmmstock.models.hmm import HMMModel

FAST_CONFIG = HMMConfig(
    covariance_type="diag",
    random_seed=13,
    init_params="stmc",
    n_fits=3,
    n_iter=10,
    tol=1e-2,
    max_components=3,
)

N_SPLITS = 2


def _synthetic_X(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


class _StubMetric:
    def evaluate(self, model, X_train, X_validate):
        return model.score(X_validate)


def test_fit_returns_none_below_min_data():
    model = HMMModel("AAPL", np.zeros((10, 2)), FAST_CONFIG, _StubMetric())

    assert model.fit(N_SPLITS) is None
    assert model.predict_states() is None
    assert model.transition_matrices() == []


def test_fit_predict_and_transition_matrices_happy_path():
    X = _synthetic_X()
    model = HMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    fitted = model.fit(N_SPLITS)

    assert fitted is not None
    assert 2 <= fitted.n_components <= FAST_CONFIG.max_components
    assert model.cv_score > float("-inf")

    states = model.predict_states()
    assert states is not None
    assert states.shape == (len(X),)
    assert set(states.tolist()) <= set(range(fitted.n_components))

    matrices = model.transition_matrices()
    assert len(matrices) == 1
    assert matrices[0].shape == (fitted.n_components, fitted.n_components)
    assert list(matrices[0].index) == [f"VS0_{i}" for i in range(fitted.n_components)]


def test_fit_deploys_a_model_refit_on_all_data(monkeypatch):
    # Confirms the design decision: the deployed model is refit on ALL of
    # X, not just some CV-internal slice.
    import hmmstock.models.hmm as hmm_module

    X = _synthetic_X()
    model = HMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    real_refit = hmm_module.refit_gaussian_hmm
    refit_lengths = []

    def spy_refit(X_arg, **kwargs):
        refit_lengths.append(len(X_arg))
        return real_refit(X_arg, **kwargs)

    monkeypatch.setattr(hmm_module, "refit_gaussian_hmm", spy_refit)

    model.fit(N_SPLITS)

    assert refit_lengths[-1] == len(X)


def test_fit_returns_none_when_every_candidate_fails():
    class _AlwaysFailMetric:
        def evaluate(self, model, X_train, X_validate):
            raise RuntimeError("forced failure")

    X = _synthetic_X()
    model = HMMModel("AAPL", X, FAST_CONFIG, _AlwaysFailMetric())

    assert model.fit(N_SPLITS) is None


def test_fit_does_not_crash_when_n_splits_exceeds_data_capacity():
    # select_best_gaussian_hmm caps n_splits down (adaptive_n_splits) for
    # data too small to support it -- sklearn's TimeSeriesSplit would
    # otherwise raise outright rather than just scoring badly.
    X = _synthetic_X(n=20)
    model = HMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    model.fit(n_splits=100)  # must not raise
