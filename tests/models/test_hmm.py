import numpy as np

from hmmstock.data.splitter import SplitConfig, train_test_holdout
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

HOLDOUT_SPLITTER = train_test_holdout  # uses SplitConfig() defaults
N_SPLITS = 2


def _synthetic_X(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


class _StubMetric:
    def evaluate(self, model, X_train, X_validate):
        return model.score(X_validate)


def test_fit_returns_none_below_min_data():
    model = HMMModel("AAPL", np.zeros((10, 2)), FAST_CONFIG, _StubMetric())

    assert model.fit(HOLDOUT_SPLITTER, N_SPLITS) is None
    assert model.predict_states() is None
    assert model.transition_matrices() == []


def test_fit_predict_and_transition_matrices_happy_path():
    X = _synthetic_X()
    model = HMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    fitted = model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert fitted is not None
    assert 2 <= fitted.n_components <= FAST_CONFIG.max_components
    assert model.best_score > float("-inf")
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
    # Confirms the design decision: after CV selects hyperparameters and
    # they're scored on the test holdout, the deployed model is refit on
    # ALL of X (train + CV + test), not just the trainval slice.
    import hmmstock.models.hmm as hmm_module

    X = _synthetic_X()
    model = HMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    real_refit = hmm_module.refit_gaussian_hmm
    refit_lengths = []

    def spy_refit(X_arg, **kwargs):
        refit_lengths.append(len(X_arg))
        return real_refit(X_arg, **kwargs)

    monkeypatch.setattr(hmm_module, "refit_gaussian_hmm", spy_refit)

    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert refit_lengths[-1] == len(X)  # final (deploy) refit used all of X


def test_fit_returns_none_when_trainval_too_small_after_holdout():
    # 25 rows total, 90% held out as "test" -> ~2 rows of trainval, below
    # the 20-row minimum for CV.
    X = _synthetic_X(n=25)
    model = HMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    def mostly_test_splitter(arr):
        return train_test_holdout(arr, config=SplitConfig(test_size=0.9))

    assert model.fit(mostly_test_splitter, N_SPLITS) is None
