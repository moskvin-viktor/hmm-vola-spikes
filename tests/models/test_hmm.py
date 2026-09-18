import numpy as np

from hmmstock.models.config import HMMConfig
from hmmstock.models.hmm import HMMModel

FAST_CONFIG = HMMConfig(
    covariance_type="diag",
    random_seed=13,
    init_params="stmc",
    n_fits=5,
    tol=1e-2,
    max_components=3,
)


def _identity_split(X):
    split = int(len(X) * 0.8)
    return X[:split], X[split:]


def _synthetic_X(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


class _StubMetric:
    def evaluate(self, model, X_validate):
        return model.score(X_validate)


def test_fit_returns_none_below_min_data():
    model = HMMModel("AAPL", np.zeros((10, 2)), FAST_CONFIG, _StubMetric())

    assert model.fit(_identity_split) is None
    assert model.predict_states() is None
    assert model.transition_matrices() == []


def test_fit_predict_and_transition_matrices_happy_path():
    X = _synthetic_X()
    model = HMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    fitted = model.fit(_identity_split)

    assert fitted is not None
    assert 2 <= fitted.n_components <= FAST_CONFIG.max_components
    assert model.best_score > float("-inf")

    states = model.predict_states()
    assert states is not None
    assert states.shape == (len(X),)
    assert set(states.tolist()) <= set(range(fitted.n_components))

    matrices = model.transition_matrices()
    assert len(matrices) == 1
    assert matrices[0].shape == (fitted.n_components, fitted.n_components)
    assert list(matrices[0].index) == [f"VS0_{i}" for i in range(fitted.n_components)]
