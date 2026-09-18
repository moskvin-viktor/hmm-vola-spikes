import numpy as np

from hmmstock.data.splitter import train_test_holdout
from hmmstock.models.config import LayeredHMMConfig
from hmmstock.models.layered import LayeredHMMModel

LAYER = dict(
    min_components=2,
    max_components=2,
    covariance_type="diag",
    init_params="stmc",
    n_iter=10,
)

FAST_CONFIG = LayeredHMMConfig(
    num_layers=2, n_fits=3, random_seed=13, tol=1e-2, layers=[LAYER, LAYER]
)

HOLDOUT_SPLITTER = train_test_holdout
N_SPLITS = 2


def _synthetic_X(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


class _StubMetric:
    def evaluate(self, model, X_train, X_validate):
        return model.score(X_validate)


def test_fit_returns_none_below_min_data():
    model = LayeredHMMModel("AAPL", np.zeros((10, 2)), FAST_CONFIG, _StubMetric())

    assert model.fit(HOLDOUT_SPLITTER, N_SPLITS) is None
    assert model.predict_states() is None
    assert model.transition_matrices() == []


def test_fit_trains_one_hmm_per_layer():
    X = _synthetic_X()
    model = LayeredHMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    fitted = model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert fitted is not None
    assert len(model.layers) == 2
    assert model.best_score > float("-inf")
    assert model.cv_score > float("-inf")

    matrices = model.transition_matrices()
    assert len(matrices) == 2
    assert list(matrices[0].index) == ["VS0_0", "VS0_1"]
    assert list(matrices[1].index) == ["VS1_0", "VS1_1"]


def test_predict_states_has_one_column_per_layer():
    X = _synthetic_X()
    model = LayeredHMMModel("AAPL", X, FAST_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    states = model.predict_states()

    assert states is not None
    assert list(states.columns) == ["regime_layer0", "regime_layer1"]
    assert len(states) == len(X)
