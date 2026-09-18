import numpy as np

from hmmstock.data.splitter import train_test_holdout
from hmmstock.models.config import HierarchicalHMMConfig
from hmmstock.models.hierarchical import HierarchicalHMMModel

LAYER = dict(
    min_components=2,
    max_components=2,
    covariance_type="diag",
    init_params="stmc",
    n_iter=10,
)

FAST_CONFIG = HierarchicalHMMConfig(
    n_fits=3, random_seed=13, tol=1e-2, top_layer=LAYER, sub_layer=LAYER
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
    model = HierarchicalHMMModel("AAPL", np.zeros((10, 2)), FAST_CONFIG, _StubMetric())

    assert model.fit(HOLDOUT_SPLITTER, N_SPLITS) is None
    assert model.predict_states() is None
    assert model.transition_matrices() == []


def test_fit_trains_top_and_sub_models():
    X = _synthetic_X()
    model = HierarchicalHMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    fitted = model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert fitted is not None
    assert model.top is not None
    assert len(model.sub_models) >= 1
    assert model.cv_score > float("-inf")

    # only the top-level matrix is currently exposed (see class docstring)
    matrices = model.transition_matrices()
    assert len(matrices) == 1
    assert matrices[0].shape == (model.top.n_components, model.top.n_components)


def test_predict_states_has_top_and_sub_columns():
    X = _synthetic_X()
    model = HierarchicalHMMModel("AAPL", X, FAST_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    states = model.predict_states()

    assert states is not None
    assert list(states.columns) == ["top_level_state", "sub_level_state"]
    assert len(states) == len(X)
