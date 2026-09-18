import numpy as np

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

N_SPLITS = 2


def _synthetic_X(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


class _StubMetric:
    def evaluate(self, model, X_train, X_validate):
        return model.score(X_validate)


def test_fit_returns_none_below_min_data():
    model = HierarchicalHMMModel("AAPL", np.zeros((10, 2)), FAST_CONFIG, _StubMetric())

    assert model.fit(N_SPLITS) is None
    assert model.predict_states() is None
    assert model.transition_matrices() == []


def test_fit_trains_top_and_sub_models():
    X = _synthetic_X()
    model = HierarchicalHMMModel("AAPL", X, FAST_CONFIG, _StubMetric())

    fitted = model.fit(N_SPLITS)

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
    model.fit(N_SPLITS)

    states = model.predict_states()

    assert states is not None
    assert list(states.columns) == ["top_level_state", "sub_level_state"]
    assert len(states) == len(X)


def test_top_and_sub_states_are_relabeled_by_volatility():
    # top_level_state/sub_level_state must be 0..K-1 volatility ranks
    # (RegimeModel._volatility_rank_map), not raw/arbitrary hmmlearn state
    # indices -- and sub_models must be keyed by the same relabeled
    # top-state values predict_states() produces, so every top state a
    # row reports actually has a corresponding sub-model.
    X = _synthetic_X()
    model = HierarchicalHMMModel("AAPL", X, FAST_CONFIG, _StubMetric())
    model.fit(N_SPLITS)

    states = model.predict_states()

    assert set(states["top_level_state"].unique()) <= set(range(model.top.n_components))
    assert set(model.sub_models.keys()) <= set(range(model.top.n_components))
    for top_state, sub_model in model.sub_models.items():
        rows = states[states["top_level_state"] == top_state]
        assert set(rows["sub_level_state"].dropna().unique()) <= set(
            range(sub_model.n_components)
        )
