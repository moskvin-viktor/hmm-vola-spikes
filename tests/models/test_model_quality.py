"""Model-quality tests: verify what a fitted model actually learned on
synthetic data with known ground-truth regime structure -- regime
separation, no degenerate states, genuine persistence, determinism. Not
just "fit()/predict_states() run without crashing" (that's
test_hmm.py/test_layered.py/test_hierarchical.py); these check the
results are right.
"""

import numpy as np
import pandas as pd

from hmmstock.data.splitter import train_test_holdout
from hmmstock.models.config import (
    HierarchicalHMMConfig,
    HMMConfig,
    LayeredHMMConfig,
)
from hmmstock.models.hierarchical import HierarchicalHMMModel
from hmmstock.models.hmm import HMMModel
from hmmstock.models.layered import LayeredHMMModel

HOLDOUT_SPLITTER = train_test_holdout
N_SPLITS = 2


class _StubMetric:
    def evaluate(self, model, X_train, X_validate):
        return model.score(X_validate)


def make_two_regime_series(
    n_per_block: int = 120,
    n_blocks: int = 4,
    low_std: float = 0.3,
    high_std: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 1-feature series alternating low-vol/high-vol blocks, so
    the true regime is known. Returns (X, true_labels); true_labels[i] is
    1 for a high-vol block, 0 for a low-vol block -- same convention
    _relabel_states_by_volatility uses (0 = lowest volatility)."""
    rng = np.random.default_rng(seed)
    chunks, labels = [], []
    for i in range(n_blocks):
        is_high = i % 2 == 1
        std = high_std if is_high else low_std
        chunks.append(rng.normal(loc=0.0, scale=std, size=n_per_block))
        labels.append(np.full(n_per_block, int(is_high)))
    X = np.concatenate(chunks).reshape(-1, 1)
    true_labels = np.concatenate(labels)
    return X, true_labels


def assert_no_degenerate_regimes(
    labels: np.ndarray, n_components: int, min_fraction: float = 0.05
):
    counts = np.bincount(labels, minlength=n_components)
    fractions = counts / len(labels)
    assert (fractions >= min_fraction).all(), (
        f"a regime captured too little data: fractions={fractions.round(3)}"
    )


def assert_transition_matrix_is_persistent(
    transition_df: pd.DataFrame, min_improvement: float = 0.3
):
    """Mean diagonal probability should sit well above the no-persistence
    baseline (1/n_components, i.e. what a state chosen independently at
    random each step would give), normalized to how much of the possible
    improvement over that baseline (up to a diagonal of 1.0) was actually
    captured -- a fixed multiple of the baseline (e.g. "2x") doesn't work
    as a threshold in general: for n_components=2, baseline is already
    0.5, and probabilities can't exceed 1.0.
    """
    n = len(transition_df)
    baseline = 1.0 / n
    mean_diag = np.diag(transition_df.to_numpy()).mean()
    improvement = (mean_diag - baseline) / (1 - baseline)
    assert improvement > min_improvement, (
        f"transition matrix diagonal ({mean_diag:.3f}) isn't meaningfully "
        f"more persistent than the no-persistence baseline ({baseline:.3f}); "
        f"improvement={improvement:.2f}"
    )


# -- HMMModel -----------------------------------------------------------

HMM_CONFIG = HMMConfig(
    covariance_type="diag",
    random_seed=13,
    init_params="stmc",
    n_fits=3,
    n_iter=30,
    tol=1e-3,
    max_components=2,
)


def test_hmm_recovers_known_volatility_regimes():
    X, true_labels = make_two_regime_series()
    model = HMMModel("TEST", X, HMM_CONFIG, _StubMetric())

    assert model.fit(HOLDOUT_SPLITTER, N_SPLITS) is not None

    predicted = model.predict_states()
    accuracy = (predicted == true_labels).mean()
    assert accuracy > 0.75, (
        f"only {accuracy:.0%} agreement with the known regime blocks"
    )


def test_hmm_has_no_degenerate_regimes():
    X, _ = make_two_regime_series()
    model = HMMModel("TEST", X, HMM_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert_no_degenerate_regimes(model.predict_states(), n_components=2)


def test_hmm_regimes_are_persistent_not_noise():
    X, _ = make_two_regime_series()
    model = HMMModel("TEST", X, HMM_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert_transition_matrix_is_persistent(model.transition_matrices()[0])


def test_hmm_fit_is_deterministic():
    X, _ = make_two_regime_series()

    model_a = HMMModel("TEST", X, HMM_CONFIG, _StubMetric())
    model_a.fit(HOLDOUT_SPLITTER, N_SPLITS)

    model_b = HMMModel("TEST", X, HMM_CONFIG, _StubMetric())
    model_b.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert model_a.best_score == model_b.best_score
    assert model_a.cv_score == model_b.cv_score
    np.testing.assert_array_equal(model_a.predict_states(), model_b.predict_states())
    pd.testing.assert_frame_equal(
        model_a.transition_matrices()[0], model_b.transition_matrices()[0]
    )


# -- LayeredHMMModel (layer 0: identical setup to HMMModel, same X) -----

LAYER = dict(
    min_components=2,
    max_components=2,
    covariance_type="diag",
    init_params="stmc",
    n_iter=30,
)

LAYERED_CONFIG = LayeredHMMConfig(
    num_layers=2, n_fits=3, random_seed=13, tol=1e-3, layers=[LAYER, LAYER]
)


def _layer0(model: LayeredHMMModel) -> np.ndarray:
    return model.predict_states()["regime_layer0"].to_numpy()


def test_layered_layer0_recovers_known_volatility_regimes():
    X, true_labels = make_two_regime_series()
    model = LayeredHMMModel("TEST", X, LAYERED_CONFIG, _StubMetric())

    assert model.fit(HOLDOUT_SPLITTER, N_SPLITS) is not None

    accuracy = (_layer0(model) == true_labels).mean()
    assert accuracy > 0.75, (
        f"only {accuracy:.0%} agreement with the known regime blocks"
    )


def test_layered_layer0_has_no_degenerate_regimes():
    X, _ = make_two_regime_series()
    model = LayeredHMMModel("TEST", X, LAYERED_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert_no_degenerate_regimes(_layer0(model), n_components=2)


def test_layered_layer0_is_persistent_not_noise():
    X, _ = make_two_regime_series()
    model = LayeredHMMModel("TEST", X, LAYERED_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert_transition_matrix_is_persistent(model.transition_matrices()[0])


def test_layered_fit_is_deterministic():
    X, _ = make_two_regime_series()

    model_a = LayeredHMMModel("TEST", X, LAYERED_CONFIG, _StubMetric())
    model_a.fit(HOLDOUT_SPLITTER, N_SPLITS)

    model_b = LayeredHMMModel("TEST", X, LAYERED_CONFIG, _StubMetric())
    model_b.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert model_a.best_score == model_b.best_score
    pd.testing.assert_frame_equal(model_a.predict_states(), model_b.predict_states())


# -- HierarchicalHMMModel (top level: identical setup to HMMModel) ------

HIERARCHICAL_CONFIG = HierarchicalHMMConfig(
    n_fits=3, random_seed=13, tol=1e-3, top_layer=LAYER, sub_layer=LAYER
)


def _top_level(model: HierarchicalHMMModel) -> np.ndarray:
    return model.predict_states()["top_level_state"].to_numpy()


def test_hierarchical_top_level_recovers_known_volatility_regimes():
    X, true_labels = make_two_regime_series()
    model = HierarchicalHMMModel("TEST", X, HIERARCHICAL_CONFIG, _StubMetric())

    assert model.fit(HOLDOUT_SPLITTER, N_SPLITS) is not None

    # top_level_state isn't volatility-relabeled the way HMMModel/
    # LayeredHMMModel's states are (see hierarchical.py's docstring), so
    # only the *separation* into two consistent groups is guaranteed to
    # line up with the truth up to a label swap -- check both orientations.
    top = _top_level(model)
    accuracy = max((top == true_labels).mean(), (top == 1 - true_labels).mean())
    assert accuracy > 0.75, (
        f"only {accuracy:.0%} agreement with the known regime blocks"
    )


def test_hierarchical_top_level_has_no_degenerate_regimes():
    X, _ = make_two_regime_series()
    model = HierarchicalHMMModel("TEST", X, HIERARCHICAL_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert_no_degenerate_regimes(_top_level(model), n_components=2)


def test_hierarchical_top_level_is_persistent_not_noise():
    X, _ = make_two_regime_series()
    model = HierarchicalHMMModel("TEST", X, HIERARCHICAL_CONFIG, _StubMetric())
    model.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert_transition_matrix_is_persistent(model.transition_matrices()[0])


def test_hierarchical_fit_is_deterministic():
    X, _ = make_two_regime_series()

    model_a = HierarchicalHMMModel("TEST", X, HIERARCHICAL_CONFIG, _StubMetric())
    model_a.fit(HOLDOUT_SPLITTER, N_SPLITS)

    model_b = HierarchicalHMMModel("TEST", X, HIERARCHICAL_CONFIG, _StubMetric())
    model_b.fit(HOLDOUT_SPLITTER, N_SPLITS)

    assert model_a.best_score == model_b.best_score
    pd.testing.assert_frame_equal(model_a.predict_states(), model_b.predict_states())
