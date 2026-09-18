import numpy as np
import pandas as pd
from hmmlearn import hmm

from hmmstock.models.base import RegimeModel


class _StubModel:
    """Minimal stand-in for a fitted GaussianHMM: n_components and
    covars_ are what _relabel_states_by_volatility/_volatility_rank_map
    read."""

    def __init__(self, covars):
        covars = np.asarray(covars)
        self.n_components = len(covars)
        self.covars_ = covars


def test_relabel_states_orders_by_increasing_total_variance():
    original_states = np.array([0, 0, 0, 1, 1, 1])
    # state 0: low variance (0.01), state 1: high variance (100)
    model = _StubModel([[[0.01]], [[100.0]]])

    relabeled = RegimeModel._relabel_states_by_volatility(original_states, model)

    assert relabeled.tolist() == [0, 0, 0, 1, 1, 1]


def test_relabel_states_flips_when_original_labels_reversed():
    original_states = np.array([0, 0, 1, 1])
    model = _StubModel([[[100.0]], [[0.01]]])  # state 0 volatile, state 1 tight

    relabeled = RegimeModel._relabel_states_by_volatility(original_states, model)

    assert relabeled.tolist() == [1, 1, 0, 0]


def test_relabel_uses_covariance_trace_not_flattened_observations():
    # Regression test for the labeling bug: relabeling used to compute
    # np.std() over raw multi-feature observations flattened together,
    # blending unrelated feature units (returns, several differently
    # scaled rolling-vol windows, ...) into one meaningless number. It
    # must instead use each state's own fitted covariance trace -- here,
    # state 0 has the larger single-feature variance but a *smaller*
    # trace than state 1's two smaller-but-summed variances, so a correct
    # trace-based ranking must put state 0 first.
    model = _StubModel(
        [
            [[9.0, 0.0], [0.0, 0.0]],  # trace = 9
            [[5.0, 0.0], [0.0, 5.0]],  # trace = 10
        ]
    )

    rank_map = RegimeModel._volatility_rank_map(model)

    assert rank_map == {0: 0, 1: 1}


def test_transition_matrix_df_shape_and_labels():
    model = hmm.GaussianHMM(n_components=3)
    model.transmat_ = np.eye(3)

    df = RegimeModel._transition_matrix_df(model, layer_idx=2)

    assert list(df.index) == ["VS2_0", "VS2_1", "VS2_2"]
    assert list(df.columns) == ["VS2_0", "VS2_1", "VS2_2"]
    pd.testing.assert_frame_equal(
        df, pd.DataFrame(np.eye(3), index=df.index, columns=df.columns)
    )
