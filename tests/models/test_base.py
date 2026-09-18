import numpy as np
import pandas as pd
from hmmlearn import hmm

from hmmstock.models.base import RegimeModel


class _StubModel:
    """Minimal stand-in for a fitted GaussianHMM: only n_components is read
    by _relabel_states_by_volatility."""

    def __init__(self, n_components):
        self.n_components = n_components


def test_relabel_states_orders_by_increasing_volatility():
    # state 0's observations are tightly clustered (low vol), state 1's
    # are spread out (high vol) -- relabeling should flip them since 0
    # must end up as the low-volatility label.
    original_states = np.array([0, 0, 0, 1, 1, 1])
    X = np.array([10.0, 10.1, 9.9, 0.0, 50.0, -50.0])
    model = _StubModel(n_components=2)

    relabeled = RegimeModel._relabel_states_by_volatility(original_states, model, X)

    # original state 0 (tight cluster) -> new label 0 (low vol)
    # original state 1 (spread out)    -> new label 1 (high vol)
    assert relabeled.tolist() == [0, 0, 0, 1, 1, 1]


def test_relabel_states_flips_when_original_labels_reversed():
    original_states = np.array([0, 0, 1, 1])
    X = np.array([0.0, 50.0, 10.0, 10.1])  # original state 0 is the volatile one
    model = _StubModel(n_components=2)

    relabeled = RegimeModel._relabel_states_by_volatility(original_states, model, X)

    # original state 1 (tight) -> new label 0; original state 0 (spread) -> new label 1
    assert relabeled.tolist() == [1, 1, 0, 0]


def test_transition_matrix_df_shape_and_labels():
    model = hmm.GaussianHMM(n_components=3)
    model.transmat_ = np.eye(3)

    df = RegimeModel._transition_matrix_df(model, layer_idx=2)

    assert list(df.index) == ["VS2_0", "VS2_1", "VS2_2"]
    assert list(df.columns) == ["VS2_0", "VS2_1", "VS2_2"]
    pd.testing.assert_frame_equal(
        df, pd.DataFrame(np.eye(3), index=df.index, columns=df.columns)
    )
