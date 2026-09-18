from types import SimpleNamespace

import numpy as np
import pandas as pd
from pydantic import BaseModel

from hmmstock.manager import RegimeModelManager, sanitize_ticker
from hmmstock.models.base import RegimeModel


def test_sanitize_ticker_strips_caret():
    assert sanitize_ticker("^GSPC") == "GSPC"


def test_sanitize_ticker_replaces_slash():
    assert sanitize_ticker("BRK/B") == "BRK_B"


def test_sanitize_ticker_leaves_plain_ticker():
    assert sanitize_ticker("AAPL") == "AAPL"


# -- RegimeModelManager orchestration, via a fake model (no real HMM fitting) --


class _FakeConfig(BaseModel):
    pass


class FakeRegimeModel(RegimeModel):
    """A RegimeModel double: no real fitting, just records that it ran."""

    config_cls = _FakeConfig
    instances: list["FakeRegimeModel"] = []

    def __init__(self, name, X, config: _FakeConfig, evaluation_metric):
        self.name = name
        self.X = X
        self.cfg = config
        self.evaluation_metric = evaluation_metric
        self.best_score = 1.0
        self.fitted = False
        FakeRegimeModel.instances.append(self)

    def fit(self, splitter):
        self.fitted = True
        splitter(self.X)  # exercise the injected splitter, like a real model would
        return SimpleNamespace(n_components=2)

    def predict_states(self):
        if not self.fitted:
            return None
        return np.zeros(len(self.X), dtype=int)

    def transition_matrices(self) -> list[pd.DataFrame]:
        if not self.fitted:
            return []
        labels = ["VS0_0", "VS0_1"]
        return [pd.DataFrame([[0.9, 0.1], [0.2, 0.8]], index=labels, columns=labels)]


def _write_config(tmp_path) -> str:
    path = tmp_path / "model.yaml"
    path.write_text(
        "FakeRegimeModel:\n  dummy: true\nsplit:\n  train_size: 0.8\n  shuffle: false\n"
    )
    return str(path)


def _sample_data() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=30)
    df = pd.DataFrame({"normalized_returns": np.arange(30, dtype=float)}, index=idx)
    return {"AAPL": df}


class _NullMetric:
    """Stand-in for LogLikelihoodWithEntropy, which loads its own
    hardcoded config/model.yaml -- irrelevant here since FakeRegimeModel
    never calls evaluate()."""

    def evaluate(self, model, X_validate):
        return 0.0


def test_train_all_writes_csv_transition_matrix_and_pickle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FakeRegimeModel.instances.clear()

    manager = RegimeModelManager(
        data_dict=_sample_data(),
        config_path=_write_config(tmp_path),
        model_class=FakeRegimeModel,
        evaluation_metric=_NullMetric,
    )
    manager.train_all()

    assert (tmp_path / "results/FakeRegimeModel/csvs/AAPL/regime_states.csv").exists()
    assert (
        tmp_path
        / "results/FakeRegimeModel/transition_matrices/AAPL_transition_matrix_layer0.csv"
    ).exists()
    assert (
        tmp_path / "results/FakeRegimeModel/saved_models/FakeRegimeModel_hmm.pkl"
    ).exists()
    assert len(FakeRegimeModel.instances) == 1


def test_train_all_second_call_loads_from_store_without_refitting(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    FakeRegimeModel.instances.clear()

    config_path = _write_config(tmp_path)
    RegimeModelManager(
        data_dict=_sample_data(),
        config_path=config_path,
        model_class=FakeRegimeModel,
        evaluation_metric=_NullMetric,
    ).train_all()
    assert len(FakeRegimeModel.instances) == 1

    RegimeModelManager(
        data_dict=_sample_data(),
        config_path=config_path,
        model_class=FakeRegimeModel,
        evaluation_metric=_NullMetric,
    ).train_all()

    # second manager loaded the pickled model instead of constructing a new one
    assert len(FakeRegimeModel.instances) == 1


def test_write_transition_matrices_skips_unknown_ticker_without_raising(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    manager = RegimeModelManager(
        data_dict=_sample_data(),
        config_path=_write_config(tmp_path),
        model_class=FakeRegimeModel,
        evaluation_metric=_NullMetric,
    )

    manager.write_transition_matrices("MSFT")  # never trained, must not raise


def test_write_transition_matrices_skips_unfitted_model_without_raising(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    manager = RegimeModelManager(
        data_dict=_sample_data(),
        config_path=_write_config(tmp_path),
        model_class=FakeRegimeModel,
        evaluation_metric=_NullMetric,
    )
    unfitted = FakeRegimeModel("AAPL", np.zeros(5), _FakeConfig(), None)
    manager.models["AAPL"] = unfitted

    manager.write_transition_matrices(
        "AAPL"
    )  # transition_matrices() == [], must not raise

    assert not (
        tmp_path
        / "results/FakeRegimeModel/transition_matrices/AAPL_transition_matrix_layer0.csv"
    ).exists()
