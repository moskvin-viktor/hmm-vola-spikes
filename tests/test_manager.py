from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
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


def _cfg():
    """A composed-config stand-in: what RegimeModelManager now receives
    directly (Hydra composes this for real via config/config.yaml)."""
    return OmegaConf.create(
        {
            "FakeRegimeModel": {"dummy": True},
            "split": {"train_size": 0.8, "shuffle": False},
        }
    )


def _sample_data() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=30)
    df = pd.DataFrame({"normalized_returns": np.arange(30, dtype=float)}, index=idx)
    return {"AAPL": df}


class _NullMetric:
    """Stand-in for LogLikelihoodWithEntropy; irrelevant here since
    FakeRegimeModel never calls evaluate()."""

    def evaluate(self, model, X_validate):
        return 0.0


def _manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return RegimeModelManager(
        data_dict=_sample_data(),
        cfg=_cfg(),
        model_class=FakeRegimeModel,
        evaluation_metric=_NullMetric,
    )


def test_train_all_writes_a_complete_version_directory(tmp_path, monkeypatch):
    FakeRegimeModel.instances.clear()
    manager = _manager(tmp_path, monkeypatch)

    version = manager.train_all()

    assert version.path.resolve() == tmp_path / "artifacts/FakeRegimeModel/version_0"
    assert (version.path / "config.yaml").exists()
    assert (version.path / "metrics.json").exists()
    assert (version.path / "models/AAPL.pkl").exists()
    assert (version.path / "regime_states/AAPL.csv").exists()
    assert (version.path / "transition_matrices/AAPL_layer0.csv").exists()
    assert len(FakeRegimeModel.instances) == 1


def test_train_all_never_overwrites_a_previous_version(tmp_path, monkeypatch):
    FakeRegimeModel.instances.clear()
    manager = _manager(tmp_path, monkeypatch)

    first = manager.train_all()
    second = manager.train_all()

    assert first.path.name == "version_0"
    assert second.path.name == "version_1"
    assert first.path.exists()  # untouched by the second run
    assert (first.path / "models/AAPL.pkl").exists()
    assert len(FakeRegimeModel.instances) == 2  # retrained, not reloaded


def test_write_transition_matrices_skips_unknown_ticker_without_raising(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch)
    version = manager.artifact_store.new_version()

    manager.write_transition_matrices("MSFT", version)  # never trained, must not raise


def test_write_transition_matrices_skips_unfitted_model_without_raising(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch)
    version = manager.artifact_store.new_version()
    unfitted = FakeRegimeModel("AAPL", np.zeros(5), _FakeConfig(), None)
    manager.models["AAPL"] = unfitted

    manager.write_transition_matrices(
        "AAPL", version
    )  # transition_matrices() == [], must not raise

    assert not (version.path / "transition_matrices/AAPL_layer0.csv").exists()
