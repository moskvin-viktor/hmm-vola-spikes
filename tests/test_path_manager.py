import pandas as pd
import pytest

from hmmstock.artifact_store import ArtifactStore
from hmmstock.path_manager import PathManager


class _FakeModel:
    pass


def _write_fake_run(root, model_name="HMMModel"):
    """Writes one version's worth of artifacts via the real ArtifactStore,
    so these tests verify PathManager reads exactly what gets written,
    not a hand-rolled directory guess."""
    store = ArtifactStore(root, model_name)
    version = store.new_version()
    version.write_config({"max_components": 2})
    version.record_metric("AAPL", fitted=True, best_score=1.5)
    version.flush_metrics()
    idx = pd.date_range("2024-01-01", periods=2)
    version.write_regime_states("AAPL", pd.DataFrame({"regime": [0, 1]}, index=idx))
    version.write_transition_matrices("AAPL", [pd.DataFrame([[0.9, 0.1], [0.2, 0.8]])])

    version.write_model("AAPL", _FakeModel())
    return version


def test_versions_empty_when_model_never_trained(tmp_path):
    pm = PathManager(tmp_path)

    assert pm.versions("HMMModel") == []
    assert pm.latest_version("HMMModel") is None


def test_versions_and_latest_version(tmp_path):
    store = ArtifactStore(tmp_path, "HMMModel")
    store.new_version()
    store.new_version()
    pm = PathManager(tmp_path)

    assert pm.versions("HMMModel") == ["version_0", "version_1"]
    assert pm.latest_version("HMMModel") == "version_1"


def test_version_dir_missing_model_raises(tmp_path):
    pm = PathManager(tmp_path)

    with pytest.raises(FileNotFoundError):
        pm.version_dir("HMMModel")


def test_version_dir_accepts_int_shorthand(tmp_path):
    _write_fake_run(tmp_path)
    _write_fake_run(tmp_path)  # version_1
    pm = PathManager(tmp_path)

    assert pm.version_dir("HMMModel", 0).name == "version_0"
    assert pm.version_dir("HMMModel", version=1).name == "version_1"


def test_version_dir_defaults_to_latest(tmp_path):
    _write_fake_run(tmp_path)
    latest = _write_fake_run(tmp_path)
    pm = PathManager(tmp_path)

    assert pm.version_dir("HMMModel") == latest.path


def test_tickers_lists_models_with_a_saved_pickle(tmp_path):
    _write_fake_run(tmp_path)
    pm = PathManager(tmp_path)

    assert pm.tickers("HMMModel") == ["AAPL"]


def test_load_metrics(tmp_path):
    _write_fake_run(tmp_path)
    pm = PathManager(tmp_path)

    metrics = pm.load_metrics("HMMModel")

    assert metrics["AAPL"]["best_score"] == 1.5
    assert metrics["AAPL"]["fitted"] is True


def test_load_regime_states(tmp_path):
    _write_fake_run(tmp_path)
    pm = PathManager(tmp_path)

    df = pm.load_regime_states("HMMModel", "AAPL")

    assert df["regime"].tolist() == [0, 1]


def test_load_transition_matrices(tmp_path):
    _write_fake_run(tmp_path)
    pm = PathManager(tmp_path)

    matrices = pm.load_transition_matrices("HMMModel", "AAPL")

    assert len(matrices) == 1
    assert matrices[0].shape == (2, 2)


def test_load_model_roundtrips_through_joblib(tmp_path):
    _write_fake_run(tmp_path)
    pm = PathManager(tmp_path)

    model = pm.load_model("HMMModel", "AAPL")

    assert type(model).__name__ == "_FakeModel"
