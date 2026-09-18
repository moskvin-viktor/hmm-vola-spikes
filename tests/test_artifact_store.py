import json
import shutil

import pandas as pd
from omegaconf import OmegaConf

from hmmstock.artifact_store import ArtifactStore


def test_new_version_starts_at_zero(tmp_path):
    store = ArtifactStore(tmp_path, "HMMModel")

    version = store.new_version()

    assert version.path == tmp_path / "HMMModel" / "version_0"
    assert version.path.exists()


def test_new_version_increments(tmp_path):
    store = ArtifactStore(tmp_path, "HMMModel")

    v0 = store.new_version()
    v1 = store.new_version()
    v2 = store.new_version()

    assert [v0.path.name, v1.path.name, v2.path.name] == [
        "version_0",
        "version_1",
        "version_2",
    ]
    assert v0.path.exists() and v1.path.exists() and v2.path.exists()


def test_new_version_reuses_a_deleted_slot(tmp_path):
    # Versioning reads the directory each call rather than keeping a
    # separate counter file, so a manually deleted version's number is
    # free to be reused -- nothing on disk gets overwritten by this,
    # since that version no longer exists.
    store = ArtifactStore(tmp_path, "HMMModel")
    store.new_version()  # version_0
    store.new_version()  # version_1
    shutil.rmtree(tmp_path / "HMMModel" / "version_1")

    v = store.new_version()

    assert v.path.name == "version_1"


def test_different_model_names_version_independently(tmp_path):
    hmm_store = ArtifactStore(tmp_path, "HMMModel")
    lhmm_store = ArtifactStore(tmp_path, "LayeredHMMModel")

    hmm_store.new_version()
    hmm_store.new_version()
    v = lhmm_store.new_version()

    assert v.path.name == "version_0"


def test_write_config_dumps_yaml(tmp_path):
    version = ArtifactStore(tmp_path, "HMMModel").new_version()

    version.write_config(OmegaConf.create({"max_components": 2, "tol": 0.001}))

    loaded = OmegaConf.load(version.path / "config.yaml")
    assert loaded.max_components == 2
    assert loaded.tol == 0.001


def test_write_config_accepts_plain_dict(tmp_path):
    version = ArtifactStore(tmp_path, "HMMModel").new_version()

    version.write_config({"max_components": 3})

    loaded = OmegaConf.load(version.path / "config.yaml")
    assert loaded.max_components == 3


def test_write_regime_states_and_transition_matrices(tmp_path):
    version = ArtifactStore(tmp_path, "HMMModel").new_version()

    csv_path = version.write_regime_states("AAPL", pd.DataFrame({"regime": [0, 1]}))
    matrix_paths = version.write_transition_matrices(
        "AAPL", [pd.DataFrame([[0.9, 0.1], [0.2, 0.8]])]
    )

    assert csv_path == version.path / "regime_states" / "AAPL.csv"
    assert csv_path.exists()
    assert matrix_paths == [version.path / "transition_matrices" / "AAPL_layer0.csv"]
    assert matrix_paths[0].exists()


def test_metrics_are_recorded_per_ticker_and_flushed(tmp_path):
    version = ArtifactStore(tmp_path, "HMMModel").new_version()

    version.record_metric("AAPL", fitted=True, best_score=1.23, n_components=2)
    version.record_metric("MSFT", fitted=False)
    version.flush_metrics()

    metrics = json.loads((version.path / "metrics.json").read_text())
    assert metrics["AAPL"]["best_score"] == 1.23
    assert metrics["AAPL"]["n_components"] == 2
    assert "recorded_at" in metrics["AAPL"]
    assert metrics["MSFT"]["fitted"] is False
