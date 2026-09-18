import json
from pathlib import Path

import joblib
import pandas as pd

from .models.base import RegimeModel


class PathManager:
    """Read-only navigator for the artifacts/{model_name}/version_N/ tree
    that ArtifactStore writes (see artifact_store.py): resolves versions
    and loads a run's config/metrics/models/regime-states/transition-
    matrices, for exploring results (e.g. from a marimo notebook). This
    doesn't write anything -- that's ArtifactStore's job.
    """

    def __init__(self, artifacts_root: str | Path = "artifacts"):
        self.root = Path(artifacts_root)

    def model_dir(self, model_name: str) -> Path:
        return self.root / model_name

    def versions(self, model_name: str) -> list[str]:
        """All version_N directory names for a model, oldest first."""
        model_dir = self.model_dir(model_name)
        if not model_dir.exists():
            return []
        names = (
            p.name
            for p in model_dir.iterdir()
            if p.is_dir() and p.name.startswith("version_")
        )
        return sorted(names, key=lambda n: int(n.removeprefix("version_")))

    def latest_version(self, model_name: str) -> str | None:
        versions = self.versions(model_name)
        return versions[-1] if versions else None

    def version_dir(self, model_name: str, version: str | int | None = None) -> Path:
        """Path to a run's version directory. version=None resolves to the
        latest; an int N is shorthand for "version_N"."""
        version = f"version_{version}" if isinstance(version, int) else version
        version = version or self.latest_version(model_name)
        if version is None:
            raise FileNotFoundError(
                f"No trained versions found for {model_name!r} under {self.root}"
            )
        path = self.model_dir(model_name) / version
        if not path.exists():
            raise FileNotFoundError(
                f"No {version!r} for {model_name!r} under {self.root}"
            )
        return path

    def tickers(self, model_name: str, version: str | int | None = None) -> list[str]:
        """Tickers with a saved model in this version."""
        models_dir = self.version_dir(model_name, version) / "models"
        if not models_dir.exists():
            return []
        return sorted(p.stem for p in models_dir.glob("*.pkl"))

    def config_file(self, model_name: str, version: str | int | None = None) -> Path:
        return self.version_dir(model_name, version) / "config.yaml"

    def metrics_file(self, model_name: str, version: str | int | None = None) -> Path:
        return self.version_dir(model_name, version) / "metrics.json"

    def model_file(
        self, model_name: str, ticker: str, version: str | int | None = None
    ) -> Path:
        return self.version_dir(model_name, version) / "models" / f"{ticker}.pkl"

    def regime_states_file(
        self, model_name: str, ticker: str, version: str | int | None = None
    ) -> Path:
        return self.version_dir(model_name, version) / "regime_states" / f"{ticker}.csv"

    def transition_matrix_files(
        self, model_name: str, ticker: str, version: str | int | None = None
    ) -> list[Path]:
        """A ticker's transition-matrix CSVs, ordered by layer."""
        matrices_dir = self.version_dir(model_name, version) / "transition_matrices"
        return sorted(matrices_dir.glob(f"{ticker}_layer*.csv"))

    # -- loaders: read a path's content directly --

    def load_metrics(self, model_name: str, version: str | int | None = None) -> dict:
        return json.loads(self.metrics_file(model_name, version).read_text())

    def load_model(
        self, model_name: str, ticker: str, version: str | int | None = None
    ) -> RegimeModel:
        return joblib.load(self.model_file(model_name, ticker, version))

    def load_regime_states(
        self, model_name: str, ticker: str, version: str | int | None = None
    ) -> pd.DataFrame:
        return pd.read_csv(
            self.regime_states_file(model_name, ticker, version),
            index_col=0,
            parse_dates=True,
        )

    def load_transition_matrices(
        self, model_name: str, ticker: str, version: str | int | None = None
    ) -> list[pd.DataFrame]:
        return [
            pd.read_csv(f, index_col=0)
            for f in self.transition_matrix_files(model_name, ticker, version)
        ]
