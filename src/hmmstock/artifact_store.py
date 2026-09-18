import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .models.base import RegimeModel


class ArtifactStore:
    """Allocates versioned run directories for one model class, under
    artifacts/{model_name}/version_N/ (auto-incrementing, N never reused).

    Mirrors the shape of PyTorch Lightning's lightning_logs/version_N/:
    every training run gets its own directory and nothing is ever
    overwritten, so history is never silently lost.
    """

    def __init__(self, root: Path, model_name: str):
        self.model_dir = root / model_name

    def new_version(self) -> "ArtifactVersion":
        self.model_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(name.removeprefix("version_"))
            for path in self.model_dir.glob("version_*")
            if (name := path.name).removeprefix("version_").isdigit()
        ]
        next_version = max(existing, default=-1) + 1
        version_dir = self.model_dir / f"version_{next_version}"
        version_dir.mkdir(parents=True)
        return ArtifactVersion(version_dir)


class ArtifactVersion:
    """One self-contained training run: config snapshot, per-ticker model,
    regime states, transition matrices, and a metrics summary -- everything
    needed to know what this run was and inspect what it produced, without
    cross-referencing the (possibly since-changed) live config."""

    def __init__(self, path: Path):
        self.path = path
        self._metrics: dict[str, dict] = {}

    def write_config(self, config: DictConfig | dict) -> Path:
        if isinstance(config, DictConfig):
            config = cast(dict, OmegaConf.to_container(config, resolve=True))
        file = self.path / "config.yaml"
        file.write_text(OmegaConf.to_yaml(config))
        return file

    def write_model(self, ticker: str, model: RegimeModel) -> Path:
        models_dir = self.path / "models"
        models_dir.mkdir(exist_ok=True)
        file = models_dir / f"{ticker}.pkl"
        joblib.dump(model, file)
        return file

    def write_regime_states(self, ticker: str, labeled_df: pd.DataFrame) -> Path:
        states_dir = self.path / "regime_states"
        states_dir.mkdir(exist_ok=True)
        file = states_dir / f"{ticker}.csv"
        labeled_df.to_csv(file)
        return file

    def write_transition_matrices(
        self, ticker: str, matrices: list[pd.DataFrame]
    ) -> list[Path]:
        matrices_dir = self.path / "transition_matrices"
        matrices_dir.mkdir(exist_ok=True)
        paths = []
        for layer_idx, trans_df in enumerate(matrices):
            file = matrices_dir / f"{ticker}_layer{layer_idx}.csv"
            trans_df.to_csv(file)
            paths.append(file)
        return paths

    def record_metric(self, ticker: str, **metrics) -> None:
        self._metrics[ticker] = {
            **metrics,
            "recorded_at": datetime.now(UTC).isoformat(),
        }

    def flush_metrics(self) -> Path:
        file = self.path / "metrics.json"
        file.write_text(json.dumps(self._metrics, indent=2))
        return file
