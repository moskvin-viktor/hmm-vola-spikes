from pathlib import Path

import joblib

from .models.base import RegimeModel


class ModelStore:
    """Joblib-backed persistence for a dict of trained RegimeModel instances."""

    def __init__(self, directory: Path, filename: str):
        self.file = directory / filename

    def save(self, models: dict[str, RegimeModel]) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, self.file)

    def load(self) -> dict[str, RegimeModel]:
        try:
            return joblib.load(self.file)
        except FileNotFoundError:
            return {}
