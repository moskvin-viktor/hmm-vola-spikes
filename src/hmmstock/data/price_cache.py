import pickle
from pathlib import Path

import pandas as pd


class PriceCache:
    """Pickle-backed cache for a fetched close-price DataFrame."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> pd.DataFrame | None:
        if not self.path.exists():
            return None
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def save(self, data: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(data, f)
