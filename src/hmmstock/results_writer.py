from pathlib import Path

import pandas as pd


class ResultsWriter:
    """Writes regime-labeled CSVs and per-layer transition matrices to disk."""

    def __init__(self, csv_dir: Path, transition_matrices_dir: Path):
        self.csv_dir = csv_dir
        self.transition_matrices_dir = transition_matrices_dir

    def write_regime_states(self, ticker: str, labeled_df: pd.DataFrame) -> Path:
        ticker_dir = self.csv_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ticker_dir / "regime_states.csv"
        labeled_df.to_csv(csv_path)
        return csv_path

    def write_transition_matrices(
        self, ticker: str, matrices: list[pd.DataFrame]
    ) -> list[Path]:
        self.transition_matrices_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for layer_idx, trans_df in enumerate(matrices):
            csv_path = (
                self.transition_matrices_dir
                / f"{ticker}_transition_matrix_layer{layer_idx}.csv"
            )
            trans_df.to_csv(csv_path)
            paths.append(csv_path)
        return paths
