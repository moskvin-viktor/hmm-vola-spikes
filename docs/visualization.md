# Visualization

Results live under `artifacts/{ModelName}/version_N/` after a `fit_model.py`
run (config snapshot, per-ticker model pickles, regime-state CSVs,
transition-matrix CSVs, `metrics.json`). They're explored with a
[marimo](https://marimo.io/) notebook, not a standalone web app.

```bash
hatch run notebook  # opens notebooks/ in the marimo editor
```

`notebooks/explore_artifacts.py` reads runs straight from `artifacts/` via
`PathManager` (`src/hmmstock/path_manager.py`) — no training, no network
access. Pick a model class, a version, and a ticker from its dropdowns to see:

- The run's config and `metrics.json` (chosen `n_components`, `cv_score`)
- Returns and rolling volatility scattered by regime
- A transition-matrix heatmap and expected regime duration
  (`1 / (1 - self_transition_prob)`, in trading days)
- Per-regime feature means, and which regime captures high-quantile
  volatility/return events
- An ANOVA test (`hmmstock.stat_testing.test_returns_anova`) for whether mean
  returns actually differ across regimes

For `HierarchicalHMMModel`, `sub_level_state` is grouped by `top_level_state`
throughout — each top-level regime has its own independently-fitted sub-HMM,
so sub-state numbers aren't comparable across different top-level regimes.

See `docs/findings.md` for write-ups from an actual training sweep.
