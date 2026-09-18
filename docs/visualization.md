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

See `docs/findings.md` for write-ups from an actual training sweep, or the
<a href="notebook.html">live notebook snapshot</a> for a static, real-data
render of `explore_artifacts.py` itself (Plotly charts stay pan/zoom-able; the dropdowns
are frozen at their default values since there's no running Python kernel
behind a static export — run `hatch run notebook` locally for the live,
interactive version).

The snapshot isn't built from a `marimo export html-wasm` in-browser version:
`hmmlearn` (a compiled extension the models depend on) has no Pyodide/WASM
build, so the notebook can't import `hmmstock` at all inside a WASM sandbox.
The snapshot is instead rendered server-side, in CI, against a small
committed example dataset (`examples/artifacts/`, config/metrics/regime-state
CSVs only, no model pickles) — the same role `examples/` played for the old
Dash app.
