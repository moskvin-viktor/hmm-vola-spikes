# HMM Stock Regime Analysis

This project implements and extends Hidden Markov Models (HMMs) to model regime-switching behavior in financial time-series data, particularly for detecting volatility regimes. Below, we outline the theoretical foundations behind standard, layered, and hierarchical HMMs and their role in capturing temporal dynamics.

---

## Live Demo

You can explore the interactive visualizations of the HMM-based analysis directly in your browser:

[https://hmm-vola-spikes.onrender.com/](https://hmm-vola-spikes.onrender.com/)

## What is a Hidden Markov Model?

A **Hidden Markov Model (HMM)** is a **probabilistic model** that assumes:

- An underlying process (hidden states) evolves over time following a **Markov process**.
- The system emits **observable signals** (data) which are probabilistically related to the hidden states.

### Key Components
- **Hidden States (`q_t`)**: Unobservable modes (e.g., "low volatility", "high volatility").
- **Observations (`x_t`)**: Measurable features (e.g., returns, VIX).
- **Transition Matrix (`A`)**: Probabilities of switching between states.
- **Emission Probabilities (`B`)**: Likelihood of observations given a hidden state.


## Model Variants in This Project

### 1. Standard Gaussian HMM
- Single-layer probabilistic model.
- Captures changes in distribution (mean/variance) across regimes.
- Implemented using [`hmmlearn`](https://hmmlearn.readthedocs.io/).

### 2. Layered HMM (LHMM)
- Stacks multiple HMMs in layers.
- Each layer takes the **posterior state probabilities** from the previous as input.
- Enhances the ability to learn hierarchical/abstract structure.

### 3. Hierarchical HMM (HHMM)
- Embeds an HMM inside each top-level state.
- Models **nested dynamics**, such as market phase → sub-regime.
- Implemented and trainable via `fit_model.py model_class=HierarchicalHMMModel`; not yet wired into the Dash app (its transition-matrix view assumes integer layer indices, which don't apply to HHMM's `top_level_state`/`sub_level_state` layers).


## Evaluation

Hyperparameters (`n_components`, random seed) are selected by **walk-forward cross-validation**, not a single lucky/unlucky split: `config/model/default.yaml`'s `split.n_splits` (default 5) expanding-window folds (`src/hmmstock/data/splitter.py`, via sklearn's `TimeSeriesSplit`) -- each fold trains on everything before a cutoff and validates on the chunk immediately after it, so nothing ever validates on data older than its own training set. `n_splits` is capped down automatically (`adaptive_n_splits`) when there's too little data to support it -- e.g. `HierarchicalHMMModel`'s per-regime sub-HMMs, already partitioned down to a handful of samples. The winning config's average CV score (`cv_score` in `metrics.json`) *is* the reported score -- there's no separate holdout-scoring pass; walk-forward CV already never looks at future data relative to what it trained on, so a second held-out slice on top of it was redundant complexity. The deployed model (saved to `artifacts/`) is then refit on all available data for maximal information.

Which scoring metric drives fold selection is set by `evaluation_metric` in `config/model/default.yaml` (`LogLikelihoodWithEntropy` or `BICMetric`, built by `build_evaluation_metric()` in `src/hmmstock/metrics.py`):
- **`LogLikelihoodWithEntropy`**: validation log-likelihood, normalized by sequence length, plus an entropy term over state-occupancy (weighted by `entropy_weight`) that favors more balanced use of the states. Scored on each fold's validation slice.
- **`BICMetric`**: the model's Bayesian Information Criterion, scored on each fold's *training* slice -- that's the point of BIC, penalizing training likelihood by parameter count as a stand-in for held-out performance, without needing validation data at all.

Regime labels (`regime_layer0`, `top_level_state`, ...) are always ordered by increasing **total variance** -- the trace of each state's fitted covariance matrix, via `RegimeModel._volatility_rank_map()` -- not by re-deriving dispersion from raw observations, which would blend unrelated feature units (returns, several differently-scaled rolling-volatility windows, a market proxy) into one meaningless number once there's more than one feature column. `HierarchicalHMMModel`'s `sub_level_state` is relabeled the same way, but independently per top-level regime's own sub-HMM -- "sub-state 0" always means "the lowest-variance sub-state within that particular regime," not a value comparable across different regimes' sub-HMMs; group by `top_level_state` before comparing.


## Features

- Configurable data pipeline for fetching and processing stock and volatility data
- Volatility proxy transformation via a dedicated abstraction (e.g., raw, smoothed, returns, normalized)
- HMM model training for multiple stocks with support for volatility-based state relabeling
- Interactive Dash app for exploring HMM results with plots
---

## Getting Started

### 1. Set Up the Environment

This project uses [Hatch](https://hatch.pypa.io/) for environment and dependency management. All dependencies are declared in `pyproject.toml`.

```bash
pip install hatch
hatch env create
```

### 2. Train HMM Models

Configuration is managed with [Hydra](https://hydra.cc/), composed from `config/config.yaml` (`config/data/default.yaml` + `config/model/default.yaml`). Use the following command to fetch data, compute features, and fit models:

```bash
hatch run python fit_model.py
```

Which model trains is set by `model_class` in `config/config.yaml` (default `LayeredHMMModel`). Any config value can be overridden on the CLI, e.g.:

```bash
hatch run python fit_model.py model_class=HMMModel model.HMMModel.max_components=3
hatch run python fit_model.py model_class=all  # train every model
```

`scripts/train_all_models.sh` trains all three in one sweep with a wider component search for HMM (`max_components` 2 -> 6), `LayeredHMMModel`'s layer0 (4 -> 6), and `HierarchicalHMMModel`'s top layer (3 -> 6) -- none capped at their config defaults' ceilings -- and fewer random restarts per candidate to keep that tractable. CV methodology (`n_splits`) is untouched. Extra Hydra overrides are forwarded, e.g. `scripts/train_all_models.sh data.tickers='[AAPL,MSFT]'`.

This will:

- Download historical data (or load from cache) using ```yfinance``` -- or swap in `hmmstock.data.fred_client.FredClient` (`run_pipeline(order, client=FredClient())`) to pull series straight from [FRED](https://fred.stlouisfed.org/) instead (e.g. `VIXCLS`, `DGS10`, `SP500`), no API key required. Both clients share the same interface, so either drops into the pipeline unchanged.

- Compute features (e.g., log returns, volatility)

- Train HMMs with up to N states (configurable)

- Save a new versioned run under `artifacts/{ModelName}/version_N/` (gitignored — regenerated locally, not shipped), each containing a snapshot of the config used, per-ticker model pickles, regime-state CSVs, transition-matrix CSVs, and a `metrics.json` summary. Nothing is ever overwritten — every `fit_model.py` run gets its own version.

3. Launch the Dash app
Start the interactive dashboard by running:

```bash
hatch run python app/app.py
```

Then open http://127.0.0.1:8050/ in your browser.

By default the dashboard reads from `examples/`, a small pre-computed dataset checked into the repo so the app (and the live demo) works without needing `yfinance` API access or a training run first. The Dash app is due to be replaced and doesn't yet read from the new `artifacts/` layout produced by `fit_model.py`.

### 5. Explore Results in a Notebook

[marimo](https://marimo.io/) notebooks read trained runs straight from `artifacts/` via `PathManager` (`src/hmmstock/path_manager.py`) — pick a model, a version (run), and a ticker, and see its config, metrics, regime states, and transition matrices. No training, no network access.

```bash
hatch run notebook  # opens notebooks/ in the marimo editor
```

`notebooks/explore_artifacts.py` is a starting example. Run `fit_model.py` at least once first so there's something under `artifacts/` to look at.

### 6. Run Tests

```bash
hatch run test
```

4. Alternatively, you can check up the ``doc`` folder for the theoretical insights.