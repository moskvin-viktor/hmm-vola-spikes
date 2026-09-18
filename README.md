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

Hyperparameters (`n_components`, random seed) are selected by **walk-forward cross-validation**, not a single lucky/unlucky split: `config/model/default.yaml`'s `split.test_size` (default 0.15) carves off a chronological holdout from the end of each ticker's series, untouched during selection; the rest is split into `split.n_splits` (default 5) expanding-window folds (`src/hmmstock/data/splitter.py`, via sklearn's `TimeSeriesSplit`) -- each fold trains on everything before a cutoff and validates on the chunk immediately after it, so nothing ever validates on data older than its own training set. The winning config is scored once on the untouched test holdout for an honest number, then refit on all data (train + CV + test) for the model that actually gets saved to `artifacts/`.

Which scoring metric drives fold selection is set by `evaluation_metric` in `config/model/default.yaml` (`LogLikelihoodWithEntropy` or `BICMetric`, built by `build_evaluation_metric()` in `src/hmmstock/metrics.py`):
- **`LogLikelihoodWithEntropy`**: validation log-likelihood, normalized by sequence length, plus an entropy term over state-occupancy (weighted by `entropy_weight`) that favors more balanced use of the states. Scored on each fold's validation slice.
- **`BICMetric`**: the model's Bayesian Information Criterion, scored on each fold's *training* slice -- that's the point of BIC, penalizing training likelihood by parameter count as a stand-in for held-out performance, without needing validation data at all.

Each trained ticker's `metrics.json` entry (under `artifacts/{Model}/version_N/`) records both `cv_score` (what selected the winning config) and `best_score` (that config's honest score on the untouched test holdout) -- distinct numbers, not the same value serving double duty.


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

`scripts/train_all_models.sh` trains all three in one sweep with a wider HMM component search (`max_components` 2 -> 6, so it's not capped at a 2-state model) and fewer random restarts per candidate to keep that tractable -- CV/test-holdout methodology (`n_splits`, `test_size`) is untouched. Extra Hydra overrides are forwarded, e.g. `scripts/train_all_models.sh data.tickers='[AAPL,MSFT]'`.

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