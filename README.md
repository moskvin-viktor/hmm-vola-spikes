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

Model selection (choosing among the random restarts and component counts in `config/model/default.yaml`) is scored per fit using `LogLikelihoodWithEntropy`: validation log-likelihood, normalized by sequence length, plus an entropy term over state-occupancy (weighted by `entropy_weight` in `config/model/default.yaml`) that favors more balanced use of the states.

A second metric, `BICMetric` (the model's Bayesian Information Criterion), also exists in `src/hmmstock/metrics.py` but isn't currently wired up anywhere — `config/model/default.yaml`'s `evaluation_metric: "BICMetric"` key is unused; only `LogLikelihoodWithEntropy` runs.


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

This will:

- Download historical data (or load from cache) using ```yfinance```

- Compute features (e.g., log returns, volatility)

- Train HMMs with up to N states (configurable)

- Save regime-labeled outputs and transition matrices to the `results/` directory (gitignored — regenerated locally, not shipped)

3. Launch the Dash app
Start the interactive dashboard by running:

```bash
hatch run python app/app.py
```

Then open http://127.0.0.1:8050/ in your browser.

By default the dashboard reads from `examples/`, a small pre-computed dataset checked into the repo so the app (and the live demo) works without needing `yfinance` API access or a training run first. Run `fit_model.py` and point `HMMResultVisualization` at `results/` to see your own data instead.

### 5. Run Tests

```bash
hatch run test
```

4. Alternatively, you can check up the ``doc`` folder for the theoretical insights.