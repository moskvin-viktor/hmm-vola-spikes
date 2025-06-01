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

### 3. (Planned) Hierarchical HMM (HHMM)
- Embeds an HMM inside each top-level state.
- Models **nested dynamics**, such as market phase → sub-regime.


## Evaluation: Log-Likelihood with Entropy Regularization


## Features

- Configurable data pipeline for fetching and processing stock and volatility data
- Volatility proxy transformation via a dedicated abstraction (e.g., raw, smoothed, returns, normalized)
- HMM model training for multiple stocks with support for volatility-based state relabeling
- Interactive Dash app for exploring HMM results with plots
---

## Getting Started

### 1. Set Up the Environment

Install dependencies using your preferred Python package manager. All dependencies are listed in `pyproject.toml`.

### 2. Train HMM Models

Use the following command to fetch data, compute features, and fit models:

```bash
python fit_model.py
```

This will:

- Download historical data (or load from cache) using ```yfinance```

- Compute features (e.g., log returns, volatility)

- Train HMMs with up to N states (configurable)

- Save regime-labeled outputs and transition matrices to the results/ directory

3. Launch the Dash app
Start the interactive dashboard by running:

```bash
python app/app.py
```

Then open http://127.0.0.1:8050/ in your browser.

4. Alternatively, you can check up the ``doc`` folder for the theoretical insights.