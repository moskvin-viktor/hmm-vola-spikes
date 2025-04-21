# HMM Stock Regime Analysis

This project applies Hidden Markov Models (HMMs) to model and analyze stock market regimes using historical returns and volatility features. It includes data handling, model training, configurable preprocessing, and a dashboard for interactive visualization.

---

## Project Structure

.   
    ├── app/ # Dash app for interactive visualization 
    ├── config/ # YAML configuration files for models and plots 
    ├── data/ # Directory for raw or processed datasets 
    ├── data_cache/ # Cached data (e.g., from yfinance) 
    ├── hmm_python39/ # Conda environment setup 
    ├── results/ # Output files: regime states, transition matrices 
    ├── src/ # Source code: data loaders, model classes, utilities 
    ├── environment.yml # Conda environment specification 
    ├── fit_model.py # Entry point for training HMMs 
    ├── pyproject.toml # Project metadata and dependencies 
    ├── README.md # Project readme (this file) 
    ├── requirements.txt # PIP requirements (alternative to environment.yml) 

---

## Features

- Configurable data pipeline for fetching and processing stock and volatility data
- Volatility proxy transformation via a dedicated abstraction (e.g., raw, smoothed, returns, normalized)
- HMM model training for multiple stocks with support for volatility-based state relabeling
- Interactive Dash app for exploring HMM results with plots
---

## Getting Started

### 1. Set up the environment

Install dependencies using either `conda` or `pip`.

#### Option A: Conda

```bash
conda env create -f environment.yml
conda activate hmm_env
```

### Option B: pip

```
pip install -r requirements.txt
```

### 2. Train the HMM models

Run the following to process stock data and train models:

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