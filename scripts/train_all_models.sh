#!/usr/bin/env bash
# Trains all three RegimeModel variants against real data in one sweep,
# via fit_model.py (Hydra): HMMModel with a wider component search than
# the config default (max_components 2 -> 6, so it's not artificially
# capped at 2 states), LayeredHMMModel, HierarchicalHMMModel. n_fits is
# trimmed from the config default (100) to 20 restarts per (model,
# n_components) candidate for all three, to keep the wider HMM sweep
# tractable -- walk-forward CV (n_splits, test_size) is left untouched.
#
# Writes a new artifacts/{ModelName}/version_N/ for each model, same as
# running fit_model.py per model class would. Any extra Hydra overrides
# passed to this script are forwarded, e.g.:
#   scripts/train_all_models.sh data.tickers='[AAPL,MSFT]'
set -euo pipefail
cd "$(dirname "$0")/.."

hatch run python fit_model.py \
  model_class=all \
  model.HMMModel.max_components=6 \
  model.HMMModel.n_fits=20 \
  model.LayeredHMMModel.n_fits=20 \
  model.HierarchicalHMMModel.n_fits=20 \
  "$@"
