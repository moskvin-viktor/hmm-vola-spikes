#!/usr/bin/env bash
# Trains all three RegimeModel variants against real data in one sweep,
# via fit_model.py (Hydra), each with a wider component search than the
# config defaults -- those were never actually explored past, not
# evidence their ceiling is the right number of regimes:
#   - HMMModel:            max_components  2 -> 6
#   - LayeredHMMModel:     layer0 max_components 4 -> 6 (layer1 stays
#                          fixed at 2 -- it's deliberately a coarse
#                          summary on top of layer0, not meant to explore)
#   - HierarchicalHMMModel: top_layer max_components 3 -> 6 (sub_layer
#                          stays 2-3 -- per-regime partitions are already
#                          small, more components there overfits faster
#                          than it helps)
#
# n_fits is trimmed from the config default (100) to 20 restarts per
# (model, n_components) candidate for all three, to keep the wider sweep
# tractable -- walk-forward CV itself (n_splits) is left untouched.
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
  model.LayeredHMMModel.layers.0.max_components=6 \
  model.LayeredHMMModel.n_fits=20 \
  model.HierarchicalHMMModel.top_layer.max_components=6 \
  model.HierarchicalHMMModel.n_fits=20 \
  "$@"
