# Findings

Results from a full training sweep (`scripts/train_all_models.sh`) across all three
model variants and five tickers (AAPL, MSFT, GSPC, VIX, AMZN; 2020-01-01 to
2024-12-31), run *after* the volatility-relabeling fix, the wider component
search, `HierarchicalHMMModel` relabeling, and the switch to walk-forward
CV-only scoring (`cv_score` everywhere — see the project history for details).
Numbers below are from that run's `artifacts/*/version_0/`; regenerate with
`scripts/train_all_models.sh` and re-read via `notebooks/explore_artifacts.py`
or `PathManager` to get current ones.

This supersedes `app/markdown/findings_*.md` (the old Dash app's static
write-ups), which predate the relabeling fix and the wider component search —
their headline claim ("regimes are highly persistent but returns don't differ
significantly across them, for any ticker") is only half true with current
results.

## 1. Regime count is data-driven now, not config-capped

Every prior run capped `HMMModel` at `max_components: 2`, `LayeredHMMModel`'s
layer0 at 4, and `HierarchicalHMMModel`'s top layer at 3 — every ticker landed
on the ceiling, which looked like "2 (or 3, or 4) regimes is what the data
wants" but was actually just "the ceiling is what the data wants." Widening
all three to 6 (`scripts/train_all_models.sh`) changed the answer:

| Ticker | Components chosen |
|---|---|
| AAPL | 4 |
| MSFT | 5 |
| GSPC | 4 |
| VIX  | 5 |
| AMZN | 6 |

None of the five hit the new ceiling either, so 6 was wide enough this time —
worth periodically re-checking with an even wider sweep.

## 2. HMMModel, LayeredHMMModel's layer0, and HierarchicalHMMModel's top layer agree exactly

`LayeredHMMModel`'s layer0 and `HierarchicalHMMModel`'s top-level HMM are
structurally the same fit as standalone `HMMModel` — same `X`, same CV
config, same evaluation metric. With the component-search ranges now aligned
(all widened to 6), they land on the *same* component count and the *same*
ANOVA p-value, per ticker, in this run:

| Ticker | K | ANOVA p-value | Significant at 5%? |
|---|---|---|---|
| AAPL | 4 | 0.352 | no |
| MSFT | 5 | 0.644 | no |
| GSPC | 4 | 0.297 | no |
| VIX  | 5 | 0.871 | no |
| AMZN | 6 | **0.032** | **yes** |

This is a useful sanity check on the pipeline — three independently coded
model classes converging on identical results for what's mathematically the
same sub-problem is a good sign the CV/relabeling machinery is behaving
consistently, not a sign the three models are redundant (they diverge past
this shared "base layer": `LayeredHMMModel` builds a second layer on top,
`HierarchicalHMMModel` adds sub-regime structure).

## 3. AMZN is the one ticker where regime actually predicts returns

Every ticker except AMZN keeps the old finding: regimes are clearly separated
by volatility, but mean returns don't differ significantly across them
(ANOVA p > 0.05) — regime tells you about volatility, not about which way
price is about to move. AMZN breaks that pattern (p = 0.032). Whether that's
a real AMZN-specific effect or one significant result out of five tickers
tested (a multiple-comparisons false positive at ~5% is expected roughly
1-in-20 times) isn't something a single ANOVA run can distinguish — worth
rechecking on a different date range or with a held-out period before reading
much into it.

## 4. Regime persistence is real and substantial

Expected regime duration (`1 / (1 - self_transition_prob)`, in trading days)
varies widely by ticker and state, from ~3 days up to ~43 days:

| Ticker | Expected durations (days), low- to high-variance state |
|---|---|
| AAPL | 26, 41, 23, 15 |
| MSFT | 13, 7, 7, 16, 12 |
| GSPC | 14, 15, 31, 43 |
| VIX  | 19, 5, 28, 30, 20 |
| AMZN | 3, 23, 20, 3, 21, 5 |

No ticker has a near-instant-switching state (which would show up as a
duration near 1) — every regime, once entered, tends to persist for at least
a few trading days, consistent with genuine regime structure rather than
frame-to-frame noise. AMZN's alternating short/long pattern (3, 23, 20, 3, 21,
5) is the most volatile-*of*-its-volatility-regimes ticker in the set.

## 5. HierarchicalHMMModel's sub-level structure is mostly inert

Grouping `sub_level_state` by `top_level_state` (never pool across groups —
each top-level regime has its own independently-fitted sub-HMM, see
`HierarchicalHMMModel`'s docstring):

| Ticker | Top-level regimes using >1 sub-state |
|---|---|
| AAPL | 0 / 4 |
| GSPC | 0 / 4 |
| MSFT | 1 / 5 |
| VIX  | 1 / 5 |
| AMZN | 2 / 6 |

For AAPL and GSPC, every single top-level regime's sub-HMM collapsed to one
effectively-used state despite `sub_layer` allowing 2-3 — the extra
hierarchical layer bought nothing beyond the top level for those two tickers
under CV/BIC selection. AMZN and VIX show some real sub-structure. This isn't
a bug (relabeling and scoring are both working correctly here — see the
project history's fixes); it's the model genuinely deciding one state is
enough for most partitions. Whether that means the sub-layer's search range
needs revisiting, or whether it means (as it plausibly does) that most of the
interesting structure in this feature set really is captured at the top
level, is an open question.

## 6. Labeling rank vs. any single feature

Regime labels (`regime_layer0`, `top_level_state`, ...) are ordered by total
variance (trace of the fitted covariance) across *all* features together, not
by `vol_20` alone. For 3 of 5 tickers (AAPL, GSPC, AMZN) the labels still come
out monotonic in `vol_20` specifically; MSFT and VIX don't. That's expected,
not a regression — a state can have lower `vol_20` but higher variance
elsewhere (another vol window, the market proxy) and rank higher overall.
Don't read "state 3" as "more `vol_20`-volatile than state 1" without checking
the actual per-state feature means (`notebooks/explore_artifacts.py`'s
"Feature means by regime" panel does this).
