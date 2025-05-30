# Findings: Layered HMM Model

## Overview

As expected, **Layer 0** of the Layered HMM behaves identically to the **standard HMM model**, serving as the base regime classifier. The added complexity emerges in **Layer 1**, which attempts to capture **higher-order regime structures** using posterior probabilities from the first layer as new features.

> One natural hypothesis is that increasing the number of components in **Layer 0** could provide a richer regime encoding for Layer 1. However, this appears to **lead to overfitting** and increased noise without delivering clear improvements.

### General Observations

- **No significant return–volatility correlation** was observed, with the exception of **AMZN**, which exhibited weak significance — likely due to idiosyncratic structure in its returns.
- Layer 1 models often **simplify** the structure learned in Layer 0 (e.g., intermediate regimes disappear).
- The **Layered HMM provides additional structure**, but the **quantitative improvement is marginal**.
- High **capture rates** of volatility spikes (>95%) remain consistent across layers.
- Layered HMM regimes tend to be **more stable/sticky**, but not necessarily more **interpretable**.

---

## AAPL

- **Three regimes**.
- Regimes are **equally represented** and **sticky**.
- Nearly **100% volatility spike capture**.
- Correlations with **vol20** and **market volatility**, though **lower** than base HMM.
- **No major improvement** in explanatory power compared to base HMM.

---

## MSFT

- Again, **only two regimes**; intermediate layer smoothed out.
- Regimes are **sticky** and equally sized.
- **Excellent capture rate** for volatility spikes.
- Correlated primarily with **vol20** and **market volatility**.
- **Slightly simplified model**, yet improves in **tail capturing**.
- Possibly a **marginally better fit** than base HMM.

---

## GSPC

- **Three regimes**.
- Perfect **volatility spike capture**.
- Regimes remain **correlated with vol20 and market volatility**.
- Model **looks nearly identical to base HMM** in behavior.
- **No added insight** from additional layer.

---

## AMZN

- **Three regimes**.
- **Sticky and equal** distribution.
- Nearly **100% volatility spike capture**.
- Some correlation with **vol10**, but low with longer horizons.
- Returns show **weak significance**, possibly due to data structure.
- Similar performance to base HMM, but **slightly stronger structure** observed.

---

## Final Thoughts

- **Layered HMM introduces architectural depth**, but **does not yield substantial performance gains** in volatility regime modeling.
- The additional layer **simplifies** regime space rather than enriching it.
- **Volatility spike detection remains strong**, but interpretability and correlation strength with external factors do not improve noticeably.
- Possible future direction: experiment with **3-layer structures** or **feature engineering** between layers to prevent over-smoothing.
- Alternatively, Layered HMM may be better suited for **non-volatility tasks** where **higher-order transitions** are expected to matter more.