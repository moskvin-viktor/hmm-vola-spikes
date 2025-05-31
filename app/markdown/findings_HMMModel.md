# Findings: HMM Model

## 1. Volatility Regimes

The model identifies **three distinct volatility regimes** across all tickers:

- **Regime 0**: Low volatility  
- **Regime 1**: Moderate to high volatility  

These regimes correspond well with the evolution of common volatility indicators over time.

---

## 2. Returns vs. Volatility

While visual inspection suggests a **negative correlation between volatility and returns**, the **ANOVA test results** indicate that these differences are **not statistically significant** across regimes for any ticker.

> This implies the perceived relationship may be driven by randomness or sample noise, rather than a consistent structural pattern.

It may also be attributed to **limited sample sizes**: periods of high volatility may coincide with lower returns, but not frequently enough to yield statistical significance.

---

## 3. Transition Matrix Interpretation

The **transition matrices** show that all regimes are **highly persistent**:

> Once a regime is entered, the model assigns a **high probability of staying** in that regime for several time steps.

This stickiness is consistent across all evaluated tickers:

- **AAPL, MSFT, GSPC, AMZN**: Exhibit **strong self-transition probabilities**, implying stable regime durations.

---

## 4. Volatility Capture by Regimes

The model's regime classification captures distinct volatility quantiles with varying strength across tickers:

- **AAPL**:  
  - Strong performance with excellent separation across regimes.  

- **MSFT**:  
  - Clear and consistent regime boundaries.  

- **GSPC**:  
  - Well-defined volatility regimes.  

- **AMZN**:  
  - Good performance, but less consistent when using market volatility as a proxy.

---

## 5. Regime Quality Metrics

These metrics assess the quality of regime modeling, balancing entropy and log-likelihood (lower score = better):

| Ticker | Best Seed | Entropy | Normalized LL | Score   |
|--------|-----------|---------|---------------|---------|
| AAPL   | 2         | 0.6432  | -1.3557       | 1.2172  |
| MSFT   | 3         | 0.3222  | 0.0125        | 1.3015  |
| GSPC   | 3         | 0.4652  | 0.6507        | 2.5115  |
| VIX    | 0         | 0.6432  | -3.6303       | -1.0574 |
| AMZN   | 1         | 0.4585  | -0.9859       | 0.8483  |

> The **Score** balances regime stability (entropy) and data fit (log-likelihood). **Lower scores are better**, though interpretability remains important.

---

## 6. Observations on Market Proxy (VIX)

One advantage of using HMMs over threshold-based regime classifiers is their ability to:

- Dynamically capture **trend-based transitions**  
- Avoid rigid boundaries by learning **data-driven regime shifts**

The model identifies different behavior in **historical volatility vs. implied volatility (VIX)**. This could reflect either:

- **Lagging effects**: Historical volatility often trails implied volatility  
- **Model limitations**: VIX may not perfectly represent forward-looking volatility for all tickers or timeframes

---

## Summary

The standard HMM offers a **robust and interpretable framework** for modeling volatility regimes. While not all statistical associations (like volatility vs. returns) are confirmed, the model reliably captures key structural patterns in market volatility across major assets.