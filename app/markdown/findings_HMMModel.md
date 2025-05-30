# Findings: HMM Model

## 1. Volatility Regimes

The model identifies **three distinct volatility regimes** across all tickers:

- **Regime 0**: Low volatility  
- **Regime 1**: Moderate (average) volatility  
- **Regime 2**: High volatility  

These regimes align well with the behavior of common volatility indicators across time.

---

## 2. Returns vs. Volatility

Although visual inspection suggests a **negative correlation between volatility and returns**, the **ANOVA test** results indicate that these differences **are not statistically significant** across regimes for any ticker. This implies that the perceived relationship may be an illusion due to randomness or sample noise, rather than a robust pattern.

---

## 3. Transition Matrix Interpretation

The **transition matrices** reveal that all regimes are **relatively sticky** — meaning:

> Once the model enters a regime, there's a high probability it stays there for some time before transitioning.  

This is consistently observed across all tickers:

- **AAPL, MSFT, GSPC, AMZN**: All show strong self-transition probabilities, indicating **stable regime persistence** over time.

---

## 4. Volatility Capture by Regimes

The model's ability to capture volatility quantiles varies across tickers:

- **AAPL**:  
  - Strong performance.  
  - Regime 2 captures nearly all of **vola10**, **vola20**, and over **95%** of the **market volatility proxy**.

- **MSFT**:  
  - Moderate performance.  
  - **vola20** is undercaptured (~65%), but **100%** of market volatility is captured.  
  - Overall, effective regime-volatility alignment.

- **GSPC**:  
  - Excellent regime separation.  
  - **Regime 2** captures **100%** of volatility signals.  
  - Other regimes display **below-average normalized volatilities** (i.e., < 0), showing clear volatility contrasts.

- **AMZN**:  
  - Very effective.  
  - **Regime 2** captures >90% of all volatility indicators.  
  - Other regimes also have mean normalized volatilities below zero.

---

## 5. Regime Correlations with Volatility Metrics

Each ticker shows unique correlations between its regimes and specific volatility indicators:

- **AAPL**: Regimes correlate most strongly with **vola20**.
- **MSFT**: Regimes align closely with **market volatility**.
- **GSPC**: Similar to MSFT, the model captures **market-level volatility** effectively.
- **AMZN**: Strongest regime association with **vola10**.

---

## 6. Regime Quality Metric (Lower is Better)

| Ticker | Metric Value |
|--------|--------------|
| AAPL   | -124.84      |
| MSFT   | 65.14        |
| GSPC   | 219.12       |
| AMZN   | -132.05      |

These values provide a comparative measure of regime model fit, with **lower values indicating better alignment** between the HMM regimes and market dynamics.

---