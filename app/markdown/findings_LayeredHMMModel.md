# Findings: Layered HMM Model

## Overview

As expected, **Layer 0** of the Layered HMM behaves **identically to the standard HMM model**, and therefore is **not the focus** of this analysis. The added complexity lies in **Layer 1 (upper level)**, which aims to model higher-order regime structures.

However, the results from Layer 1 raise both **interesting observations** and **methodological concerns**.

---

## 1. Return vs. Volatility: Still No Statistical Significance

Across tickers, even at the upper layer, there is **no consistent statistical significance** between **returns and volatility regimes**. This suggests that **adding another hierarchical layer does not improve explanatory power** in this dimension.

---

## 2. Ticker-Specific Observations

### **AAPL**

- The upper layer reveals **two regimes**, roughly corresponding to **high** and **low volatility** states.
- Regimes are **sticky**, showing strong self-transition probabilities.
- **100% volatility quantile capture**, especially effective in **vola20** and **market volatility proxy**.
- Interpretation: The Layered HMM provides **clean volatility segmentation**, but no new insight beyond the base model.

---

### **MSFT**

- Results show **unexpected behavior**:
  - The model identifies a statistically significant difference in returns across regimes — **a surprising outlier**.
  - The regime labeled as "low volatility" (Regime 1) captures **all market volatility extremes**, contradicting its own internal metrics.
  - Fit for **historical volatilities (vola10, vola20)** is **poor**, and the model appears to be **reversed** in logic.
  - Regimes are **anti-correlated** with **vola20** — counterintuitive.

> ⚠️ Interpretation: These results likely indicate **anomaly or instability** in the training process for MSFT under the Layered HMM structure.

---

### **GSPC**

- Regime 0: Higher volatility  
- Regime 1: Lower volatility  
- No significant return-volatility relationship
- Poor fit to volatility indicators — **ineffective segmentation**

> Interpretation: The Layered HMM **fails to capture meaningful regimes** for GSPC. It adds complexity without performance gain.

---

### **AMZN**

- Regime 0: Higher volatility — **100% volatility capture**
- Regime 1: Lower volatility — **anti-correlated** with volatility indicators
- No significance in return segmentation
- Structure is clear, but again **adds no new explanatory power** beyond Layer 0

---

## 3. Model Fit Metric (Lower is Better)

| Tickers Trained | Layered HMM Model Fit |
|------------------|-----------------------|
| 1 Ticker         | 3493.59               |
| 2 Tickers        | 2517.33               |
| 3 Tickers        | 1884.79               |
| 4 Tickers        | 2654.21               |
| 5 Tickers        | 1576.99               |

> Interpretation: The metric **improves as more tickers are trained**, suggesting some **shared learning**. However, the fit values are **significantly worse than the base HMM model**, particularly on fewer tickers.

---

## 4. Conclusion: Is Layered HMM Worth It?

While the **Layered HMM adds hierarchical structure**, its **practical benefits appear limited**:

- For **AAPL and AMZN**, it captures volatility reasonably well, but adds little beyond HMM.
- For **MSFT and GSPC**, results are **confusing or contradictory**.
- The **return-volatility relationship remains weak or nonexistent**.
- Training metrics suggest **high loss values**, raising doubts about stability and generalization.

### ❗ Verdict:
> The **Layered HMM structure introduces more complexity** but does **not consistently improve interpretability or performance**. In some cases, it leads to **incoherent or misleading regime classification**. Further tuning or alternative hierarchical designs may be required.