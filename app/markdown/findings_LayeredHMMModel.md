# Findings: Layered HMM Model

## Overview

As expected, **Layer 0** of the Layered Hidden Markov Model (HMM) behaves identically to the standard HMM, serving as the base regime classifier. We introduced architectural depth by adding **Layer 1**, which utilizes the posterior probabilities from Layer 0—combined with the original features—as inputs. This approach aims to uncover **higher-order regime structures**.

> A key hypothesis is that increasing the number of components in **Layer 0** may enrich the regime encoding available to Layer 1.

## General Observations

- The overall behavior of the Layered HMM is **broadly similar** to that of a standard HMM.
- **Layer 1 exhibits weaker correlations** with the original features, potentially indicating an **over-smoothing effect** or a reduced sensitivity to raw input dynamics.
- A **notable improvement was observed for AMZN**, suggesting the model may still offer **selective benefits**.
- **Volatility spike detection** remains robust across both layers.

## Final Thoughts

- **Layered HMM adds architectural depth**, but **does not lead to substantial performance gains** in volatility regime modeling.

In summary, while Layered HMM offers an elegant extension of the standard HMM, its utility in volatility modeling remains **situational** and **highly dependent on domain-specific tuning**.