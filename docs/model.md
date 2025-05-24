# Model 

## At a glance Hidden Markov Model

A **Hidden Markov Model (HMM)** is a statistical model that assumes the system being modeled is a Markov process with unobserved (hidden) states. It is particularly well-suited for time-series data where the true state of the system is not directly observable but can be inferred through observed signals.

Key concepts:

- **States**: Hidden regimes that evolve over time according to a Markov process.

- **Observations**: The data we observe, which is probabilistically emitted from the hidden states.

- **Transition Probabilities**: The likelihood of moving from one state to another.

- **Emission Probabilities**: The likelihood of observing a given output from a particular state.

HMMs are commonly used in:

- Financial market regime detection

- Speech recognition

- Bioinformatics (e.g., gene prediction)

- Anomaly detection in time-series data

This implementation leverages the `hmmlearn` library to train a Gaussian HMM on financial volatility features. Layered variants extend the base HMM by stacking multiple models, where each layer learns patterns from the probabilistic output of the previous layer, enabling more expressive and hierarchical regime detection.

## Hidden Markov Model 

The `HMMModel` is a probabilistic time series model built using a **Gaussian Hidden Markov Model (HMM)**. It is suitable for modeling **regime-switching behavior**, such as different volatility regimes in financial markets. This implementation leverages the `hmmlearn` library and is designed for use on multivariate time series data.

A **Hidden Markov Model** consists of:

- **Hidden States**: Unobservable regimes (e.g., low, medium, high volatility).

- **Observations**: Observable data (e.g., volatility of returns).

- **Transition Probabilities**: Probabilities of switching between states.

- **Emission Probabilities**: Distributions (e.g., Gaussians) describing how observations are generated from hidden states.

```yaml
HMMModel:
  covariance_type: "full"
  random_seed: 13
  init_params: "stmc"
  n_fits: 100
  tol: 1e-4
  max_components: 3
```

### Hyperparameters

The model supports the following hyperparameters through the configuration object:

| Hyperparameter     | Type    | Description                                                                 |
|--------------------|---------|-----------------------------------------------------------------------------|
| `covariance_type`  | `str`   | Specifies the form of the covariance matrix.                                |
| `random_seed`      | `int`   | Seed for reproducibility. Ensures consistent random number generation.      |
| `init_params`      | `str`   | Specifies which model parameters to initialize:                             |
| `n_fits`           | `int`   | Number of training attempts per configuration. Used for robustness.         |
| `tol`              | `float` | Convergence threshold for log-likelihood change between iterations.         |
| `max_components`   | `int`   | Maximum number of hidden states (components) to try. Minimum is 2.          |


### Iterative Fitting

The model is trained using the Expectation-Maximization (EM) algorithm:

For each `n_components` in `[2, ..., max_components]`:

  - Train `n_fits` different models with different random seeds.

  - Fit using `GaussianHMM.fit(X_train)` from `hmmlearn`.

  - Evaluate each using the provided `evaluation_metric`.

### Output

The model can output:
- **Trained HMM model object**
- **Predicted states** (relabeled by ascending volatility)
- **Log files** detailing errors, warnings, and training status

### Code

::: hmmstock.HMMModel

## Layered Hidden Markov Model 

### Overview

`LayeredHMMModel` is an extension of traditional Hidden Markov Models (HMMs) that allows for **layered (hierarchical) modeling of temporal regimes**. Each layer is trained sequentially:

- **Layer 1** is trained on raw input features.

- **Layer 2** is trained on the **posterior probabilities** (state likelihoods) from Layer 1.

- ...

- **Layer N** is trained on outputs from Layer N-1.

This design enables the model to **capture increasingly abstract temporal structures** in the data across multiple layers of representation.

### Use Cases

- Capturing hierarchical or multi-scale regimes in financial time series

- Improving regime separation through feature transformation between layers

- Modeling complex temporal dependencies and latent dynamics


### Hyperparameters 

| Parameter        | Type     | Description                                                              |
|------------------|----------|--------------------------------------------------------------------------|
| `min_components` | `int`    | Minimum number of hidden states to try                                   |
| `max_components` | `int`    | Maximum number of hidden states to try                                   |
| `covariance_type`| `str`    | Form of covariance matrix (`'full'`, `'diag'`, etc.)                     |
| `init_params`    | `str`    | Parameters to initialize during training (`'stmc'`)                      |

```yaml
LayeredHMMModel:
  num_layers: 2
  n_fits: 100
  random_seed: 13
  tol: 1e-4
  layers:
    - min_components: 2
      max_components: 3
      covariance_type: "full"
      init_params: "stmc"
    - min_components: 2
      max_components: 2
      covariance_type: "full"
      init_params: "stmc"
```

General parameters inherited from the global config:

| Parameter     | Description                                           |
|---------------|-------------------------------------------------------|
| `num_layers`  | Number of HMM layers to stack                         |
| `n_fits`      | Number of random initializations per configuration    |
| `tol`         | Convergence threshold for EM algorithm                |
| `random_seed` | Ensures reproducibility                               |


### Advantages of Layered HMMs

- **Hierarchical abstraction**: Later layers learn abstract temporal structures
- **Noise reduction**: Intermediate posterior probabilities can be smoother than raw features
- **Better separation**: Progressive modeling can separate regimes more clearly



### Limitations

- **Computational cost**: Training multiple HMMs in sequence is resource-intensive
- **Overfitting risk**: Especially with many layers or high `max_components`
- **Data requirement**: Each layer needs sufficient data to train reliably


### Output

- A list of trained HMM models, one per layer
- Layered regime predictions (`predict_states`)
- Log messages to track training process, warnings, and failures


::: hmmstock.LayeredHMMModel

## Hierarchical Hidden Markov Model


### Overview 


The **Hierarchical HMM (H-HMM)** incorporates a two-level generative process:

- A **top-level HMM** governs transitions between high-level regimes (e.g., market regimes, phonemes).
- For each top-level state, a distinct **sub-HMM** is responsible for modeling the local dynamics of the observations within that regime.

1. At time \( t \), a regime \( Z_t \in \{1, \ldots, K\} \) is selected from the top-level HMM.
2. Given \( Z_t = k \), a sub-HMM \( HMM_k \) is used to emit \( X_t \) via its own internal hidden states.

This leads to a nested generative structure:
$$
P(X_{1:T}, Z_{1:T}, S_{1:T}) = \prod_{t=1}^T P(Z_t | Z_{t-1}) P(S_t^Z | S_{t-1}^Z) P(X_t | S_t^Z)
$$

Where:
- \( Z_t \): top-level regime
- \( S_t^Z \): sub-HMM hidden state at time \( t \) given \( Z_t \)


### Advantages

- Models regime-specific dynamics.
- Enables contextual transitions and multi-modal observation modeling.
- Especially useful when different segments of the data exhibit qualitatively different behaviors.


## References

- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE.
- `hmmlearn` documentation: [https://hmmlearn.readthedocs.io/](https://hmmlearn.readthedocs.io/)

An introduction to the use of hidden Markov
models for stock return analysis
Chun Yu Hong∗
, Yannik Pitcan†

Hidden Markov Models Fundamentals
Daniel Ramage
CS229 Section Notes
December 1, 2007