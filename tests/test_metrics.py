import numpy as np
import pytest
from hmmlearn import hmm
from omegaconf import OmegaConf

from hmmstock.metrics import (
    BICMetric,
    LogLikelihoodWithEntropy,
    build_evaluation_metric,
)


def _fitted_model(n_components=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 2))
    model = hmm.GaussianHMM(n_components=n_components, random_state=seed, n_iter=20)
    model.fit(X)
    return model, X


def test_bic_metric_is_negated_so_higher_is_better():
    model, X = _fitted_model()

    metric = BICMetric()
    score = metric.evaluate(model, X, X)

    # hmmlearn's model.bic() is lower-is-better (standard BIC convention);
    # the metric must negate it, since every selection loop in this
    # codebase does `if score > best_score`.
    assert score == pytest.approx(-model.bic(X))


def test_bic_metric_scores_on_train_not_validate():
    model, X_train = _fitted_model()
    rng = np.random.default_rng(999)
    X_validate = rng.normal(size=(50, 2))

    metric = BICMetric()
    score = metric.evaluate(model, X_train, X_validate)

    # Must use X_train for BIC (that's the entire point of BIC: penalize
    # training likelihood by parameter count, no held-out set needed),
    # not silently score on X_validate instead.
    assert score == pytest.approx(-model.bic(X_train))
    assert score != pytest.approx(-model.bic(X_validate))


def test_log_likelihood_with_entropy_scores_on_validate_not_train():
    model, X_train = _fitted_model()
    rng = np.random.default_rng(999)
    X_validate = rng.normal(size=(50, 2))

    metric = LogLikelihoodWithEntropy(entropy_weight=0)
    score = metric.evaluate(model, X_train, X_validate)

    expected = model.score(X_validate) / len(X_validate)
    assert score == pytest.approx(expected)


def test_log_likelihood_with_entropy_weight_scales_entropy_term():
    model, X = _fitted_model()

    low = LogLikelihoodWithEntropy(entropy_weight=0).evaluate(model, X, X)
    high = LogLikelihoodWithEntropy(entropy_weight=10).evaluate(model, X, X)

    # entropy term is >= 0 (Shannon entropy), so a bigger weight can only
    # push the score up, never down, for the same model/data.
    assert high >= low


def test_build_evaluation_metric_defaults_to_log_likelihood():
    cfg = OmegaConf.create({})

    metric = build_evaluation_metric(cfg)

    assert isinstance(metric, LogLikelihoodWithEntropy)
    assert metric.entropy_weight == 3


def test_build_evaluation_metric_reads_entropy_weight():
    cfg = OmegaConf.create(
        {"evaluation_metric": "LogLikelihoodWithEntropy", "entropy_weight": 7}
    )

    metric = build_evaluation_metric(cfg)

    assert metric.entropy_weight == 7


def test_build_evaluation_metric_selects_bic():
    cfg = OmegaConf.create({"evaluation_metric": "BICMetric"})

    metric = build_evaluation_metric(cfg)

    assert isinstance(metric, BICMetric)


def test_build_evaluation_metric_rejects_unknown_name():
    cfg = OmegaConf.create({"evaluation_metric": "NotARealMetric"})

    with pytest.raises(ValueError, match="NotARealMetric"):
        build_evaluation_metric(cfg)
