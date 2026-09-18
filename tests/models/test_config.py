import pytest
from pydantic import ValidationError

from hmmstock.models.config import (
    HierarchicalHMMConfig,
    HMMConfig,
    LayerConfig,
    LayeredHMMConfig,
)

LAYER = dict(
    min_components=2, max_components=4, covariance_type="full", init_params="stmc"
)


def test_hmm_config_from_real_yaml_shape():
    cfg = HMMConfig(
        covariance_type="full",
        random_seed=13,
        init_params="stmc",
        n_fits=100,
        tol=1e-4,
        max_components=2,
    )

    assert cfg.max_components == 2
    assert cfg.tol == pytest.approx(1e-4)


def test_hmm_config_is_frozen():
    cfg = HMMConfig(
        covariance_type="full",
        random_seed=13,
        init_params="stmc",
        n_fits=100,
        tol=1e-4,
        max_components=2,
    )

    with pytest.raises(ValidationError):
        cfg.max_components = 5


def test_layered_hmm_config_accepts_matching_num_layers():
    cfg = LayeredHMMConfig(
        num_layers=2, n_fits=100, random_seed=13, tol=1e-4, layers=[LAYER, LAYER]
    )

    assert len(cfg.layers) == 2
    assert isinstance(cfg.layers[0], LayerConfig)


def test_layered_hmm_config_rejects_num_layers_mismatch():
    with pytest.raises(ValidationError):
        LayeredHMMConfig(
            num_layers=3, n_fits=100, random_seed=13, tol=1e-4, layers=[LAYER, LAYER]
        )


def test_hierarchical_hmm_config_requires_both_layers():
    with pytest.raises(ValidationError):
        HierarchicalHMMConfig(
            n_fits=100, random_seed=13, tol=1e-4, top_layer=LAYER
        )  # missing sub_layer


def test_hierarchical_hmm_config_valid():
    cfg = HierarchicalHMMConfig(
        n_fits=100, random_seed=13, tol=1e-4, top_layer=LAYER, sub_layer=LAYER
    )

    assert cfg.top_layer.max_components == 4
    assert cfg.sub_layer.max_components == 4
