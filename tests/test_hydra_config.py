from pathlib import Path

import hydra
from omegaconf import OmegaConf

from hmmstock.data.config import DataConfig
from hmmstock.models import HierarchicalHMMModel, HMMModel, LayeredHMMModel

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def _compose(overrides=None):
    with hydra.initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return hydra.compose(config_name="config", overrides=overrides or [])


def test_composes_data_model_and_model_class():
    cfg = _compose()

    assert cfg.model_class == "LayeredHMMModel"
    assert "tickers" in cfg.data
    assert {"HMMModel", "LayeredHMMModel", "HierarchicalHMMModel", "split"} <= set(
        cfg.model.keys()
    )


def test_data_section_validates_as_dataconfig():
    cfg = _compose()

    order = DataConfig.from_omegaconf(cfg.data)

    assert "^VIX" in order.all_tickers


def test_model_section_validates_for_every_model_class():
    cfg = _compose()

    for model_class in [HMMModel, LayeredHMMModel, HierarchicalHMMModel]:
        raw = OmegaConf.to_container(cfg.model[model_class.__name__], resolve=True)
        # RegimeModelManager._build_model_config does the equivalent of
        # this validation; assert it doesn't raise for the real config.
        model_class.config_cls.model_validate(raw)


def test_model_class_override_is_applied():
    cfg = _compose(overrides=["model_class=HMMModel"])

    assert cfg.model_class == "HMMModel"


def test_nested_model_override_is_applied():
    cfg = _compose(overrides=["model.HMMModel.max_components=7"])

    assert cfg.model.HMMModel.max_components == 7
