import hydra
from omegaconf import DictConfig

from hmmstock import (
    DataConfig,
    HierarchicalHMMModel,
    HMMModel,
    LayeredHMMModel,
    RegimeModelManager,
    run_pipeline,
)

MODEL_CLASSES = {
    "HMMModel": HMMModel,
    "LayeredHMMModel": LayeredHMMModel,
    "HierarchicalHMMModel": HierarchicalHMMModel,
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Fetch data, compute features, and fit HMM regime models.

    Which model(s) to train is set by cfg.model_class (config/config.yaml),
    overridable on the CLI: `hatch run python fit_model.py model_class=HMMModel`.
    Use `model_class=all` to train every model.
    """
    if cfg.model_class not in {*MODEL_CLASSES, "all"}:
        raise ValueError(
            f"Unknown model_class {cfg.model_class!r}; expected one of "
            f"{[*MODEL_CLASSES, 'all']}"
        )
    model_names = list(MODEL_CLASSES) if cfg.model_class == "all" else [cfg.model_class]

    order = DataConfig.from_omegaconf(cfg.data)
    data = run_pipeline(order)

    for model_name in model_names:
        manager = RegimeModelManager(
            data_dict=data,
            cfg=cfg.model,
            model_class=MODEL_CLASSES[model_name],
        )
        manager.train_all()


if __name__ == "__main__":
    main()
