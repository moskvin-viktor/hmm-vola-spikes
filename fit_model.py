import argparse

from omegaconf import OmegaConf

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch data, compute features, and fit HMM regime models."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CLASSES, "all"],
        default="LayeredHMMModel",
        help="Which model to train (default: LayeredHMMModel). 'all' trains every model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = list(MODEL_CLASSES) if args.model == "all" else [args.model]

    order = DataConfig.from_omegaconf(OmegaConf.load("config/data.yaml"))
    data = run_pipeline(order)

    for model_name in model_names:
        model = RegimeModelManager(
            data_dict=data,
            config_path="config/model.yaml",
            model_class=MODEL_CLASSES[model_name],
        )
        model.train_all()


if __name__ == "__main__":
    main()
