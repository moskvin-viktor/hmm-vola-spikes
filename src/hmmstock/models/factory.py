from enum import Enum
import numpy as np

from .hmm_model import HMMModel
from .lhmm_model import LayeredHMMModel
from .hhmm_model import HierarchicalHMMModel
from .markov_model import MarkovModel
from ..metrics import EvaluationMetric

class MarkovModelEnum(Enum):
    HMM = "HMMModel"
    LAYERED_HMM = "LayeredHMMModel"
    HIERARCHICAL_HMM = "HierarchicalHMMModel"

class MarkovModelFactory:
    @staticmethod
    def create_model(
        model_type: MarkovModelEnum,
        X: np.ndarray,
        config: dict,
        evaluation_metric: EvaluationMetric,
        name: str = "",
    ) -> MarkovModel:
        if model_type == MarkovModelEnum.HMM:
            return HMMModel(X=X, config=config, evaluation_metric=evaluation_metric)
        elif model_type == MarkovModelEnum.LAYERED_HMM:
            return LayeredHMMModel(name=name, X=X, config=config, evaluation_metric=evaluation_metric)
        elif model_type == MarkovModelEnum.HIERARCHICAL_HMM:
            return HierarchicalHMMModel(name=name, X=X, config=config, evaluation_metric=evaluation_metric)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
