import logging
import os

from .data.config import DataConfig
from .data.pipeline import run_pipeline
from .manager import RegimeModelManager
from .metrics import *
from .models import HierarchicalHMMModel, HMMModel, LayeredHMMModel, RegimeModel

_LOG_DIR = "results/logs"
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_LOG_DIR, "hmm_model.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
