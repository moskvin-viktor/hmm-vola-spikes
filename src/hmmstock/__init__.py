import logging
import os

from .data.datamanager import DataManager
from .hhmm_model import HierarchicalHMMModel
from .hmm_model import HMMModel
from .lhmm_model import LayeredHMMModel
from .metrics import *
from .model import RegimeModelManager
from .plots import *
from .plots import HMMResultVisualization

_LOG_DIR = "results/logs"
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_LOG_DIR, "hmm_model.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
