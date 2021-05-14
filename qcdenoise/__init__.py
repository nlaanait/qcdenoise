from .circuit_samplers import *
# from .circuit_constructors import GHZCircuit, GraphCircuit
from .circuit_sampling_utils import *
from .io_utils import *
from .ml_models import *
from .graph_data import *
from .graph_state import *
from .graph_circuit import *
from .config import *

import logging

__version__ = "0.0.1"

import logging
from pytorch_lightning import seed_everything

# set global seed
__seed__ = 1234
seed_everything(__seed__)

# flake8: noqa: F403, F401

# set logger
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# set plotting flag
# plotting options
try:
    import matplotlib.pyplot as plt
    __plots__ = True
except ImportError:
    logger.warning(
        "matplotlib could not be imported- skipping plotting.")
    __plots__ = False
