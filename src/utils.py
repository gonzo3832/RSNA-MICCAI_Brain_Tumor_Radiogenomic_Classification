import os
import torch
import random
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def get_debug_config(config):
    config['globals']['num_epochs'] = 3
    config['split']['params']['n_splits'] = 2
    return config