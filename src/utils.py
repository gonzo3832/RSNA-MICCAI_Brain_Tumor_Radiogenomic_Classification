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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

def get_debug_config(config):
    config['globals']['num_epochs'] = 3
    config['split']['params']['n_splits'] = 4
    config['globals']['folds'] =  [1,]
    return config


def get_debug_df(df):
    cases = df['MGMT_value'].unique()
    df_merge = pd.DataFrame(columns=df.columns)
    for case in cases:
        mask = df['MGMT_value'] == case
        df_merge = df_merge.append(df[mask].iloc[0:30, :])
    df_merge['MGMT_value'] = df_merge['MGMT_value'].astype(np.int64)
    df_merge = df_merge.reset_index(drop=True)
    return df_merge

def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.info('hello')

    set_seed()

if __name__ == '__main__':
    main()