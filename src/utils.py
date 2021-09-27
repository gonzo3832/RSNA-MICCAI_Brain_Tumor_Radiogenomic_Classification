import os
import random

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import torch

import logging

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
    config['globals']['num_epochs'] = 1
    config['split']['params']['n_splits'] = 2
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

def ensemble(df, n_ignore_columns: int, name_ens_column: str) -> pd.DataFrame:
    """[summary]

    Args:
        df ([type]): [description]
        n_ignore_columns (int): [description]
        name_ens_column (str): [description]

    Returns:
        pd.DataFrame: [description]
    """    
    n_ensemble = len(df.columns)-n_ignore_columns
    df[name_ens_column] = df[name_ens_column].astype(np.float64)

    for index,row in df.iterrows():
        preds = row[n_ignore_columns:]
        vote = preds >= 0.5

        if vote.sum() == n_ensemble/2:
            if preds[preds >= 0.5].mean() > 1 - preds[preds < 0.5].mean():
                df.at[index, name_ens_column] = preds.max()
        
            else:
                df.at[index, name_ens_column] = preds.min()
        
        elif vote.sum() > n_ensemble/2:
            df.at[index, name_ens_column] = preds.max()
        
        else:
            df.at[index, name_ens_column] = preds.min()
    
    return df

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