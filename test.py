# test code の避難場所

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

def test_ensemble_from_df() -> pd.DataFrame:
    column_names = ['id', 'predict',
                     'A', 'B', 'C', 'D']
    df = pd.DataFrame(columns=column_names)
    df['id'] = np.arange(10)
    df['predict'] = 0
    for cap in column_names[2:]:
        df[cap] = np.random.rand(10)
    print(df.head())
    print(type(df))
    n_ensemble = len(df.columns)-2

    df['predict'] = df['predict'].astype(np.float64)
    for index,row in df.iterrows():
        preds = row[2:]
        vote = preds >= 0.5
        if vote.sum() == n_ensemble/2:
            if preds[preds >= 0.5].mean() > 1 - preds[preds<0.5].mean():
                df.at[index, 'predict'] = preds.max()
            else:
                df.at[index, 'predict'] = preds.min()
        elif vote.sum() > n_ensemble/2:
            df.at[index, 'predict'] = preds.max()
        else:
            df.at[index, 'predict'] = preds.min()
    print(df.head(10))


if __name__ == '__main__':
    test_ensemble_from_df()