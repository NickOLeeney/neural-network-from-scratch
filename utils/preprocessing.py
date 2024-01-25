import pandas as pd


def process_data(data):
    if type(data) == pd.core.frame.Series:
        raise Exception('Must pass data in Numpy or Pandas DataFrame object')
    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy().T
    return data
