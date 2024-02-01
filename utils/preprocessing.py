import pandas as pd


def process_data(data):
    """
    Process data into a numpy array and traspose it.

    Arguments:
    data -- dataframe or numpy array of shape (n_features, number of examples)

    Returns:
    data -- data in a suitable form for the model
    """
    if type(data) == pd.core.frame.Series:
        raise Exception('Must pass data in Numpy or Pandas DataFrame object')
    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy().T
    return data
