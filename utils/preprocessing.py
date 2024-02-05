import pandas as pd
import numpy as np


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


def target_encoder(target, n_classes):
    """
    Encode target variable for the softmax activation function in case of multiple classification.

    Arguements:
    target -- array-like containing the target variable

    Returns:
    output -- encoded data suitable for softmax function.
    """
    output = np.zeros(n_classes)
    for value in range(0, n_classes):
        if value == target:
            output[value] = 1
            return output
