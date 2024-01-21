import numpy as np


def cross_entropy_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (-1 / m) * (np.dot(Y, np.log(AL.T)) + np.dot((1 - Y), np.log(1 - AL.T)))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def cross_entropy_derivative(AL, Y):
    """
    Implement the Cross Entropy analytical derivative

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    derivative_cost -- cross-entropy derivative cost
    """

    derivative_cost = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    return derivative_cost
