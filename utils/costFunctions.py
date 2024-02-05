import numpy as np


def cross_entropy_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- predicted vector, shape (1, number of examples)
    Y -- true value vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of training examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def rmse_cost(AL, Y):
    """
    Implement the RMSE cost function.

    Arguments:
    AL -- predicted vector, shape (1, number of examples)
    Y -- true value vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of training examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / (2. * m)) * (np.sum((AL - Y) ** 2))**0.5
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def cross_entropy_cost_softmax(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- predicted vector, shape (1, number of examples)
    Y -- true value vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of training examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * ((-Y * np.log(AL)) - (1 - Y) * np.log(1 - AL)).sum(axis=1).sum()

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def cross_entropy_derivative(Y, AL):
    """
    Implement the Cross Entropy analytical derivative.

    Arguments:
    AL -- predicted vector, shape (1, number of examples)
    Y -- true value vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of training examples)

    Returns:
    derivative_cost -- cross-entropy derivative cost
    """

    derivative_cost = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    return derivative_cost


def rmse_derivative(Y, AL):
    """
    Implement the RMSE analytical derivative.

    Arguments:
    AL -- predicted vector, shape (1, number of examples)
    Y -- true value vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of training examples)

    Returns:
    derivative_cost -- RMSE derivative cost
    """

    derivative_cost = (AL - Y)
    return derivative_cost
