import numpy as np
import math


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
    cost = (1. / m) * (-np.dot(Y, np.log(AL+1e-9).T) - np.dot(1 - Y, np.log(1 - AL+1e-9).T))
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
    cost = ((1. / (2. * m)) * np.sum((AL - Y) ** 2))
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
    cost = -np.mean(Y * np.log(AL + 1e-8))
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

    derivative_cost = - (np.divide(Y, AL + 1e-9) - np.divide(1 - Y, 1 - AL + 1e-9))

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


def l2_regularization(lambd, parameters, cost, m):
    L2_regularization_cost = 0
    for layer in range(1, ((len(parameters) // 2) + 1)):
        Wl = parameters['W' + str(layer)]
        L2_regularization_cost_item = np.sum(np.square(Wl))
        L2_regularization_cost += L2_regularization_cost_item

    L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost
    cost = cost + L2_regularization_cost
    return cost
