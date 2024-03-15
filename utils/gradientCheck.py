import numpy as np
from utils.costFunctions import *
from utils.forwardPropagation import L_model_forward


def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    theta = None
    count = 0
    for key in list(parameters.keys()):

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dictionary(theta, actual_parameters):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = dict()
    length_prev = 0
    for key in actual_parameters.keys():
        shape_0 = actual_parameters[key].shape[0]
        shape_1 = actual_parameters[key].shape[1]
        length = length_prev + shape_0 * shape_1
        parameters[key] = theta[length_prev: length].reshape((shape_0, shape_1))
        length_prev = length

    # parameters["W1"] = theta[: 20].reshape((5, 4))
    # parameters["b1"] = theta[20: 25].reshape((5, 1))
    # parameters["W2"] = theta[25: 40].reshape((3, 5))
    # parameters["b2"] = theta[40: 43].reshape((3, 1))
    # parameters["W3"] = theta[43: 46].reshape((1, 3))
    # parameters["b3"] = theta[46: 47].reshape((1, 1))

    return parameters


def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    theta = None
    param_list = list()
    count = 0

    dW_list = [x for x in list(gradients.keys()) if ('W' in x)][::-1]
    db_list = [x for x in list(gradients.keys()) if ('b' in x)][::-1]

    for i in range(0, len(dW_list)):
        param_list.append(dW_list[i])
        param_list.append(db_list[i])

    for key in param_list:  # should be like ["dW1", "db1", "dW2", "db2", "dW3", "db3"]
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def gradient_check_n(parameters, gradients, X, Y, task, epsilon=1e-7, print_msg=True):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters
    X -- input datapoint, of shape (input size, number of examples)
    Y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    cost = None

    # Compute gradapprox
    for i in range(num_parameters):

        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        theta_plus = np.copy(parameters_values)
        theta_plus[i] = theta_plus[i] + epsilon

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, _, _ = L_model_forward(X, vector_to_dictionary(theta_plus, parameters), task, 1)

        # Compute cost.
        if task == 'binary_classification':
            cost = cross_entropy_cost(AL, Y)
        elif task == 'regression':
            cost = rmse_cost(AL, Y)
        elif task == 'multiple_classification':
            cost = cross_entropy_cost_softmax(AL, Y)

        J_plus[i] = cost

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        theta_minus = np.copy(parameters_values)
        theta_minus[i] = theta_minus[i] - epsilon
        AL, _, _ = L_model_forward(X, vector_to_dictionary(theta_minus, parameters), task, 1)

        # Compute cost.
        if task == 'binary_classification':
            cost = cross_entropy_cost(AL, Y)
        elif task == 'regression':
            cost = rmse_cost(AL, Y)
        elif task == 'multiple_classification':
            cost = cross_entropy_cost_softmax(AL, Y)

        J_minus[i] = cost

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2.5 * epsilon:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                difference) + "\033[0m")

    return difference
