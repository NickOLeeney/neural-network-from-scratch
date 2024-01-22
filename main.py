import numpy as np

from utils.backwardPropagation import L_model_backward, update_parameters
from utils.costFunctions import cross_entropy_cost
from utils.initialization import initialize_parameters_deep
from utils.forwardPropagation import L_model_forward


# def main():
#     X = None
#     Y = None
#     layers_dims = None
#     cost_function = 'cross_entropy'
#     learning_rate = 0.0075
#     num_iterations = 3000
#     print_cost = False
#
#     parameters, costs = deep_neural_network(X, Y, layers_dims, cost_function, learning_rate, num_iterations, print_cost)


def deep_neural_network(X, Y, layers_dims, cost_function, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        if cost_function == 'cross_entropy':
            cost = cross_entropy_cost(AL, Y)
        else:
            raise Exception('Must specify a valid Cost Function')

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, cost_function)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p

# if __name__ == '__main__':
#     main()
