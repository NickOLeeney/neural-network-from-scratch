import numpy as np

from utils.backwardPropagation import L_model_backward, update_parameters
from utils.initialization import initialize_parameters_deep
from utils.forwardPropagation import L_model_forward
from utils.costFunctions import *


class NeuralNetworkFromScratch:
    def __init__(self, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        self.costs = None
        self.parameters = None
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost

    # GRADED FUNCTION: L_layer_model

    def fit(self, X, Y):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

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
        # (≈ 1 line of code)
        # parameters = ...
        # YOUR CODE STARTS HERE
        parameters = initialize_parameters_deep(self.layers_dims)
        # YOUR CODE ENDS HERE

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            # (≈ 1 line of code)
            # AL, caches = ...
            # YOUR CODE STARTS HERE
            AL, caches = L_model_forward(X, parameters)
            # YOUR CODE ENDS HERE

            # Compute cost.
            # (≈ 1 line of code)
            # cost = ...
            # YOUR CODE STARTS HERE
            cost = compute_cost(AL, Y)
            # YOUR CODE ENDS HERE

            # Backward propagation.
            # (≈ 1 line of code)
            # grads = ...
            # YOUR CODE STARTS HERE
            grads = L_model_backward(AL, Y, caches)
            # YOUR CODE ENDS HERE

            # Update parameters.
            # (≈ 1 line of code)
            # parameters = ...
            # YOUR CODE STARTS HERE
            parameters = update_parameters(parameters, grads, self.learning_rate)
            # YOUR CODE ENDS HERE

            # Print the cost every 100 iterations
            if self.print_cost and i % 100 == 0 or i == self.num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(cost)

        return parameters, costs

    def predict(self, X, y):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(self.parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = L_model_forward(X, self.parameters)

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
