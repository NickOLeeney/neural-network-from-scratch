import numpy as np

from utils.backwardPropagation import L_model_backward, update_parameters
from utils.costFunctions import *
from utils.initialization import initialize_parameters_deep
from utils.forwardPropagation import L_model_forward
from utils.preprocessing import process_data


class NeuralNetworkFromScratch:
    def __init__(self, layers_dims, cost_function, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        self.costs = None
        self.parameters = None
        self.layers_dims = layers_dims
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost

    def fit(self, X, Y):
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

        # X = process_data(X)
        # Y = process_data(Y)

        np.random.seed(1)
        costs = []  # keep track of cost

        # Parameters initialization.
        parameters = initialize_parameters_deep(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X, parameters, self.cost_function)

            # Compute cost.
            if self.cost_function == 'cross_entropy':
                cost = cross_entropy_cost(AL, Y)
            elif self.cost_function == 'RMSE':
                cost = rmse_cost(AL, Y)
            else:
                raise Exception('Must specify a valid Cost Function')

            # Backward propagation.
            grads = L_model_backward(AL, Y, caches, self.cost_function)

            # Update parameters.
            parameters = update_parameters(parameters, grads, self.learning_rate)

            # Print the cost every 100 iterations
            if self.print_cost and i % 100 == 0 or i == self.num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost**0.5)))
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(cost**0.5)

        self.parameters = parameters
        self.costs = costs
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

        # X = process_data(X)
        # y = process_data(y)
        outcome = None

        if self.cost_function == 'cross_entropy':
            # Forward propagation
            inference, caches = L_model_forward(X, self.parameters, self.cost_function)
            # convert probas to 0/1 predictions
            m = X.shape[1]
            n = len(self.parameters) // 2  # number of layers in the neural network
            p = np.zeros((1, m))
            for i in range(0, inference.shape[1]):
                if inference[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0
            outcome = p
            print("Accuracy: " + str(np.sum((outcome == y) / m)))

        elif self.cost_function == 'RMSE':
            outcome = list()
            for x in X[0]:
                # Forward propagation
                inference, caches = L_model_forward(x, self.parameters, self.cost_function)
                outcome.append(inference[0][0])
            outcome = np.array(outcome)
            m = 1

            cost = rmse_cost(outcome, y)
            print("RMSE: " + str(cost**0.5))

        return outcome
