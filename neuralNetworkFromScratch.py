import numpy as np
import matplotlib.pyplot as plt

from utils.backwardPropagation import L_model_backward, update_parameters
from utils.costFunctions import *
from utils.initialization import initialize_parameters_deep
from utils.forwardPropagation import L_model_forward
from utils.preprocessing import process_data, target_encoder


class NeuralNetworkFromScratch:
    """
    Simple Python package to construct a neural network.

    Parameters
    ----------
        layers_dims : list
                   Specify the number of layers and its relative nodes.
        task : str
            Specify the task (regression, binary classification or multiple classification).
        learning_rate : float (default = 0.0075)
                     Learning rate for the gradient descent.
        n_epochs : int (default = 3000)
                     Number of epochs for training the neural network
        print_cost : bool (default = False)
                  Specify wheter or not to print the cost function during training
    """

    def __init__(self, layers_dims, task, learning_rate=0.0075, n_epochs=3000, print_cost=False):
        self.costs = None
        self.parameters = None
        self.layers_dims = layers_dims
        self.task = task
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.print_cost = print_cost

    def fit(self, X, Y, plot_cost_function=False, print_every=100):
        """
        Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_features, number of training examples).
        Y -- true "label" vector.
        plot_cost_function -- if True, it plots the cost function during the training.

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)

        X = process_data(X)
        Y = process_data(Y)

        if self.task == 'multiple_classification':
            n_classes = len(np.unique(Y))
            Y = np.array(list((map(target_encoder, Y[0], [n_classes for x in Y[0]]))))
            Y = np.concatenate(Y, axis=0).reshape(n_classes, X.shape[1])

        costs = list()  # keep track of cost

        # Parameters initialization.
        parameters = initialize_parameters_deep(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, self.n_epochs):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X, parameters, self.task)

            # Compute cost.
            if self.task == 'binary_classification':
                cost = cross_entropy_cost(AL, Y)
            elif self.task == 'regression':
                cost = rmse_cost(AL, Y)
            elif self.task == 'multiple_classification':
                cost = cross_entropy_cost_softmax(AL, Y)
            else:
                raise Exception('Must specify a valid Cost Function')

            # Backward propagation.
            grads = L_model_backward(AL, Y, caches, self.task)

            # Update parameters.
            parameters = update_parameters(parameters, grads, self.learning_rate)

            # Print the cost every 100 iterations
            if self.print_cost and i % print_every == 0 or i == self.n_epochs - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost ** 0.5)))
            if i % print_every == 0 or i == self.n_epochs:
                costs.append(cost ** 0.5)

        self.parameters = parameters
        self.costs = costs

        if plot_cost_function:
            plt.plot(np.arange(0, len(self.costs)) * print_every, self.costs)
            plt.title(f'Cost Function vs Number of Epochs ({self.learning_rate})')
            plt.xlabel('Epochs')
            plt.ylabel('Cost Function')
            plt.grid()

        return parameters, costs

    def predict(self, X, y):
        """
        This function is used to predict the results of an L-layer neural network.

        Arguments:
        X -- data set of examples you would like to predict.

        Returns:
        outcome -- predictions for the given dataset X.
        """

        X = process_data(X)
        y = process_data(y)
        outcome = None

        if self.task == 'binary_classification':
            # Forward propagation
            inference, caches = L_model_forward(X, self.parameters, self.task)
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

        elif self.task == 'regression':
            outcome = list()
            for x in X[0]:
                # Forward propagation
                inference, caches = L_model_forward(x, self.parameters, self.task)
                outcome.append(inference[0][0])
            outcome = np.array(outcome)
            # m = 1

            cost = rmse_cost(outcome, y)
            print("RMSE: " + str(cost ** 0.5))

        return outcome
