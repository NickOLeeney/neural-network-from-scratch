import warnings

import numpy as np
import matplotlib.pyplot as plt

import warnings

from utils.backwardPropagation import L_model_backward, update_parameters
from utils.costFunctions import *
from utils.initialization import initialize_parameters_deep
from utils.forwardPropagation import L_model_forward
from utils.preprocessing import process_data, target_encoder
from utils.gradientCheck import gradient_check_n
from sklearn.preprocessing import OneHotEncoder


# warnings.filterwarnings("ignore")


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

    def __init__(self, layers_dims, task, learning_rate=0.0075, n_epochs=3000, print_cost=False, initialization='He',
                 lambd=None, keep_prob=1):
        self._costs = None
        self._parameters = None
        self._layers_dims = layers_dims
        self._task = task
        self._learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._print_cost = print_cost
        self.init = initialization
        self.lambd = lambd
        self.keep_prob = keep_prob

    @property
    def layers_dims(self):
        return self._layers_dims

    @property
    def task(self):
        return self._task

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def print_cost(self):
        return self._print_cost

    def fit(self, X, Y, plot_cost_function=False, debug=None, print_every=100):
        """
        Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_features, number of training examples).
        Y -- true "label" vector.
        plot_cost_function -- if True, it plots the cost function during the training.

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        # np.random.seed(1)

        if self._task == 'multiple_classification':
            n_classes = len(np.unique(Y))
            enc = OneHotEncoder(sparse_output=False, categories='auto')
            X = process_data(X)
            Y = enc.fit_transform(Y.to_numpy().reshape(len(Y), -1)).T
            Y = np.concatenate(Y, axis=0).reshape(n_classes, X.shape[1])

        else:
            X = process_data(X)
            Y = process_data(Y)

        costs = list()  # keep track of cost

        # Parameters initialization.
        parameters = initialize_parameters_deep(self._layers_dims, self.init)

        # Loop (gradient descent)
        for i in range(0, self._n_epochs):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches, dropout_cache = L_model_forward(X, parameters, self._task, self.keep_prob)

            # Compute cost.
            if self._task == 'binary_classification':
                cost = cross_entropy_cost(AL, Y)
            elif self._task == 'regression':
                cost = rmse_cost(AL, Y)
            elif self._task == 'multiple_classification':
                cost = cross_entropy_cost_softmax(AL, Y)
            else:
                raise Exception('Must specify a valid Cost Function')

            if math.isnan(cost.item()):
                self._parameters = parameters
                self._costs = costs

                if plot_cost_function and self._n_epochs >= print_every:
                    plt.plot(np.arange(0, len(self._costs)) * print_every, self._costs)
                    plt.title(f'Cost Function vs Number of Epochs ({self._learning_rate})')
                    plt.xlabel('Epochs')
                    plt.ylabel('Cost Function')
                    plt.grid()

                return parameters, costs

            if self.lambd:
                m = Y.shape[1]
                cost = l2_regularization(self.lambd, parameters, cost, m)

            # Backward propagation.
            grads = L_model_backward(AL, Y, caches, self._task, self.lambd, self.keep_prob, dropout_cache)

            # GRADIENT CHECK
            if debug:
                if i % debug == 0:
                    _ = gradient_check_n(parameters, grads, X, Y, self._task, epsilon=1e-7, print_msg=True)

            # Update parameters.
            parameters = update_parameters(parameters, grads, self._learning_rate)

            # Print the cost every 100 iterations
            if self._print_cost and i % print_every == 0 or i == self._n_epochs - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % print_every == 0 or i == self._n_epochs:
                costs.append(cost)

        self._parameters = parameters
        self._costs = costs

        if plot_cost_function and self._n_epochs >= print_every:
            plt.plot(np.arange(0, len(self._costs)) * print_every, self._costs)
            plt.title(f'Cost Function vs Number of Epochs ({self._learning_rate})')
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

        if self._task == 'binary_classification':
            # Forward propagation
            inference, caches, _ = L_model_forward(X, self._parameters, self._task, self.keep_prob)
            # convert probas to 0/1 predictions
            m = X.shape[1]
            n = len(self._parameters) // 2  # number of layers in the neural network
            p = np.zeros((1, m))
            for i in range(0, inference.shape[1]):
                if inference[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0
            outcome = p
            print("Accuracy: " + str(np.sum((outcome == y) / m)))

        if self._task == 'multiple_classification':
            output = list()
            m = X.shape[1]

            # Forward propagation
            inference, caches, _ = L_model_forward(X, self._parameters, self._task, self.keep_prob)

            # convert probas to 0/1 predictions
            for col in range(0, m):
                out_class = np.where(inference[:, col] == max(inference[:, col]))[0]
                output.append(out_class)
            outcome = np.array(output).flatten()
            outcome = outcome.reshape(1, outcome.shape[0])

            print("Accuracy: " + str(np.sum((outcome == y) / m)))

        elif self._task == 'regression':
            outcome = list()
            # for i in range(0, X.shape[1]):
            #     x = X[:, i]
            #     x = x.reshape(-1, 1)
            #     # Forward propagation
            #     inference, caches = L_model_forward(x, self._parameters, self._task)
            #     outcome.append(inference[0][0])

            inference, caches, _ = L_model_forward(X, self._parameters, self._task, self.keep_prob)
            outcome = inference[0].reshape(1, -1)

            cost = rmse_cost(outcome, y)
            print("RMSE: " + str(cost))

        return outcome
