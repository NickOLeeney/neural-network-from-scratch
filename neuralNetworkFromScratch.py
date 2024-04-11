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
from utils.gradientDescent import *
from utils.learningRate import *
from sklearn.preprocessing import OneHotEncoder


# warnings.filterwarnings("ignore")


class NeuralNetworkFromScratch:
    """
    Simple Python package to construct a neural network. \n

    Parameters: \n

    ----------
    - layers_dims (list) : Specify the number of layers and its relative nodes. \n
    - task (str): Specify the task (regression, binary classification or multiple classification). \n
    - learning_rate (float) : Learning rate for the gradient descent. \n
    - n_epochs (int) :  Number of epochs for training the neural network. \n
    - print_cost (bool): Specify wheter or not to print the cost function during training. \n
    - initialization (str): Specify the initialization for the parameters, possible choiches: "He" (default), "Xavier". \n
    - lambd (float) : regularization parameters. Set to None to not use it. \n
    - dropout (float) : dropout pararameter. It defines the percentage (from 0 to 1) of nodes to drop for every layer (default 1, menaning no drop out). \n
    ----------
    """

    def __init__(self, layers_dims, task, learning_rate=0.0075, n_epochs=3000, print_cost=False, initialization='He',
                 lambd=None, dropout=1):
        self._costs = None
        self._parameters = None
        self._layers_dims = layers_dims
        self._task = task
        self.learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._print_cost = print_cost
        self.init = initialization
        self.lambd = lambd
        self.keep_prob = dropout

    @property
    def layers_dims(self):
        return self._layers_dims

    @property
    def task(self):
        return self._task

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def print_cost(self):
        return self._print_cost

    def fit(self, X, Y, optimizer='adam', debug=None, print_every=None,
            mini_batch_size=None, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=(None, None)):
        """
        Implements an L-layer neural network. \n

        ---------- \n
        Arguments: \n
        - X (*pandas Dataframe, numpy array*): input data, of shape (n_features, number of training examples). \n
        - Y (*pandas Dataframe, numpy array*): true "label" vector. \n
        - optimizer (*str*) : optimizer used for gradient descent, options are: "gd", "momentum", "adam". \n
        - debug (*bool*) : wheter or not you want to do gradient check. \n
        - print_every (*int*) : frequency to plot the cost function. If set to None, it does not print anything. \n
        - mini_batch_size (*int*) : mini-batch size to perform gradient descent. Set to 1 for Stochastic Gradient Descent, for Classic Gradient Descent, n for Mini-batch Gradient Descent. Consider using power of 2 number to increase efficinency. \n
        - beta (*float*) : momentum parameter used for "momentum" optimizer. Optimal values range from 0.8 to 0.999, 0.9 is often a reasonable default. \n
        - beta1 (*float*) : first momentum parameter used for "adam" optimizer. Optimal values range from 0.8 to 0.999, 0.9 is often a reasonable default. \n
        - beta2 (*float*) : second momentum parameter used for "adam" optimizer. Optimal values range from 0.8 to 0.999, 0.999 is often a reasonable default. \n
        - epsilon (*float*) : parameters used to avoid dividing by zero numerical problems, consider not to modify it. \n
        - decay (*tuple*) : learning rate decay. Specifies decay rate (float) and frequency of updated, i.e. how many epochs to update learning rate. \n
        ---------- \n
        Returns: \n
        - parameters : parameters learnt by the model. They can then be used to predict.
        """

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
        L = len(self.layers_dims)  # number of layers in the neural networks
        t = 0  # initializing the counter required for Adam update
        seed = int(np.random.rand()*100)
        m = X.shape[1]  # number of training examples
        decay_rate, freq_update = decay

        learning_rate0 = self.learning_rate  # Setting leraning rate for scehduling
        v = None
        s = None

        if not mini_batch_size:
            mini_batch_size = m

        # Parameters initialization.
        parameters = initialize_parameters_deep(self._layers_dims, self.init)

        # Initialize the optimizer
        if optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif optimizer == "momentum":
            v = initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = initialize_adam(parameters)

        # Loop (gradient descent)
        for i in range(0, self._n_epochs):
            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
            cost_total = 0

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                AL, caches, dropout_cache = L_model_forward(minibatch_X, parameters, self._task, self.keep_prob)

                # Compute cost.
                if self._task == 'binary_classification':
                    cost = cross_entropy_cost(AL, minibatch_Y)
                elif self._task == 'regression':
                    cost = rmse_cost(AL, minibatch_Y)
                elif self._task == 'multiple_classification':
                    cost = cross_entropy_cost_softmax(AL, minibatch_Y)
                else:
                    raise Exception('Must specify a valid Cost Function')

                if math.isnan(cost.item()):
                    self._parameters = parameters
                    self._costs = costs

                    if print_every and self._n_epochs >= print_every:
                        plt.plot(np.arange(0, len(self._costs)) * print_every, self._costs)
                        plt.title(f'Cost Function vs Number of Epochs ({self.learning_rate})')
                        plt.xlabel('Epochs')
                        plt.ylabel('Cost Function')
                        plt.grid()

                    return parameters, costs

                if self.lambd:
                    m = Y.shape[1]
                    cost = l2_regularization(self.lambd, parameters, cost, m)

                cost_total += cost

                # Backward propagation.
                grads = L_model_backward(AL, minibatch_Y, caches, self._task, self.lambd, self.keep_prob, dropout_cache)

                # GRADIENT CHECK
                if debug:
                    if i % debug == 0:
                        _ = gradient_check_n(parameters, grads, minibatch_X, minibatch_Y, self._task, epsilon=1e-7,
                                             print_msg=True)

                # Update parameters
                if optimizer == "gd":
                    parameters = update_parameters_with_gd(parameters, grads, self.learning_rate)
                elif optimizer == "momentum":
                    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, self.learning_rate)
                elif optimizer == "adam":
                    t = t + 1  # Adam counter
                    parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                                         t, self.learning_rate, beta1, beta2, epsilon)

            if decay_rate:
                self.learning_rate = lr_decay(self.learning_rate, i, decay_rate, freq_update)

            cost_avg = cost_total / m

            # Print the cost every 100 iterations
            if self._print_cost and i % print_every == 0 or i == self._n_epochs - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost_avg)))
            if i % print_every == 0 or i == self._n_epochs:
                costs.append(cost_avg)

        self._parameters = parameters
        self._costs = costs

        if self._print_cost and self._n_epochs >= print_every:
            plt.plot(np.arange(0, len(self._costs)) * print_every, self._costs)
            plt.title(f'Cost Function vs Number of Epochs ({learning_rate0})')
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
