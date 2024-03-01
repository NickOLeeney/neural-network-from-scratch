from utils.activationFunctions import *


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation, keep_prob):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    linear_cache = None
    activation_cache = None
    A = None

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == 'linear':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = linear(Z)

    elif activation == 'softmax':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    # Dropout
    dropout = np.random.rand(A.shape[0], A.shape[1])
    dropout = (dropout <= keep_prob).astype(int)
    A = A * dropout
    A = A / keep_prob

    cache = (linear_cache, activation_cache)

    return A, cache, dropout


def L_model_forward(X, parameters, task, keep_prob):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->FINAL ACTIVATION FUNCTION computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    dropout_cache = [None]
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A
        A, cache, dropout = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                                      'relu',
                                                      keep_prob)
        caches.append(cache)
        dropout_cache.append(dropout)

    if task == 'binary_classification':
        AL, cache, dropout = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid',
                                                       keep_prob=1)
    elif task == 'regression':
        AL, cache, dropout = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'linear',
                                                       keep_prob=1)
    elif task == 'multiple_classification':
        AL, cache, dropout = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'softmax',
                                                       keep_prob=1)
    else:
        raise Exception('Must specify a valid task')

    caches.append(cache)
    dropout_cache.append(dropout)

    return AL, caches, dropout_cache
