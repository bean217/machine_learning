import numpy as np
from typing import Callable

from activation_functions import softmax, d_softmax
from activation_functions import sigmoid, d_sigmoid
from activation_functions import relu, d_relu
from activation_functions import tanh, d_tanh

from error_functions import mse, mse_l_esig, mse_h_esig
from error_functions import mce, mce_l_esig, mce_h_esig

class Train_Funcs:
    """Class which defines all of the necessary equations for implementing backpropagation. This
        class uses method definitions to ensure that the structure of function 
    """
    def __init__(self, activation, activation_derivative, error_func, last_error_signal, hidden_error_signal):
        # perform some checks on each function to make sure they have the right parameters and return the right data
        self.__act = activation
        self.__d_act = activation_derivative
        # error function for individual output layer neurons
        self.__err = error_func
        self.__L_esig = last_error_signal
        self.__H_esig = hidden_error_signal

    def act(self, x):
        return self.__act(x)
        
    def d_act(self, l_out, x):
        return self.__d_act(l_out, x)
        
    # error function for individual output layer neurons
    def err(self, y, y_pred):
        return self.__err(y, y_pred)
       
    def L_esig(self, l_in, y, y_pred):
        return self.__L_esig(l_in, y, y_pred, self.__d_act)
        
    def H_esig(self, l_in, esig, weights):
        return self.__H_esig(l_in, esig, weights, self.__d_act)


### TODO: CREATE AN **INTERFACE** SO THAT METHOD CALL STRUCTURE IS WELL-DEFINED


### Mean Square Error + Sigmoid
mean_square_error_sigmoid = Train_Funcs(
    activation = lambda x: sigmoid(x),
    activation_derivative = lambda x: d_sigmoid(x),
    error_func = lambda y, y_pred: mse(y, y_pred),
    last_error_signal = lambda l_in, y, y_pred, d_act: mse_l_esig(l_in, y, y_pred, d_act),
    hidden_error_signal = lambda l_in, esig, weights, d_act: mse_h_esig(l_in, esig, weights, d_act)
)


### Mean Square Error + ReLU
mean_square_error_relu = Train_Funcs(
    activation = lambda x: relu(x),
    activation_derivative = lambda x: d_relu(x),
    error_func = lambda y, y_pred: mse(y, y_pred),
    last_error_signal = lambda l_in, y, y_pred, d_act: mse_l_esig(l_in, y, y_pred, d_act),
    hidden_error_signal = lambda l_in, esig, weights, d_act: mse_h_esig(l_in, esig, weights, d_act)
)


### Mean Square Error + tanh
mean_square_error_tanh = Train_Funcs(
    activation = lambda x: tanh(x),
    activation_derivative = lambda x: d_tanh(x),
    error_func = lambda y, y_pred: mse(y, y_pred),
    last_error_signal = lambda l_in, y, y_pred, d_act: mse_l_esig(l_in, y, y_pred, d_act),
    hidden_error_signal = lambda l_in, esig, weights, d_act: mse_h_esig(l_in, esig, weights, d_act)
)


### Multiple Cross Entropy Loss + Softmax
multiple_cross_entropy_softmax = Train_Funcs(
    activation = lambda x: softmax(x),
    activation_derivative = lambda x: d_softmax(x),
    error_func = lambda y, y_pred: mce(y, y_pred),
    last_error_signal = lambda l_in, y, y_pred, d_act: mce_l_esig(y, y_pred),
    hidden_error_signal = lambda l_in, esig, weights, d_act: mce_h_esig(l_in, esig, weights, d_act)
)