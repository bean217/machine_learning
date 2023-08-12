# File: activation_functions.py
# Author: Benjamin Piro, benpiro1118@gmail.com
# Date: 12 August 2023
# Description: Provides pre-defined activation functions and their derivatives for
#   performing a forward pass of an artificial neural network, as well as training one.
#

import numpy as np

##### ACTIVATION FUNCTIONS

# SIGMOID

sigmoid = lambda x: 1/(1+np.exp(-x))

def d_sigmoid(x):
    act = sigmoid(x)
    return act * (1 - act)


# RELU

relu = lambda x: 0 if x < 0 else x

d_relu = lambda x: 0 if x < 0 else 1


# TANH

def tanh(x):
    ep = np.exp(x)
    en = np.exp(-x)
    return (ep-en)/(ep+en)

d_tanh = lambda x: 1 - tanh(x) ** 2


# SOFTMAX
# https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/

softmax = lambda x: np.exp(x) / sum(np.exp(x))

def d_softmax(x):
    act = softmax(x)
    return act * (1 - act)
