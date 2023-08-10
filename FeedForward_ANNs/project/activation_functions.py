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

softmax = lambda l, x: np.exp(x) / sum(np.exp(l))

def d_softmax(l, x):
    act = softmax(l, x)
    return act * (1 - act)
