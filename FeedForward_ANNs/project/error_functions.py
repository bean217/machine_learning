import numpy as np

##### ERROR FUNCTIONS

### MEAN SQUARED ERROR

mse = lambda y, y_pred: ((y - y_pred) ** 2)/2

# MSE Last Layer Error Signal

mse_l_esig = lambda l_in, y, y_pred, d_act: d_act(l_in) * (y - y_pred)

# MSE Hidden Layer Error Signal

mse_h_esig = lambda l_in, esig, weights, d_act: \
    d_act(l_in) * np.asarray([sum(weights[:,i] * esig) for i in range(l_in.shape[0])])


### BINARY CROSS ENTROPY

bce = lambda y, y_pred: -y*np.log(y_pred) - (1-y)*np.log(y_pred)


### MULTIPLE CROSS ENTROPY

mce = lambda y, y_pred: -sum(y*np.log(y_pred))

# MCE Last Layer Error Signal

mce_l_esig = lambda y, y_pred: (y_pred - y)

# MSE Hidden Layer Error Signal

mce_h_esig = lambda l_in, esig, weights, d_act: \
    d_act(l_in) * np.asarray([sum(weights[:,i] * esig) for i in range(l_in.shape[0])])