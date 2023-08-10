import numpy as np

class Train_Funcs:
    def __init__(self, activation: function, activation_derivative: function, 
                 error_func: function, last_error_signal: function, hidden_error_signal: function):
        # perform some checks on each function to make sure they have the right parameters and return the right data
        self.act = activation
        self.d_act = activation_derivative
        # error function for individual output layer neurons
        self.err = error_func
        self.L_esig = last_error_signal
        self.H_esig = hidden_error_signal


### TODO: CREATE AN **INTERFACE** SO THAT METHOD CALL STRUCTURE IS WELL-DEFINED