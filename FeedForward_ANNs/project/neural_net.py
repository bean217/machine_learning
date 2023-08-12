import numpy as np
from matplotlib import pyplot as plt

from training_functions import Train_Funcs


def plot_errs(train_errs: np.ndarray, validation_errs: np.ndarray):
    """Plots the average error of the network, as determined by the desired error function
        @param train_errs: matrix of (epoch, average training error)
        @param validation_errs: matrix of (epoch, average validation error)
    """
    plt.plot(train_errs[:,0], train_errs[:,1])
    plt.plot(validation_errs[:,0], validation_errs[:,1])
    plt.xlabel("Epoch (e)")
    plt.ylabel("Average Net Network Error")
    plt.show()


class ANN:
    def __init__(self, training_functions: Train_Funcs, architecture: tuple):
        """Creates a feed forward neural network model based on an architecture
            @param architecture: list of integers containing number of neurons in each layer
        """
        if len(architecture) < 2:
            raise Exception("Must have at least 2 network layers")

        # create network layers
        self.layers = []
        for i in range(len(architecture)-1):
            self.layers.append(Layer((architecture[i], architecture[i+1])))

        # training functions
        self.tfs = training_functions


    def forward(self, x: np.ndarray):
        """Performs a forward pass through the network
            @param x: data instance serving as the network input
            @returns: an array of the network's input and outputs through each layer
        """

        if self.tfs == None:
            raise Exception("Activation function not set")
        # keep track of inputs/outputs as they propagate through layers
        outputs = [x]
        inputs = []
        curr = x
        for l in self.layers:
            # pad with 1 for bias
            curr = np.concatenate(([1], curr))
            # get input and output of layer
            input_sum = np.asarray([curr @ l.weights[i,:] for i in range(l.dim_out)])
            curr = self.tfs.act(input_sum)
            # append to inputs and outputs
            inputs.append(input_sum)
            outputs += [curr]
        return inputs, outputs
    

    def activate(self, x: np.ndarray):
        """Performs a forward pass through the network
            @param x: data instance serving as the network input
            @returns: the networks output
        """
        if self.tfs == None:
            raise Exception("Activation function not set")
        # return only the last output
        return self.forward(x)[1][-1]
    

    def backprop(self, batch_X, batch_y, learning_rate):
        """Performs backpropagation on a batch of training data
            @param batch_X: training data input batch
            @param train_y: training data labels batch
            @returns: error of network after adaptation
        """
        # set delta_weights - list of layer weight matrices update values
        del_w = [np.zeros(shape=layer.weights.shape) for layer in self.layers]
        
        # total error
        err_tot = 0

        # calculate the error signal of the last layer
        for i in range(batch_X.shape[0]):
            # perform forward pass on training data instance
            ins, out = self.forward(batch_X[i])
            
            
            # # get network error for the training data instance
            err_tot += self.tfs.err(batch_y[i], out[-1]) # sum up over all output neurons
            
            # calculate the error signal of the last layer
            # pass in predicted output, layer output, and layer inputs
            e_sig = self.tfs.L_esig(ins[-1], batch_y[i], out[-1])
            
            for j in range(len(self.layers)-1, -1, -1):
                # calculate change in weights for this layer
                dw = np.asarray([learning_rate * e_sig[k] for k in range(e_sig.shape[0])])
                dw = np.asarray([dw] + [dw * out[j][i] for i in range(len(out[j]))])
                #print(dw.T)
                del_w[j] += dw.T

                # get the previous layer's error signal
                # pass in successor layer's error signal, the layer's activations, 
                #   and the weight matrix between the layer and it's successor layer
                if j > 0:
                    e_sig = self.tfs.H_esig(ins[j-1], e_sig, self.layers[j].weights)
        
        # after all weight updates have been found, apply them
        for i in range(len(self.layers)):
            self.layers[i].weights += del_w[i]

        # return network error
        return err_tot / batch_X.shape[0]


    def train(self, epochs: int, batch_size: int, learning_rate: float, 
              train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray):
        """Trains the neural network based on the given hyperparameters
            @param epochs: number of epochs, aka learning iterations
            @param batch_size: size of each training batch in each epoch
            @param learning_rate: step size of learning for gradient descent
            @returns: two arrays containing total network error for training set and
                validation set data for each epoch
        """
        # First check to make sure training data and labels are compatible with the network architecture
        
        # training and validation data cannot be empty
        if train_X.shape[0] < 1 or test_X.shape[0] < 1:
            raise Exception("Training data must not be empty")
        
        # training and validation data must have equal amounts of corresponding inputs/outputs
        if train_X.shape[0] != train_y.shape[0] or test_X.shape[0] != test_y.shape[0]:
            raise Exception("Amount of training data and labels must be equal")
        if (len(train_X.shape) == 1 and self.layers[0].weights.shape[1]-1 != 1) \
            or (len(train_X.shape) > 1 and train_X.shape[1] != self.layers[0].weights.shape[1]-1) \
            or (len(test_X.shape) == 1 and self.layers[0].weights.shape[1]-1 != 1) \
            or (len(test_X.shape) > 1 and test_X.shape[1] != self.layers[0].weights.shape[1]-1):
            raise Exception("Training data must have the same input dimensionality as the network")
        if (len(train_y.shape) == 1 and self.layers[-1].weights.shape[0] != 1) \
            or (len(train_y.shape) > 1 and train_y.shape[1] != self.layers[-1].weights.shape[0]) \
            or (len(test_y.shape) == 1 and self.layers[-1].weights.shape[0] != 1) \
            or (len(test_y.shape) > 1 and test_y.shape[1] != self.layers[-1].weights.shape[0]):
            raise Exception("Training labels must have the same output dimensionality as the network")


        # if batch_size > amount of training data, then set batch size to training data size
        bs = train_X.shape[0] if batch_size > train_X.shape[0] else batch_size

        # set plot interval (this step is not necessary for training)
        plot_interval = 1
        if epochs > 100:
            plot_interval = epochs // 100

        training_error = []
        validation_error = []

        for e in range(epochs):
            # generate indices of random training data sample/mini-batch
            batch_indices = np.random.choice(train_X.shape[0], bs, replace=False)
            error = self.backprop(train_X[batch_indices], train_y[batch_indices], learning_rate)

            # use this section for plotting error during training
            if e % plot_interval == 0 or e == epochs-1:
                training_error.append((e, error))
                validation_error.append((e, sum([self.tfs.err(test_y[i], self.forward(test_X[i])[1][-1]) for i in range(test_X.shape[0])]) / test_X.shape[0]))

        return (np.asarray(training_error), np.asarray(validation_error))


class Layer:
    """Represents a layer of a neural network
    """
    def __init__(self, architecture: (int, int)):
        """Creates a layer of a neural network with random normal weights
            @param architecture: 2-tuple representing (dim_in, dim_out) dimensions
        """
        # weights represents the weight matrix w_(k, k-1)
        self.weights = np.random.normal(size=(architecture[1], architecture[0]+1))
        self.dim_in = architecture[0]
        self.dim_out = architecture[1]