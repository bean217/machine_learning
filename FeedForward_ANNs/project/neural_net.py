import numpy as np

from training_functions import Train_Funcs
from training_functions import mean_square_error_sigmoid, mean_square_error_relu, mean_square_error_tanh, multiple_cross_entropy_softmax

def train_test_split(data: np.ndarray, labels: np.ndarray, validation_percentage: float = 0.33):
    """Splits training data into a training and validation set
        @params data: training data inputs
        @params labels: training data labels
        @params validation_percentage: percentage of training data to use as validation
        @returns: 4-tuple with (training_data, training_labels, validation_data, validation_labels)
    """
    # get number of validation points based on percentage
    num_validation = int(data.shape[0] * validation_percentage)
    # get validation data points from training data
    validation_indices = np.random.choice(data.shape[0], num_validation, replace=False)
    # return (training_data, training_labels, validation_data, validation_labels)
    return (np.delete(data, validation_indices), np.delete(labels, validation_indices), 
            data[validation_indices], labels[validation_indices])


class ANN:
    def __init__(self, training_functions: Train_Funcs, architecture: tuple):
        """Creates a feed forward neural network model based on an architecture
            @param architecture: list of integers containing number of neurons in each layer
        """
        if len(architecture) < 3:
            raise Exception("Must have at least 3 network layers")

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
    

    def backprop(self, batch_X, batch_y, learning_rate):
        """Performs backpropagation on a batch of training data
            @param batch_X: training data input batch
            @param train_y: training data labels batch
            @returns: error of network after adaptation
        """
        # set delta_weights - list of layer weight matrices update values
        del_w = [np.zeros(shape=layer.weights.shape) for layer in self.layers]
        # TODO:
        # total error
        err_tot = 0

        # calculate the error signal of the last layer
        for i in range(batch_X.shape[0]):
            # perform forward pass on training data instance
            ins, out = self.forward(batch_X[i])
            
            # TODO:
            # # get network error for the training data instance
            err_tot += sum(self.tfs.err(batch_y[i], out[-1])) # sum up over all output neurons
            
            # calculate input for each neuron in the last layer

            # calculate the error signal of the last layer
            # pass in predicted output, layer output, and layer inputs
            e_sig = self.tfs.L_esig(ins[-1], batch_y[i], out[-1])
            
            for j in range(len(self.layers)-1, -1, -1):
                # calculate change in weights for this layer
                dw = np.asarray([learning_rate * e_sig[k] for k in range(e_sig.shape[0])])
                dw = np.asarray([dw] + [dw * out[j][i] for i in range(len(out[j]))])
                #print(dw.T)
                del_w[j] += dw.T
                #TODO: REMOVE: del_w[j] += np.asarray([-learning_rate * e_sig[k] * out[j][] for k in range(e_sig.shape[0])])
                # get the previous layer's error signal
                # pass in successor layer's error signal, the layer's activations, 
                #   and the weight matrix between the layer and it's successor layer
                e_sig = self.tfs.H_esig(ins[j-1], e_sig, self.layers[j].weights)
        
        # after all weight updates have been found, apply them
        # for i in range(len(self.layers)):
        #     print("Layer",i)
        #     print(self.layers[i].weights)
        # print()
        for i in range(len(self.layers)):
            self.layers[i].weights += del_w[i]
        # print()
        # for i in range(len(self.layers)):
        #     print("Layer",i)
        #     print(self.layers[i].weights)

        # TODO:
        return err_tot


    def train(self, epochs: int, batch_size: int, learning_rate: float, 
              data: np.ndarray, labels: np.ndarray, validation_percentage: float = 0.33):
        """Trains the neural network based on the given hyperparameters
            @param epochs: number of epochs, aka learning iterations
            @param batch_size: size of each training batch in each epoch
            @param learning_rate: step size of learning for gradient descent
            @returns: two arrays containing total network error for training set and
                validation set data for each epoch
        """
        # First check to make sure training data and labels are compatible with the network architecture
        if data.shape[0] < 1:
            raise Exception("Training data must not be empty")
        if len(data.shape) != 2:
            raise Exception("Training instances must be vectors")
        if data.shape[0] != labels.shape[0]:
            raise Exception("Amount of training data and labels must be equal")
        if data.shape[1] != self.layers[0].weights.shape[1]-1:
            raise Exception("Training data must have the same input dimensionality as the network")
        if labels.shape[1] != self.layers[-1].weights.shape[1]-1:
            raise Exception("Training labels must have the same output dimensionality as the network")

        # split up training data into training and validation
        train_X, train_y, test_X, test_y = train_test_split(data, labels, validation_percentage)

        # if batch_size > amount of training data, then set batch size to training data size
        bs = self.data.shape[0] if batch_size > self.data.shape[0] else batch_size

        # set plot interval (this step is not necessary for training)
        plot_interval = epochs
        if plot_interval > 100:
            plot_interval /= 100

        training_error = []
        validation_error = []

        for e in range(epochs):
            # generate indices of random training data sample/mini-batch
            batch_indices = np.random.choice(self.train_X.shape[0], bs, replace=False)
            error = self.backprop(train_X[batch_indices], train_y[batch_indices], learning_rate)

            # use this section for plotting error during training
            if e % plot_interval == 0:
                training_error.append(error)
                validation_error.append(sum([self.tfs.err(test_X[i], test_y[i]) for i in range(test_X.shape[0])]))
        
        return (training_error, validation_error)


class Layer:
    """Represents a layer of a neural network
    """
    def __init__(self, architecture: (int, int)):
        """Creates a layer of a neural network with random normal weights
            @param architecture: 2-tuple representing (dim_in, dim_out) dimensions
        """
        # weights represents the weight matrix w_(k, k-1)
        #self.weights = np.ones(shape=(architecture[1], architecture[0]+1)) * 1.5
        self.weights = np.random.normal(size=(architecture[1], architecture[0]+1))
        self.dim_in = architecture[0]
        self.dim_out = architecture[1]
        print(self.weights)
        print(self.weights.shape)
        pass

def main():
    # set training function parameters
    # sigmoid = lambda x: 1/(1+np.exp(-x))
    # d_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
    # err = lambda y, y_pred: ((y - y_pred) ** 2)/2
    # tfs = Train_Funcs(sigmoid, d_sigmoid, err, None, None)

    # create network
    ann = ANN(mean_square_error_sigmoid, (2, 2, 1))

    res = ann.forward(np.asarray([1, 1]))

    print("ins", res[0])
    print("outs", res[1])
    
    bs = 2
    for i in range(2500):
        x = np.array([[0, -.5], [-.5, 0], [0, .5], [.5, 0]])
        l = np.array([0, 0, 1, 1])
        
        for j in range(4):

            err = ann.backprop(x[0:bs],
                         l[0:bs],
                         0.1)
        if i == 0 or i == 2499:
            print("ERROR:", err)


    res = ann.forward(np.asarray([1, 1]))

    print("ins", res[0])
    print("outs", res[1])

    # l = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # idxs = np.random.choice(l.shape[0], 2, replace=False) 
    # print(l[idxs])

    pass

if __name__ == "__main__":
    main()