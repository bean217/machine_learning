import numpy as np

class ANN:
    def __init__(self, architecture: tuple):
        """Creates a feed forward neural network model based on an architecture
            @param architecture: list of integers containing number of neurons in each layer
        """
        if len(architecture) < 3:
            raise Exception("Must have at least 3 network layers")

        # create network layers
        self.layers = []
        for i in range(len(architecture)-1):
            self.layers.append(Layer((architecture[i], architecture[i+1])))

        # instantiate activation function to None (to be set later)
        self.act = lambda x: 1/(1 + np.exp(-x))#None
        self.data = None
        self.labels = None


    def fit(self, data: np.ndarray, labels: np.ndarray):
        # Check to make sure training data and labels are compatible with the network architecture
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
        # Set network training data properties
        self.data = data
        self.labels = labels


    def forward(self, x: np.ndarray):
        """Performs a forward pass through the network
            @param x: data instance serving as the network input
            @returns: an array of the network's input and outputs through each layer
        """
        if self.act == None:
            raise Exception("Activation function not set")
        # keep track of inputs/outputs as they propagate through laters
        outputs = [x]
        curr = x
        for l in self.layers:
            # pad with 1 for bias
            curr = np.concatenate(([1], curr))
            # get output of layer
            curr = np.asarray([self.act(curr @ l.weights[i,:]) for i in range(l.dim_out)])
            # append to outputs
            outputs.append([curr])
        return outputs
    

    def backprop(self):
        if self.data == None or self.labels == None:
            raise Exception("Network must have training data to perform backpropagation")
        


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
        print(self.weights)
        print(self.weights.shape)
        pass

def main():
    ann = ANN((2, 3, 2))
    print(ann.forward(np.asarray([1, 1])))
    pass

if __name__ == "__main__":
    main()