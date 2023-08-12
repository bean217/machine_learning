# File: main.py
# Author: Benjamin Piro, benpiro1118@gmail.com
# Date: 12 August 2023
# Description: Example of using the neural network using scikit-learn's MNIST dataset.
#   Also plots batch average network error and displays resulting network accuracy.
#

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neural_net import ANN, plot_errs
from training_functions import mean_square_error_sigmoid, mean_square_error_relu, mean_square_error_tanh, multiple_cross_entropy_softmax


def main():
    # load mnist data
    mnist = load_digits()

    # map targets into a one-hot-encoding vector format
    new_targets = np.asarray([[0 if i != val else 1 for i in range(10)] for val in mnist.target])

    # create the neural network with the desired learning parameters (in this case, mse with a sigmoid activation)
    ann = ANN(mean_square_error_sigmoid, (64, 32, 16, 10))

    # split the data into training and validation partitions
    train_X, test_X, train_y, test_y = train_test_split(
        mnist.data, new_targets, test_size=0.25, random_state=123)

    # train the network
    train_errs, validation_errs = ann.train(epochs = 1000000,
              batch_size = 1,
              learning_rate = 0.01,
              train_X = train_X,
              train_y = train_y,
              test_X = test_X,
              test_y = test_y)

    # show the average error graph
    plot_errs(train_errs, validation_errs)

    # calculate training accuracy
    train_right = 0
    train_wrong = 0
    for i, x in enumerate(train_X):
        approx = np.argmax(ann.activate(x))
        if approx == np.argmax(train_y[i]):
            train_right += 1
        else:
            train_wrong += 1
    
    # calculate validation accuracy
    test_right = 0
    test_wrong = 0
    for i, x in enumerate(test_X):
        approx = np.argmax(ann.activate(x))
        if approx == np.argmax(test_y[i]):
            test_right += 1
        else:
            test_wrong += 1
    
    print("Training Accuracy:",train_right / (train_right + train_wrong))
    print("Validation Accuracy:", test_right / (test_right + test_wrong))

if __name__ == "__main__":
    main()