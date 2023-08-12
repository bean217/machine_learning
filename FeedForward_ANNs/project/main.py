import numpy as np
from sklearn.datasets import load_digits

from neural_net import ANN, plot_errs
from training_functions import mean_square_error_sigmoid, mean_square_error_relu, mean_square_error_tanh, multiple_cross_entropy_softmax


def main():
    # fetch data
    mnist = load_digits()

    new_targets = np.asarray([[0 if i != val else 1 for i in range(10)] for val in mnist.target])

    ann = ANN(mean_square_error_sigmoid, (64, 10, 10))

    train_errs, validation_errs = ann.train(epochs = 10000,
              batch_size = 32,
              learning_rate = 0.001,
              data = mnist.data,
              labels = new_targets,
              validation_percentage = 0.2)
    
    plot_errs(train_errs, validation_errs)


if __name__ == "__main__":
    main()