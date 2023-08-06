from keras.datasets import mnist

def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print('X_train:', str(train_X))
    print('Y_train:', str(train_y))
    print('X_test:', str(test_X))
    print('Y_test:', str(test_y))
    pass

if __name__ == "__main__":
    main()