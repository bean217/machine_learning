from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


import numpy as np


from sim_measure import manhattan, euclidean, chebyshev, cosine


from matplotlib import pyplot as plt


from KNN import KNN


def evaluate():
    accuracies = {
        "manhattan": [],
        "euclidean": [],
        "chebyshev": [],
        #"cosine": []
    }

    mnist = load_digits()
    train_X, test_X, train_y, test_y = train_test_split(
      mnist.data, mnist.target, test_size=0.25, random_state=123)

    knn = KNN()
    knn.fit(train_X, train_y)

    k_vals = np.arange(1, 100, 2)
    for k in k_vals:
        print("evaluating k =", k)
        knn.set_num_neighbors(k)
        for key in accuracies.keys():
            knn.set_sim_measure(key)
            accuracies[key].append(knn.evaluate(test_X, test_y))
    
    for key in accuracies.keys():
        print(key)
        print("\t",accuracies[key])

    return (k_vals, accuracies)
            

def plot(k_vals, accuracies):
    for key in accuracies.keys():
        plt.plot(k_vals, accuracies[key])
        plt.xlabel("K Value")
        plt.ylabel("Accuracy")
    
    plt.show()


def main():    
    # mnist = load_digits()
    # train_X, test_X, train_y, test_y = train_test_split(
    #   mnist.data, mnist.target, test_size=0.25, random_state=123)

    # knn = KNN(5, sim_measure="L2")
    # knn.fit(train_X, train_y)
    # knn.evaluate(test_X, test_y)
    # knn.eval_report()
    k_vals, accuracies = evaluate()
    plot(k_vals, accuracies)

if __name__ == "__main__":
    main()
