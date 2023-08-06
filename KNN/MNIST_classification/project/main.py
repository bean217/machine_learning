from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


from KNN import KNN


def main():    
    mnist = load_digits()
    train_X, test_X, train_y, test_y = train_test_split(
      mnist.data, mnist.target, test_size=0.25, random_state=123)

    knn = KNN(5, sim_measure="L2")
    knn.fit(train_X, train_y)
    knn.evaluate(test_X, test_y)
    knn.eval_report()

if __name__ == "__main__":
    main()
