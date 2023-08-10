import numpy as np

class test:
    def __init__(self):
        print("made")

    def myfunc(self, x, y):
        print("hi")

    def myfunc(self, x):
        print("hello")

def main():
    # l = np.asarray([1, 2, 3, 4])

    # a = np.asarray([3, 2])

    # print(np.asarray([l * a[i] for i in range(len(a))]))

    c = test()

    c.myfunc(1)
    c.myfunc(1, 2)


if __name__ == "__main__":
    main()