import numpy as np

def main():
    # l = np.asarray([1, 2, 3, 4])

    # a = np.asarray([3, 2])

    # print(np.asarray([l * a[i] for i in range(len(a))]))
    x = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.asarray([1, 2, 3, 4])
    idxs = np.asarray([1])
    print(x)
    print(y)
    print(x[idxs])
    print(y[idxs])
    print(np.delete(x, idxs, 0))
    print(np.delete(y, idxs, 0))
    print(x)
    print(y)

if __name__ == "__main__":
    main()