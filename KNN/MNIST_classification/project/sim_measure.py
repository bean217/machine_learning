import numpy as np


def manhattan(a, b):
    """Calculates the manhattan distance between 2 vectors

    @param a: number/python array/numpy vector a
    @param b: number/python array/numpy vector b

    @returns: float64 representing the chebyshev distance between a and b
    """
    return p_norm(a, b, 1)


def euclidean(a, b):
    """Calculates the euclidean distance between 2 vectors

    @param a: number/python array/numpy vector a
    @param b: number/python array/numpy vector b

    @returns: float64 representing the chebyshev distance between a and b
    """
    return p_norm(a, b, 2)


def chebyshev(a, b):
    """Calculates the chebyshev distance between 2 vectors

    @param a: number/python array/numpy vector a
    @param b: number/python array/numpy vector b

    @returns: float64 representing the chebyshev distance between a and b
    """
    return max(np.abs(np.asarray(a).flatten() - np.asarray(b).flatten()))


def p_norm(a, b, p):
    """Calculates the p_norm distance between 2 vectors based on the value of p

    @param a: number/python array/numpy vector a
    @param b: number/python array/numpy vector b
    @param p: norm value, which must be a positive real number

    @returns: float64 representing the p-norm distance between a and b
    """
    if p < 0:
        raise Exception("p must be a positive real number")
    
    return np.sum(np.abs(np.asarray(a).flatten() - np.asarray(b).flatten()) ** p) ** (1./p)


def cosine(a, b) -> float:
    """Calculates the cosine similarity of two vectors

    @param a: number/python array/numpy vector a
    @param b: number/python array/numpy vector b

    @returns: float64 between 0.0 and 1.0
    """

    # convert a and b to vectors, because they may be numbers
    vec_a = np.asarray(a).flatten()
    vec_b = np.asarray(b).flatten()

    # calculate their individual magnitudes
    mag_a = np.sqrt(np.sum(vec_a ** 2))
    mag_b = np.sqrt(np.sum(vec_b ** 2))

    # normalize and return cosine similarity
    return np.sum(vec_a * vec_b) / (mag_a * mag_b)


#### TESTING ####

def main():
    print(cosine([1, 1, 0], [0, 1, 1]))


if __name__ == "__main__":
    main()