import numpy as np

def linearRegCostFunction(theta, X, y, reg):
    m = y.size

    h = X.dot(theta)

    J = (1 / (2 * m)) * np.sum(np.square(h - y)) + (reg / (2 * m)) * np.sum(np.square(theta[1:]))

    return (J)