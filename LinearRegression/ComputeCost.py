import numpy as np

def computeCost(X, y, theta=[]):
    m = y.size
    J = 0
    n=X.shape[1]
    if len(theta) == 0:
        theta = [[0]] * n
    h = X.dot(theta)

    J = 1 / (2 * m) * np.sum(np.square(h - y))

    return (J)