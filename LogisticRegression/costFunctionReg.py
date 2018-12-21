import numpy as np
from scipy.special import expit

def costFunctionReg(theta, reg, XX,y):
    m = y.size
    h = expit(XX.dot(theta))

    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg / (2 * m)) * np.sum(
        np.square(theta[1:]))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])