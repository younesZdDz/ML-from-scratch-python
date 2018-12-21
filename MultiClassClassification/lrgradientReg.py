from scipy.special import expit
import numpy as np

def lrgradientReg(theta, reg, XX,y):
    m = y.size
    h = expit(XX.dot(theta.reshape(-1, 1)))

    grad = (1 / m) * XX.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())