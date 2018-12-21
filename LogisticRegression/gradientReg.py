from sigmoid import sigmoid
import numpy as np

def gradientReg(theta, reg, XX,y):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1, 1)))

    grad = (1 / m) * XX.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())