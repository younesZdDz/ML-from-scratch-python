import numpy as np
from scipy.special import expit


def predict(theta_1, theta_2, X):
    z2 = theta_1.dot(X.T)
    a2 = np.c_[np.ones((X.shape[0], 1)), expit(z2).T]

    z3 = a2.dot(theta_2.T)
    a3 = expit(z3)

    return (np.argmax(a3, axis=1) + 1)