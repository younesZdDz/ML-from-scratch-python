import numpy as np


def normalEquation(X,y):
    theta = np.zeros(X.shape[1])
    theta= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta