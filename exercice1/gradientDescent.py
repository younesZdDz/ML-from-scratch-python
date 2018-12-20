
import numpy as np
import math
from ComputeCost import computeCost

def gradientDescent(X, y, theta=[], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    n=X.shape[1]
    if len(theta)==0 :
        theta=[[0]]*n
    n=X.shape[1]
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1 / m) * (X.T.dot(h - y))
        J_history[iter] = computeCost(X, y, theta)
    return (theta, J_history)