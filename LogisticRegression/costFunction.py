import numpy as np
from sigmoid import sigmoid
def costFunction(theta,X,y) :
    m=X.shape[0]
    theta=theta.reshape(-1,1)
    h=sigmoid(X.dot(theta))
    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])

