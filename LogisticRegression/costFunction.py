import numpy as np

def costFunction(X,y,theta=[]) :
    n=X.shape[1]
    m=X.shape[0]
    if(len(theta)==0) :
        theta=[[0]]*n
    h=sigmoid(X*theta)
    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))