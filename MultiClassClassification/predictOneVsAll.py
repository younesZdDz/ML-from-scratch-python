from scipy.special import expit
import numpy as np


def predictOneVsAll(all_theta, X):
    predict= expit(X.dot(all_theta.T))
    np.argmax
    return(np.argmax(predict, axis=1)+1)
