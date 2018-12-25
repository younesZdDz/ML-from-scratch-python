from linearRegCostFunction import linearRegCostFunction
from lrgradientReg import lrgradientReg
from scipy.optimize import minimize
import numpy as np
def trainLinearReg(X, y, reg):
    # initial_theta = np.zeros((X.shape[1],1))
    initial_theta = np.array([[15], [15]])
    # For some reason the minimize() function does not converge when using
    # zeros as initial theta.

    res = minimize(linearRegCostFunction, initial_theta, args=(X, y, reg), method=None, jac=lrgradientReg,
                   options={'maxiter': 5000})

    return (res)