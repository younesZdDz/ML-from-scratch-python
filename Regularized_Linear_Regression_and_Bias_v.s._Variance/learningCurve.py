from linearRegCostFunction import linearRegCostFunction
from lrgradientReg import lrgradientReg
import numpy as np
from trainLinearReg import trainLinearReg

def learningCurve(X, y, Xval, yval, reg):
    m = y.size

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in np.arange(m):
        res = trainLinearReg(X[:i + 1], y[:i + 1], reg)
        error_train[i] = linearRegCostFunction(res.x, X[:i + 1], y[:i + 1], reg)
        error_val[i] = linearRegCostFunction(res.x, Xval, yval, reg)

    return (error_train, error_val)