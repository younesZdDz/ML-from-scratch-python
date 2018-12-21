import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from loadData import loadData
from plotData import plotData
from gradientReg import gradientReg
import matplotlib.pyplot as plt
from costFunctionReg import costFunctionReg

from scipy.optimize import minimize
from scipy.special import expit
from predict import predict

data = loadData('../data/ex2data2.txt', ',')
y = np.c_[data[:,2]]
X = data[:,0:2]

plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
plt.show()

poly = PolynomialFeatures(6)
XX = poly.fit_transform(data[:,0:2])
XX.shape

initial_theta = np.zeros(XX.shape[1])
print(costFunctionReg(initial_theta, 1, XX, y))

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

# Decision boundaries
# Lambda = 0 : No regularization --> too flexible, overfitting the training data
# Lambda = 1 : Looks about right
# Lambda = 100 : Too much regularization --> high bias

for i, C in enumerate([0, 1, 100]):
    # Optimize costFunctionReg
    res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), method=None, jac=gradientReg,
                    options={'maxiter': 3000})

    # Accuracy
    accuracy = 100 * sum(predict(res2.x, XX) == y.ravel()) / y.size

    # Scatter plot of X,y
    plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])

    # Plot decisionboundary
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = expit(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
plt.show()


