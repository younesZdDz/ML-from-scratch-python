import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from nnCostFunction import  nnCostFunction


data = loadmat('../data/ex4data1.mat')
#data.keys()

weights = loadmat('../data/ex4weights.mat')
#weights.keys()


y = data['y']
# Add constant for intercept
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]

#print('X: {} (with intercept)'.format(X.shape))
#print('y: {}'.format(y.shape))


theta1, theta2 = weights['Theta1'], weights['Theta2']
#print('theta1: {}'.format(theta1.shape))
#print('theta2: {}'.format(theta2.shape))


params = np.r_[theta1.ravel(), theta2.ravel()]

print(nnCostFunction(params, 400, 25, 10, X, y, 0)[0])

print(nnCostFunction(params, 400, 25, 10, X, y, 1)[0])

