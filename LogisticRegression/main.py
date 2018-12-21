import numpy as np
import matplotlib.pyplot as plt
from loadData import loadData
from plotData import plotData
from costFunction import costFunction
from gradient import  gradient
from scipy.optimize import minimize
from sigmoid import  sigmoid
from predict import predict
data=loadData('../data/ex2data1.txt')

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
plt.show()
X=data[:,0:-1]
X = np.c_[np.ones(data.shape[0]),X]
y =np.c_[data[:,-1]]


initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta,X, y)
grad = gradient(initial_theta,X, y)
print('////////////////////////////////////////////////////////////////////')
print('Cost: \n', cost)
print('Grad: \n', grad)


res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})
print('////////////////////////////////////////////////////////////////////')

print(sigmoid(np.array([1, 45, 85]).dot(res.x.T)))

print('////////////////////////////////////////////////////////////////////')

p = predict(res.x, X)
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

print('////////////////////////////////////////////////////////////////////')


plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max()
x2_min, x2_max = X[:,2].min(), X[:,2].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.2,0.5,0.7], linewidths=1, colors='b');
plt.show()

print('////////////////////////////////////////////////////////////////////')

