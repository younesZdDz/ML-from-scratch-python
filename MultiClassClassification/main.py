import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from sklearn.linear_model import LogisticRegression
from predict import predict


data = loadmat('../data/ex3data1.mat')
#data.keys()

weights = loadmat('../data/ex3weights.mat')
#weights.keys()


y = data['y']
# Add constant for intercept
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]

#print('X: {} (with intercept)'.format(X.shape))
#print('y: {}'.format(y.shape))


theta1, theta2 = weights['Theta1'], weights['Theta2']
#print('theta1: {}'.format(theta1.shape))
#print('theta2: {}'.format(theta2.shape))

sample = np.random.choice(X.shape[0], 20)
plt.imshow(X[sample,1:].reshape(-1,20).T)
plt.show()


theta = oneVsAll(X, y, 10, 0.1)


pred = predictOneVsAll(theta, X)

print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))


clf = LogisticRegression(C=10, penalty='l2', solver='liblinear')
# Scikit-learn fits intercept automatically, so we exclude first column with 'ones' from X when fitting.
clf.fit(X[:,1:],y.ravel())
pred2 = clf.predict(X[:,1:])
print('Training set accuracy: {} %'.format(np.mean(pred2 == y.ravel())*100))


pred = predict(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))
