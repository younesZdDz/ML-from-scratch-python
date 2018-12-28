from scipy.io import loadmat
from sklearn.svm import SVC
import numpy as np
from plotData import plotData
from plot_svc import plot_svc
import matplotlib.pyplot as plt

data1 = loadmat('../data/ex6data1.mat')

y1 = data1['y']
X1 = data1['X']


plotData(X1,y1)

plt.show()

clf = SVC(C=1.0, kernel='linear')
clf.fit(X1,y1.ravel())
plot_svc(clf,X1,y1)
plt.show()


data2 = loadmat('../data/ex6data2.mat')
y2 = data2['y']
X2 = data2['X']
plotData(X2, y2)
clf2 = SVC(C=50, kernel='rbf', gamma=6)
clf2.fit(X2, y2.ravel())
plot_svc(clf2, X2, y2)
plt.show()


data3 = loadmat('../data/ex6data3.mat')
y3 = data3['y']
X3 = data3['X']
plotData(X3, y3)
clf3 = SVC(C=1.0, kernel='poly', degree=8, gamma=10)
clf3.fit(X3, y3.ravel())
plot_svc(clf3, X3, y3)
plt.show()
