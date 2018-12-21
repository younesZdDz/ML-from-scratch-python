import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt('../data/ex2data1.txt',delimiter=',')

X=data[:,0:-1]
X = np.c_[np.ones(data.shape[0]),X]
y=np.c_(data[-1])

