import numpy as np
import pandas as pd
from scipy.special import expit
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, features, classes, reg):
    m=features.shape[0]
    n=features.shape[1]

    theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size,
                                                                               (input_layer_size + 1))
    theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1))

    theta1_grad=np.zeros((hidden_layer_size,input_layer_size))
    theta2_grad=np.zeros((hidden_layer_size,input_layer_size))

    J=0

    a1=features
    a2=np.c_[np.ones((features.shape[0],1)), expit(a1.dot(theta1.T))]
    h= expit(a2.dot(theta2.T))

    y_matrix = pd.get_dummies(classes.ravel()).as_matrix()

    J=-1/m*np.sum(y_matrix * (np.log(h)) +  (1-y_matrix) * (np.log(1-(h))))  \
      +  reg/(2*m)*(np.sum(np.square(theta1[:,1:]))+np.sum(np.square(theta2[:,1:])))

    # Gradients
    d3 = h - y_matrix  # 5000x10
    d2 = theta2[:, 1:].T.dot(d3.T) * sigmoidGradient(theta1.dot(a1.T))  # 25x10 *10x5000 * 25x5000 = 25x5000

    delta1 = d2.dot(a1)  # 25x5000 * 5000x401 = 25x401
    delta2 = d3.T.dot(a2)  # 10x5000 *5000x26 = 10x26

    theta1_ = np.c_[np.ones((theta1.shape[0], 1)), theta1[:, 1:]]
    theta2_ = np.c_[np.ones((theta2.shape[0], 1)), theta2[:, 1:]]

    theta1_grad = delta1 / m + (theta1_ * reg) / m
    theta2_grad = delta2 / m + (theta2_ * reg) / m

    return (J, theta1_grad, theta2_grad)