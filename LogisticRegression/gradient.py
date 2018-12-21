from sigmoid import sigmoid


def gradient (theta,X,y) :

    m=X.shape[0]
    theta=theta.reshape(-1,1)
    h=sigmoid(X.dot(theta))
    gradient=1/m*X.T.dot(h-y)
    return gradient.flatten()

