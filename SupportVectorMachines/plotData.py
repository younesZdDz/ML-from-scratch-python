import matplotlib.pyplot as plt

def plotData(X, y):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.scatter(X[pos, 0], X[pos, 1], s=60, c='k', marker='+', linewidths=1)
    plt.scatter(X[neg, 0], X[neg, 1], s=60, c='y', marker='o', linewidths=1)