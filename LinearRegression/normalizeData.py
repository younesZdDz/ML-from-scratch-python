def normalizeData(X):
    X= (X - X.mean()) / (X.max() - X.min())
    return X
