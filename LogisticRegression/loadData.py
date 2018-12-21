import numpy as np

def loadData(file,delimiter=',') :
    data = np.loadtxt(file, delimiter=delimiter)
    #print('Dimensions: ', data.shape)
    #print(data[1:6, :])
    return(data)


