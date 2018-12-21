import matplotlib.pyplot as plt
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    neg = data[:, -1] == 0
    pos = data[:, -1] == 1

    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0],data[pos][:,1],marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True);
