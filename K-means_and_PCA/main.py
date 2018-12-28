import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from scipy import linalg

data1 = loadmat('../data/ex7data2.mat')

X1 = data1['X']
km1 = KMeans(3)
km1.fit(X1)
plt.scatter(X1[:,0], X1[:,1], s=40, c=km1.labels_, cmap=plt.cm.prism)
plt.title('K-Means Clustering Results with K=3')
plt.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);
plt.show()
img = plt.imread('../data/bird_small.png')
img_shape = img.shape
A = img/255

AA = A.reshape(128*128,3)

km2 = KMeans(16)
km2.fit(AA)
B = km2.cluster_centers_[km2.labels_].reshape(img_shape[0], img_shape[1], 3)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,9))
ax1.imshow(img)
ax1.set_title('Original')
ax2.imshow(B*255)
ax2.set_title('Compressed, with 16 colors')

for ax in fig.axes:
    ax.axis('off')

plt.show()


data2 = loadmat('../data/ex7data1.mat')
X2 = data2['X']
scaler = StandardScaler()
scaler.fit(X2)
U, S, V = linalg.svd(scaler.transform(X2).T)
print(U)
print(S)


plt.scatter(X2[:,0], X2[:,1], s=30, edgecolors='b',facecolors='None', linewidth=1);
# setting aspect ratio to 'equal' in order to show orthogonality of principal components in the plot
plt.gca().set_aspect('equal')
plt.quiver(scaler.mean_[0], scaler.mean_[1], U[0,0], U[0,1],scale=S[0], color='r')
plt.quiver(scaler.mean_[0], scaler.mean_[1], U[1,0], U[1,1], scale=S[1], color='r');
plt.show()
print(S[1])
