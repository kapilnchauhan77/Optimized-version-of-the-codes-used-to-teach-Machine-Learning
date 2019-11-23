from sklearn.datasets.samples_generator import make_blobs
from matplotlib import style
from sklearn.cluster import MeanShift
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
style.use('fivethirtyeight')

centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1)

ms = MeanShift()
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(f"cluster centers:{cluster_centers}")
n_clusters = len(np.unique(labels))
print(f"Number of clusters: {n_clusters}")
colors = 10*["g", "r", "c", "b", "k", "o", "y"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(X)):
    ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=150, marker='x', color=colors[labels[i]])

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
           cluster_centers[:, 2], s=150, marker='o', zorder=10, linewidths=5)
plt.show()
