import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])


class KMeans(object):
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid])
                             for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = KMeans()
clf.fit(X)

colors = ["g", "r", "c", "b", "k", "o", "y"]

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid]
                [1], marker='o', s=150, linewidths=5, color='k')

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x',
                    color=color, s=150, linewidths=5)

to_predicts = np.array([[1, 0],
                        [-5, 3],
                        [3, 4],
                        [-4, 5],
                        [5, 7],
                        [7, -1],
                        [3, 9]])

for to_predict in to_predicts:
    classification = clf.predict(to_predict)
    plt.scatter(to_predict[0], to_predict[1], marker="*",
                color=colors[classification], s=150, linewidths=5)


plt.show()
