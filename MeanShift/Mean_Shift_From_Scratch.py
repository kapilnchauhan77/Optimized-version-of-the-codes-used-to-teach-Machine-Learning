import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import random
from matplotlib import style
style.use('fivethirtyeight')

centers = random.randrange(2, 5)

X, _ = make_blobs(n_samples=50, centers=centers, n_features=3)

colors = 10 * ["g", "r", "c", "b", "k", "y"]


class MeanShift(object):
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):
        if self.radius == None:
            all_data_centroid = np.average(abs(data), axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        for idx, i in enumerate(data):
            centroids[idx] = i

        weights = [i for i in range(self.radius_norm_step)][::-1]
        while 1:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 1e-8
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1
                    to_add = (weights[weight_index]**2) * [featureset]

                    in_bandwidth += to_add
                    # in_bandwidth += [(weights[weight_index]**2) * feature for feature in featureset]

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if ii == i:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius and ii not in to_pop:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                uniques.remove(i)

            prev_centroids = dict(centroids)

            centroids = {}
            for idx, i in enumerate(uniques):
                centroids[idx] = np.array(i)

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

                if not optimized:
                    break
            if optimized:
                break

        self.centroids = centroids

        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid])
                         for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        classifications = []
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid])
                         for centroid in self.centroids]
            classifications.append(distances.index(min(distances)))
        return classifications


clf = MeanShift(radius_norm_step=100)
clf.fit(X)

centroids = clf.centroids
print(centroids)

for classification in clf.classifications:
    color = colors[classification]
    for features in clf.classifications[classification]:
        plt.scatter(features[0], features[1], marker='x',
                    color=color, s=150, linewidths=5)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()
print(centers)
