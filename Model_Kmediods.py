import numpy as np
from numpy.random import uniform
import random

from Evaluation import evaluation


# from Evaluation_Cluster import evaluation


def Model_Kmediods(X_train, true_labels):

    def median(point, data):
        """
        Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        return np.sqrt(np.sum((point - data) ** 2, axis=1).astype('float')) #np.median(point, data).astype('float')

    class KMediods:
        def __init__(self, n_clusters, sorted_points=0, max_iter=300):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.sorted_points = sorted_points

        def fit(self, X_train):
            # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
            # then the rest are initialized w/ probabilities proportional to their distances to the first
            # Pick a random point from train data for first centroid
            self.centroids = [random.choice(X_train)]
            for _ in range(self.n_clusters - 1):
                # Calculate distances from points to the centroids
                dists = np.sum([median(centroid, X_train) for centroid in self.centroids], axis=0)
                # Normalize the distances
                dists /= np.sum(dists)
                # Choose remaining points based on their distances
                new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
                # new_centroid_idx, = sol
                self.centroids += [X_train[new_centroid_idx]]
            # Iterate, adjusting centroids until converged or until passed max_iter
            iteration = 0
            prev_centroids = None
            while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
                # Sort each datapoint, assigning to nearest centroid
                self.sorted_points = [[] for _ in range(self.n_clusters)]
                for x in X_train:
                    dists = median(x, self.centroids)
                    centroid_idx = np.argmin(dists)
                    self.sorted_points[centroid_idx].append(x)
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                prev_centroids = self.centroids
                self.centroids = [np.mean(cluster, axis=0) for cluster in self.sorted_points]
                for i, centroid in enumerate(self.centroids):
                    if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                        self.centroids[i] = prev_centroids[i]
                iteration += 1


    centers = len(np.unique(true_labels))
    kmead = KMediods(n_clusters=centers)
    kmead.fit(X_train)
    return kmead






def Model__Kmediods(X_train, true_labels):

    def median(point, data):
        """
        Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        return np.sqrt(np.sum((point - data) ** 2, axis=1).astype('float')) #np.median(point, data).astype('float')

    class KMediods:
        def __init__(self, n_clusters, max_iter=300):
            self.n_clusters = n_clusters
            self.max_iter = max_iter

        def fit(self, X_train):
            # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
            # then the rest are initialized w/ probabilities proportional to their distances to the first
            # Pick a random point from train data for first centroid
            self.centroids = [random.choice(X_train)]
            for _ in range(self.n_clusters - 1):
                # Calculate distances from points to the centroids
                dists = np.sum([median(centroid, X_train) for centroid in self.centroids], axis=0)
                # Normalize the distances
                dists /= np.sum(dists)
                # Choose remaining points based on their distances
                new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
                self.centroids += [X_train[new_centroid_idx]]
            # This initial method of randomly selecting centroid starts is less effective
            # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
            # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
            # Iterate, adjusting centroids until converged or until passed max_iter
            iteration = 0
            prev_centroids = None
            while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
                # Sort each datapoint, assigning to nearest centroid
                self.sorted_points = [[] for _ in range(self.n_clusters)]
                for x in X_train:
                    dists = median(x, self.centroids)
                    centroid_idx = np.argmin(dists)
                    self.sorted_points[centroid_idx].append(x)
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                prev_centroids = self.centroids
                self.centroids = [np.mean(cluster, axis=0) for cluster in self.sorted_points]
                for i, centroid in enumerate(self.centroids):
                    if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                        self.centroids[i] = prev_centroids[i]
                iteration += 1

    centers = len(np.unique(true_labels))
    kmead = KMediods(n_clusters=centers)
    kmead.fit(X_train)
    kmead_cluster = np.zeros((X_train.shape[0])).astype('int')
    for i in range(len(X_train)):
        for j in range(len(kmead.sorted_points)):
            for k in range(len(kmead.sorted_points[j])):
                print(i, j, k)
                if (X_train[i] == kmead.sorted_points[j][k]).all():
                    kmead_cluster[i] = j + 1
    uni = np.unique(true_labels)
    Target = np.zeros((true_labels.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where(true_labels == uni[i])
        Target[ind[0], i] = 1

    Predict = np.zeros((kmead_cluster.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where(kmead_cluster == uni[i])
        Predict[ind[0], i] = 1
    Eval = evaluation(Predict, Target)
    return np.asarray(Eval).ravel()

