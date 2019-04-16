import sys

import heat as ht


class KMeans:
    def __init__(self, n_clusters, max_iter=1000, tol=1e-4, random_state=42):
        # TODO: document me
        # TODO: extend the parameter list
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def initialize_centroids(k, dimensions, seed, device):
        # TODO: document me
        # TODO: extend me with further initialization methods
        # zero-centered uniform random distribution in [-1, 1]
        ht.random.set_gseed(seed)
        return ht.random.uniform(low=-1.0, high=1.0, size=(1, dimensions, k), device=device)

    def fit(self, data):
        # TODO: document me
        data = data.expand_dims(axis=2)

        # initialize the centroids randomly
        centroids = self.initialize_centroids(
            self.n_clusters, data.shape[1], self.random_state, data.device)
        new_centroids = centroids.copy()

        for epoch in range(self.max_iter):
            # calculate the distance matrix and determine the closest centroid
            distances = ((data - centroids) ** 2).sum(axis=1, keepdim=True)
            matching_centroids = distances.argmin(axis=2, keepdim=True)

            # update the centroids
            for i in range(self.n_clusters):
                selection = (matching_centroids == i).astype(ht.int64)
                new_centroids[:, :, i:i + 1] = ((data * selection).sum(axis=0, keepdim=True) /
                                                selection.sum(axis=0, keepdim=True).clip(1.0, sys.maxsize))

            # check whether centroid movement has converged
            epsilon = ((centroids - new_centroids) ** 2).sum()
            centroids = new_centroids.copy()
            if self.tol is not None and epsilon <= self.tol:
                break

        return centroids
