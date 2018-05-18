
from heat import tensor


class KMeans:
    def __init__(self, n_clusters, max_iter, tol, random_state=42):
        # TODO: random_state = None default
        # TODO: make independent of torch
        self.tensor_type = 'torch.FloatTensor'
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def initialize_centroids(k, dimensions, seed):
        # zero-centered uniform random distribution in [-1, 1]
        tensor.set_gseed(seed)
        return tensor.uniform(shape=(1, dimensions, k), lower=-1, upper=1)

    @staticmethod
    def standardize(data):
        mean = data.mean(axis=0)
        diff = data - mean
        std = (diff ** 2).mean(axis=0) ** 0.5

        return mean, std, diff / std

    def fit(self, data):
        data.expand_dims(axis=2)

        # initialize the centroids randomly
        centroids = self.initialize_centroids(self.n_clusters, data.gshape[1], self.random_state)
        new_centroids = centroids.copy()

        for epoch in range(self.max_iter):
            # calculate the distance matrix and determine the closest centroid
            distances = ((data - centroids) ** 2).sum(axis=1)
            # TODO: changed order, splitaxis not None not implemented

            matching_centroids = distances.argmin(axis=2)

            # update the centroids
            for i in range(self.n_clusters):
                selection = (matching_centroids == i).astype(self.tensor_type)

                new_centroids[:, :, i:i + 1] = ((data * selection).sum(axis=0) /
                                                selection.sum(axis=0).clip(1.0, float('inf')))

            # check whether centroid movement has converged
            epsilon = ((centroids - new_centroids) ** 2).sum()
            centroids = new_centroids.copy()
            #print('Iteration:', epoch + 1, '/', self.max_iter, 'centroid movement:', epsilon)
            if self.tol is not None and epsilon <= self.tol:
                break

        return centroids
