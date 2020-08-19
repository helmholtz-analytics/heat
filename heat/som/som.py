import warnings

import heat as ht
import numpy as np
import random


class FixedSOM(ht.BaseEstimator, ht.ClusteringMixin):
    """
    HeAT implementation of a self-organizing map, reducing the n-dimensional data vectors onto a 2-D fixed grid of
    size height x width.

    Parameters
    ----------
    height: int
        Height of the fixed grid
    width:
        Width of the fixed grid
    data_dim:
        dimension of the data vectors
    learning_rate: float
        the learning rate
    --------
    """

    def __init__(
        self,
        height,
        width,
        data_dim,
        initial_learning_rate,
        target_learning_rate,
        initial_radius,
        target_radius,
        max_epoch,
        seed,
        batch_size=1,
        data_split=None,
    ):
        self.height = height
        self.width = width
        self.data_dim = data_dim
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.target_learning_rate = target_learning_rate
        self.initial_radius = initial_radius
        self.radius = initial_radius
        self.target_radius = target_radius
        self.max_epoch = max_epoch

        ht.core.random.seed(seed)

        self.network = ht.random.rand(height * width, data_dim, dtype=ht.float64, split=data_split)
        self.batch_size = batch_size

        self.network_indices = ht.array(
            [(j, i) for i in range(0, self.width) for j in range(0, self.height)], split=0
        )
        self.distances = self.precompute_distances()

    def fit_iterative(self, X):
        self.learning_rate = self.initial_learning_rate
        self.radius = self.initial_radius

        batch_count = int(X.gshape[0] / self.batch_size)
        offset = 0
        batches = []

        for count in range(1, batch_count + 1):
            batches.append(X[offset : count * self.batch_size])
            batches[-1].balance_()
            offset = count * self.batch_size

        for epoch in range(1, self.max_epoch):
            for count in range(batch_count):
                batch = batches[count]
                distances = ht.spatial.cdist(batch, self.network)
                row_min = ht.min(distances, axis=1, keepdim=True)
                min_distances = ht.where(distances == row_min, 1, 0)

                self.update_weights(min_distances, batch)

                self.update_learning_rate(epoch)
                self.update_radius(epoch)

    def fit_batch(self, X, c):

        self.radius = self.initial_radius
        self.learning_rate = self.initial_learning_rate

        for epoch in range(1, self.max_epoch + 1):
            distances = ht.spatial.cdist(X, self.network)
            row_min = ht.min(distances, axis=1, keepdim=True)

            scalars = self.in_radius()
            distances = ht.where(distances == row_min, 1, 0)
            scalars = ht.matmul(distances, scalars)
            scaled_weights = ht.matmul(X.T, scalars)
            scalar_sum = ht.sum(scalars, axis=0, keepdim=True)
            new_network = scaled_weights.T / scalar_sum.T
            dist = ht.sum(ht.spatial.cdist(self.network, new_network))

            self.network = new_network

            self.update_learning_rate(epoch)
            self.update_radius(epoch)

            if dist <= c:
                break

    def predict(self, X):
        distances = ht.spatial.cdist(X, self.network)
        winner_ind = ht.argmin(distances, 1, axis=1)
        translated_winner_ind = self.network_indices[winner_ind.tolist()]
        translated_winner_ind.balance_()
        return translated_winner_ind

    def update_learning_rate(self, epoch):
        self.learning_rate = self.initial_learning_rate + (
            self.target_learning_rate - self.initial_learning_rate
        ) * (epoch / self.max_epoch)

    def update_radius(self, epoch):
        self.radius = self.initial_radius * (
            (self.target_radius / self.initial_radius) ** (epoch / self.max_epoch)
        )

    def in_radius(self,):
        dist = ht.exp((ht.pow(self.distances, 2) * -1) / (2 * ht.pow(self.radius, 2)))
        return dist

    def precompute_distances(self,):
        return ht.resplit(ht.spatial.cdist(self.network_indices, self.network_indices), axis=1)

    def update_weights(self, winner_cells, weights):
        scalars = self.learning_rate * self.in_radius()
        scalars = ht.where(self.distances <= self.radius, scalars, 0)
        scalars = ht.matmul(winner_cells, scalars)
        weights = ht.expand_dims(weights, 1)
        distances = ht.sub(weights, self.network)
        scalars = ht.expand_dims(scalars, 2)
        scaled_distances = scalars * distances
        scaled_distances = ht.sum(scaled_distances, axis=0)
        self.network = self.network + scaled_distances

    def umatrix(self,):
        """
        Returns a umatrix representation of the network. Each Cell contains the distance to the neighbouring weights.
        Neighbours are cells in the moore neighbourhood 1.
        """
        network_distances = ht.spatial.cdist(self.network, self.network)
        radius = ht.where(self.distances != 0, self.distances, 2)
        radius = ht.resplit(radius, 0)
        radius = ht.where(radius < 2, 1, 0)
        selected_distances = network_distances * radius
        sum_distances = ht.sum(selected_distances, axis=1)
        distances = ht.reshape(sum_distances, (self.height, self.width))

        return distances

    def get_2d_network(self, array):
        array = self.network_indices[array]
        array.balance_()

        return array
