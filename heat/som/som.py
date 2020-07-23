import warnings

import heat as ht
import numpy as np


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
        print(ht.min(self.network))
        self.batch_size = batch_size

        self.network_indices = ht.array(
            [(j, i) for i in range(0, self.width) for j in range(0, self.height)], split=0
        )
        self.distances = self.precompute_distances()

    def fit(self, X):
        self.learning_rate = self.initial_learning_rate
        batch_count = int(X.gshape[0] / self.batch_size)
        for epoch in range(1, self.max_epoch):
            offset = 0
            for count in range(1, batch_count + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    batch = X[offset : count * self.batch_size]
                batch.balance_()

                distances = ht.spatial.cdist(batch, self.network)
                winner_indices = ht.argmin(distances, axis=1)

                self.update_weights(winner_indices, batch)
                offset = count * self.batch_size

                self.update_learning_rate(epoch)
                self.update_neighbourhood_radius(epoch)

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

    def update_neighbourhood_radius(self, epoch):
        self.radius = self.target_radius + (self.initial_radius - self.target_radius) * np.exp(
            -epoch * (1 / self.max_epoch) * np.e ** 2
        )

    def in_radius(self, indices):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            dist = self.distances[indices]
        dist.balance_()
        dist = ht.reshape(dist, (dist.shape[1], dist.shape[0]))
        dist = ht.exp((ht.pow(dist, 2) * -1) / self.radius)
        return dist

    def precompute_distances(self,):
        return ht.spatial.cdist(self.network_indices, self.network_indices)

    def update_weights(self, indices, weights):
        scalars = self.learning_rate * self.in_radius(indices)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            network = self.network[indices]
        network.balance_()
        distances = weights - network
        scaled_distances = ht.matmul(scalars, distances)
        print(ht.min(scaled_distances, axis=1))
        self.network = self.network + scaled_distances
