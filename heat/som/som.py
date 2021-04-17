import warnings

import heat as ht
import numpy as np
import random


class FixedSOM(ht.BaseEstimator, ht.ClusteringMixin):
    """
    HeAT implementation of a self-organizing map using the batch computation described by Kohonen [1].
    This algorithm is reducing n-dimensional data vectors onto a two dimensional fixed grid of size height x width.

    Parameters
    ----------
    height: int
        Height of the fixed grid
    width: int
        Width of the fixed grid
    data_dim: int
        dimension of the data vectors
    initial_learning_rate: float
        How much the network adapts per iteration
    target_learning_rate: float
        How much the network adapts in the last iteration (decreases monotonous from initial_learning_rate)
    max_epoch: int
        Number of learning iterations
    seed: int
        Can be used to seed the random network initialization
    batch_size: int
        Size of batches used for minibatching. Has to be a divisor of the length of the training data.
    References
    --------
    [1] Teuvo Kohonen. „Essentials of the self-organizing map“. In: Neural networks 37 (2013), S. 52–65. doi: 10.1016/j.neunet.2012.09.018
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
    ):
        self.height = height
        self.width = width
        self.data_dim = data_dim
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.initial_radius = initial_radius
        self.radius = initial_radius
        self.target_radius = target_radius
        self.target_learning_rate = target_learning_rate
        self.max_epoch = max_epoch

        ht.core.random.seed(seed)

        self.batch_size = batch_size

        self.network_indices = ht.array(
            [(j, i) for i in range(0, self.width) for j in range(0, self.height)], split=0
        )
        self.distances = self.precompute_distances()

    def fit(self, X):
        """
        Adapts the network with a new dataset.
        The resulting network will be split along the same axis as the input data.

        Parameters
        --------
        X: ht.array
            The samples to be used for learning
        """

        self.radius = self.initial_radius
        self.learning_rate = self.initial_learning_rate

        self.network = ht.random.rand(
            self.height * self.width, self.data_dim, dtype=ht.float64, split=0
        )

        batch_count = int(X.gshape[0] / self.batch_size)
        batches = self.create_batches(batch_count, X)
        del X

        for epoch in range(1, self.max_epoch + 1):
            for count in range(batch_count):
                batch = batches[count]
                batch = ht.resplit(batch, axis=0)
                batch.balance_()

                distances = ht.spatial.cdist(batch, self.network)
                row_min = ht.argmin(distances, axis=1)

                scalars = self.in_radius()
                scalars = scalars[row_min]
                scalars.balance_()

                scaled_weights = ht.matmul(batch.T, scalars)
                scalar_sum = ht.sum(scalars, axis=0, keepdim=True)

                new_network = ht.div(scaled_weights.T, scalar_sum.T)

                self.network = new_network

            self.update_learning_rate(epoch)
            self.update_radius(epoch)

    def predict(self, X):
        """
        Returns the x,y coordinates of the best matching unit in the network.

        Parameters
        --------
        X: ht.array
            The samples to be reduced in dimension
        """
        distances = ht.spatial.cdist(X, self.network)
        winner_ind = ht.argmin(distances, 1)
        winner_ind = ht.resplit(winner_ind, None)
        translated_winner_ind = self.network_indices[winner_ind]
        translated_winner_ind.balance_()
        return translated_winner_ind

    def update_learning_rate(self, epoch):
        """
        Reduces the learning rate for each further epoch/iteration.

        Parameters
        --------
        epoch: int
            The iteration
        """
        self.learning_rate = self.initial_learning_rate + (
            self.target_learning_rate - self.initial_learning_rate
        ) * (epoch / self.max_epoch)

    def update_radius(self, epoch):
        """
        Returns the radius for each epoch/iteration.

        Parameters
        --------
        epoch: int
            The iteration
        """
        self.radius = self.initial_radius * (
            (self.target_radius / self.initial_radius) ** (epoch / self.max_epoch)
        )

    def in_radius(self,):
        """
        Calculates the Neighbourhood adaptaion rate for each cell based on the current radius
        """
        dist = ht.exp((ht.pow(self.distances, 2) * -1) / (2 * ht.pow(self.radius, 2)))
        return dist

    def precompute_distances(self,):
        """
        Utility method to precompute the distances between all the network nodes.
        """
        return ht.spatial.cdist(self.network_indices, self.network_indices)

    def umatrix(self,):
        """
        Returns a umatrix representation of the network.
        Each cell contains the sum of distances to the neighbouring weights.
        Neighbours are cells in the moore neighbourhood 1.

        Returns
        -------
        result: ht.DNDarray
            A DNDarray of shape (height, width) containing the umatrix
        """
        network_distances = ht.spatial.cdist(self.network, self.network)
        radius = ht.where(self.distances != 0, self.distances, 2)
        radius = ht.resplit(radius, 0)
        radius = ht.where(radius < 2, 1, 0)
        selected_distances = network_distances * radius
        sum_distances = ht.sum(selected_distances, axis=1)
        distances = ht.reshape(sum_distances, (self.height, self.width))

        return distances

    def create_batches(self, batch_count, X):
        """
        Utility method to create equally sized, balanced batches
        Parameters
        --------
        X: ht.array
            The Data to be distributed
        """
        if batch_count > 1:
            X = ht.resplit(X, axis=1)
            batches = ht.stack(
                [
                    X[(count - 1) * self.batch_size : count * self.batch_size]
                    for count in range(1, batch_count + 1)
                ],
                axis=0,
            )
            batches.balance_()

        else:
            batches = ht.resplit(ht.expand_dims(X, axis=-1), axis=0)
            batches.balance_()
        return batches
