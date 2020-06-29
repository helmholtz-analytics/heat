import heat as ht


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
        self.network = ht.random.randn(height * width, data_dim)

        self.network_indices = ht.array(
            [(j, i) for i in range(0, self.width) for j in range(0, self.height)]
        )
        self.distances = self.precompute_distances()

    def fit(self, X):
        self.learning_rate = self.initial_learning_rate
        for epoch in range(1, self.max_epoch + 1):
            distances = ht.spatial.cdist(X, self.network)
            _, winner_indices = ht.topk(distances, 1, largest=False, dim=1)
            for ind in winner_indices:
                distances, indices = self.find_neighbours(ind)
                self.update_weights(distances, indices)

            self.update_learning_rate(epoch)
            self.update_neighbourhood_radius(epoch)

    def predict(self, X):
        distances = ht.spatial.cdist(X, self.network)
        _, winner_ind = ht.topk(distances, 1, largest=False, dim=1)
        translated_winner_ind = self.indices[winner_ind.tolist()]

        return translated_winner_ind

    def update_learning_rate(self, t):
        self.learning_rate = self.initial_learning_rate + (
            self.target_learning_rate - self.initial_learning_rate
        ) * (t / self.max_epoch)

    def update_neighbourhood_radius(self, t):
        self.radius = self.initial_radius * ht.pow(
            (self.target_radius / self.initial_radius), t / self.max_epoch
        )

    def find_neighbours(self, ind):
        neighbour_indices = ht.nonzero(self.distances[ind.tolist()] < self.radius)
        elements = self.network[neighbour_indices.tolist()]
        return elements, neighbour_indices

    def precompute_distances(self,):
        return ht.spatial.cdist(self.network_indices, self.network_indices)

    def update_weights(self, distances, indices):
        print(distances)
        print(indices)
        self.network[indices.tolist()] = self.network[indices.tolist()] + (
            self.learning_rate * ht.exp([-ht.pow(dist, 2) / 2 * self.radius for dist in distances])
        )
