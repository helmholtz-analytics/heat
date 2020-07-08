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
        batch_size=1,
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
        self.network = ht.random.randn(height * width, data_dim, dtype=ht.float64)
        self.batch_size = batch_size

        self.network_indices = ht.array(
            [(j, i) for i in range(0, self.width) for j in range(0, self.height)]
        )
        self.distances = self.precompute_distances()

    def fit(self, X):
        self.learning_rate = self.initial_learning_rate
        batch_count = int(X.shape[0] / self.batch_size)
        # print(self.network)
        for epoch in range(1, self.max_epoch + 1):
            offset = 0
            for count in range(1, batch_count + 1):
                batch = ht.array(X[offset : count * self.batch_size], is_split=X.split)
                distances = ht.spatial.cdist(batch, self.network)
                _, winner_indices = ht.topk(distances, 1, largest=False, dim=1)
                self.update_weights(winner_indices, batch)
                offset = count * self.batch_size
                self.update_learning_rate(epoch)
                self.update_neighbourhood_radius(epoch)
            # print(epoch, self.network)

    def predict(self, X):
        distances = ht.spatial.cdist(X, self.network)
        _, winner_ind = ht.topk(distances, 1, largest=False, dim=1)
        translated_winner_ind = self.network_indices[winner_ind.tolist()]

        return translated_winner_ind

    def update_learning_rate(self, t):
        self.learning_rate = self.initial_learning_rate + (
            self.target_learning_rate - self.initial_learning_rate
        ) * (t / self.max_epoch)

    def update_neighbourhood_radius(self, t):
        self.radius = self.initial_radius * ht.pow(
            (self.target_radius / self.initial_radius), t / self.max_epoch
        )

    def distance_weight(self, winner_ind):
        return ht.where(
            self.distances[winner_ind].flatten() < self.radius,
            ht.ones((self.height * self.width,)),
            ht.zeros((self.height * self.width,)),
        )

    def precompute_distances(self,):
        return ht.spatial.cdist(self.network_indices, self.network_indices)

    def update_weights(self, indices, batch):
        for i, (winner_ind, weight) in enumerate(zip(indices, batch)):
            scalar = self.learning_rate * self.distance_weight(winner_ind)
            scalar = ht.expand_dims(scalar, axis=1)
            weight = ht.expand_dims(weight, axis=0)
            if i == 0:
                print(self.network.dtype, weight.dtype, scalar.dtype)
                print("prod", scalar * self.network - weight)
                print("sum", self.network + scalar * self.network - weight)
            self.network = self.network + scalar * self.network - weight
