import heat as ht


class KNN(ht.ClassificationMixin, ht.BaseEstimator):
    """
    HeAT implementation of the K-Nearest-Neighbours Algorithm [1].

    This algorithm predicts labels to data vectors by using an already labeled training dataset as reference.
    The input vector to be predicted is compared to the training vectors by calculating the distance.
    We use the euclidean distance, this can be expanded to other distance functions as well.
    Then a majority vote of the k nearest (smallest distances) training vectors labels is chosen as prediction.

    Parameters
    ----------
    x : ht.DNDarray
        Array of shape (n_samples, sample_length,), required
        The training vectors
    y : ht.DNDarray
        Array of shape (n_samples, n_features), required
        Labels for the training set, bool value for each class in n_features
    num_neighbours: int
        Number of neighbours to consider when choosing label

    References
    --------
    [1] T. Cover and P. Hart, "Nearest neighbor pattern classification," in IEEE Transactions on Information Theory,
        vol. 13, no. 1, pp. 21-27, January 1967, doi: 10.1109/TIT.1967.1053964.
    """

    def __init__(self, x, y, num_neighbours):
        self.x = x
        self.y = y
        self.num_neighbours = num_neighbours

    def fit(self, X, Y):
        """
        Parameters
        ----------
        X : ht.DNDarray
            Data vectors used for prediction
        Y : ht.DNDarray
            Labels for the data, shape (n_samples, n_features)
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "Number of samples and labels needs to be the same, got {}, {}".format(
                    X.shape[0], Y.shape[0]
                )
            )

        self.x = X
        self.y = Y

    def predict(self, X) -> ht.dndarray:
        """
        Parameters
        ----------
        X : ht.DNDarray
            Input data to be predicted
        """

        distances = ht.spatial.cdist(X, self.x)
        _, indices = ht.topk(distances, self.num_neighbours, largest=False)

        labels = self.y[indices.flatten()]
        labels.balance_()
        labels = ht.reshape(labels, (indices.gshape + (self.y.gshape[1],)))

        labels = ht.sum(labels, axis=1)
        maximums = ht.max(labels, axis=1, keepdim=True)

        return ht.where(labels == maximums, 1, 0)
