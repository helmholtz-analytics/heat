import heat as ht


class KNN(ht.ClassificationMixin, ht.BaseEstimator):
    """
    HeAT implementation of the K-Nearest-Neighbours Algorithm

    Parameters
    ----------
    x : ht.DNDarray
        array of shape (n_samples,), required
        The Training samples
    y : ht.DNDarray
        array of shape (n_samples,), required
        Labels for the training set
    num_neighbours: int, required
        number of neighbours to consider when choosing label
    --------
    """

    def __init__(self, x, y, num_neighbours):
        self.x = x
        self.y = y
        self.num_neighbours = num_neighbours

    def fit(self, X, Y):
        self.x = X
        self.y = Y

    def predict(self, X):
        """
        Parameters
        ----------
        X : ht.DNDarray
            Input data to be predicted
        Returns
        -------
        labels : ht.DNDarray
            The predicted classes
        """

        distances = ht.spatial.cdist(X, self.x)
        _ , indices = ht.topk(distances, self.num_neighbours, largest=False)

        labels = self.y[indices._DNDarray__array]
        uniques = ht.unique(labels, sorted=True)
        uniques = ht.resplit(ht.expand_dims(uniques, axis=0), axis=0)
        labels = ht.expand_dims(labels, axis=2)

        return ht.argmax(ht.sum(labels == uniques, axis=1), axis=1)

