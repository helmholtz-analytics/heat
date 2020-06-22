import heat as ht


class KNN(ht.ClassificationMixin, ht.BaseEstimator):
    """
    HeAT implementation of the K-Nearest-Neighbours Algorithm.

    This algorithm predicts labels to data vectors by using an already labeled training dataset as reference.
    The input vector to be predicted is compared to the training vectors by calculating the distance.
    We use the euclidean distance, this can be expanded to other distance functions as well.
    Then a majority vote of the k nearest (smallest distances) training vectors labels is chosen as prediction.

    Parameters
    ----------
    x : ht.DNDarray
        array of shape (n_samples,), required
        The Training vectors
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

    def predict(self, X) -> ht.dndarray:
        """
        Parameters
        ----------
        X : ht.DNDarray
            Input data to be predicted
        """
        print(self.x.split, X.split)
        print(self.x.lshape, X.lshape)
        distances = ht.spatial.cdist(X, self.x)
        _, indices = ht.topk(distances, self.num_neighbours, largest=False)

        labels = self.y[indices._DNDarray__array]
        uniques = ht.unique(labels, sorted=True)
        uniques = ht.resplit(ht.expand_dims(uniques, axis=0), axis=0)
        labels = ht.expand_dims(labels, axis=2)

        return ht.argmax(ht.sum(labels == uniques, axis=1), axis=1)
