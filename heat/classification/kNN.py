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

        # Maybe improved later when torch.topk is implemented in HeAT
        distances, indices = ht.sort(distances, axis=1)
        indices = indices[:, : self.num_neighbours]
        indices = [[ind.item() for ind in ind_list] for ind_list in indices]
        labels = self.y[indices]
        unique = ht.unique(labels)

        label_count = ht.empty((labels.shape[0], unique.shape[0]))
        for index, unique_label in enumerate(unique):
            equals = ht.eq(labels, unique_label)
            label_count[:, index] = ht.sum(equals, axis=1)

        return ht.argmax(label_count, axis=1)


