import heat as ht


class KNN:
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

    def assign_label(self, item):
        """
        Parameters
        ----------
        item : ht.DNDarray
            tensor that is to be identified
        Returns
        -------
        selected_label
            the guessed label
        """

        # Hack to make cdist work (only takes 2D-Input)
        item_list = ht.vstack((item, item))
        distances = ht.spatial.cdist(item_list, self.x)[0]
        distances, indices = ht.sort(distances)
        labels = self.y[[ind.item() for ind in indices][:self.num_neighbours]]
        unique = ht.unique(labels)
        max_count = 0
        selected_label = None

        # TODO probably more efficient with heat.eq
        for unique_label in unique:
            count = 0
            for label in labels:
                if label == unique_label:
                    count += 1
            if count > max_count:
                max_count = count
                selected_label = unique_label

        return selected_label
