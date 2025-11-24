Module heat.classification.kneighborsclassifier
===============================================
Implements the k-nearest neighbors (kNN) classifier

Classes
-------

`KNeighborsClassifier(n_neighbors: int = 5, effective_metric_: Callable = None)`
:   Implementation of the k-nearest-neighbors Algorithm [1].

    This algorithm predicts labels to data vectors by using an labeled training dataset as reference. The input vector
    to be predicted is compared to the training vectors by calculating the Euclidean distance between each of them. A
    majority vote of the k-nearest, i.e. closest or smallest distanced, training vectors labels is selected as
    predicted class.

    Parameters
    ----------
    n_neighbors : int, optional, default: 5
        Number of neighbours to consider when choosing label.
    effective_metric_ : Callable, optional
        The distance function used to identify the nearest neighbors, defaults to the Euclidean distance.

    References
    ----------
    [1] T. Cover and P. Hart, "Nearest Neighbor Pattern Classification," in IEEE Transactions on Information Theory,
    vol. 13, no. 1, pp. 21-27, January 1967, doi: 10.1109/TIT.1967.1053964.

    ### Ancestors (in MRO)

    * heat.core.base.BaseEstimator
    * heat.core.base.ClassificationMixin

    ### Static methods

    `one_hot_encoding(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   One-hot-encodes the passed vector or single-column matrix.

        Parameters
        ----------
        x : DNDarray
            The data to be encoded.

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray)`
    :   Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        x : DNDarray
            Labeled training vectors used for comparison in predictions, Shape=(n_samples, n_features).
        y : DNDarray
            Corresponding labels for the training feature vectors. Must have the same number of samples as ``x``.
            Shape=(n_samples) if integral labels or Shape=(n_samples, n_classes) if one-hot-encoded.

        Raises
        ------
        TypeError
            If ``x`` or ``y`` are not DNDarrays.
        ValueError
            If ``x`` and ``y`` shapes mismatch or are not two-dimensional matrices.

        Examples
        --------
        >>> samples = ht.rand(10, 3)
        >>> knn = KNeighborsClassifier(n_neighbors=1)
        >>> knn.fit(samples)

    `predict(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Predict the class labels for the provided data.

        Parameters
        ----------
        x : DNDarray
            The test samples.
