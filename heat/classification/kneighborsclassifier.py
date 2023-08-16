"""Implements the k-nearest neighbors (kNN) classifier"""
from typing import Callable

import heat as ht

from heat.core.dndarray import DNDarray


class KNeighborsClassifier(ht.BaseEstimator, ht.ClassificationMixin):
    """
    Implementation of the k-nearest-neighbors Algorithm [1].

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
    --------
    [1] T. Cover and P. Hart, "Nearest Neighbor Pattern Classification," in IEEE Transactions on Information Theory,
    vol. 13, no. 1, pp. 21-27, January 1967, doi: 10.1109/TIT.1967.1053964.
    """

    def __init__(self, n_neighbors: int = 5, effective_metric_: Callable = None):
        self.n_neighbors = n_neighbors
        self.effective_metric_ = (
            effective_metric_ if effective_metric_ is not None else ht.spatial.cdist
        )

        # init declaration to appease flake
        self.x = None
        self.y = None
        self.n_samples_fit_ = -1
        self.outputs_2d_ = True
        self.classes_ = None

    @staticmethod
    def one_hot_encoding(x: DNDarray) -> DNDarray:
        """
        One-hot-encodes the passed vector or single-column matrix.

        Parameters
        ----------
        x : DNDarray
            The data to be encoded.
        """
        n_samples = x.shape[0]
        n_features = ht.max(x).item() + 1

        one_hot = ht.zeros((n_samples, n_features), split=x.split, device=x.device, comm=x.comm)
        one_hot.lloc[range(one_hot.lshape[0]), x.larray] = 1

        return one_hot

    def fit(self, x: DNDarray, y: DNDarray):
        """
        Fit the k-nearest neighbors classifier from the training dataset.

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
        """
        # check for type consistency
        if not isinstance(x, DNDarray) or not isinstance(y, DNDarray):
            raise TypeError(f"x and y must be DNDarrays but were {type(x)} {type(y)}")

        # ensure that x is a two-dimensional matrix
        if len(x.shape) != 2:
            raise ValueError(f"x must be two-dimensional, but was {len(x.shape)}")
        self.x = x
        self.n_samples_fit_ = x.shape[0]

        # ensure that x and y have the same number of samples
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples x and y samples mismatch, got {x.shape[0]}, {y.shape[0]}"
            )

        # checks the labels for correct dimensionality and encode one-hot
        if len(y.shape) == 1:
            self.y = self.one_hot_encoding(y)
            self.outputs_2d_ = False
        elif len(y.shape) == 2:
            self.y = y
            self.outputs_2d_ = True
        else:
            raise ValueError(f"y needs to be one- or two-dimensional, but was {len(y.shape)}")

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        x : DNDarray
            The test samples.
        """
        distances = self.effective_metric_(x, self.x)
        _, indices = ht.topk(distances, self.n_neighbors, largest=False)

        predictions = self.y[indices.flatten()]
        predictions.balance_()
        predictions = ht.reshape(predictions, (indices.gshape + (self.y.gshape[1],)))
        predictions = ht.sum(predictions, axis=1)

        self.classes_ = ht.argmax(predictions, axis=1)

        return self.classes_
