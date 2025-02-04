"""Implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
from heat.core.dndarray import DNDarray


class LOF:
    """
    Implementation of the Local Outlier Factor (LOF) algorithm based on [1].
    """

    def __init__(
        self,
        n_neighbors=20,
        metric="euclidean",
    ):
        """
        Initialize the LOF model.

        Parameters
        ----------
        n_neighbors : int, optional (default=20)
            Number of neighbors used to calculate the density of points in the lof algorithm.
        metric : str, optional (default="euclidean")
            The distance metric to use for the tree.

        Raises
        ------
        ValueError
            If ``n_neighbors`` is in a non-suitable range for the lof.

        References
        ----------
        [1] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        # input sanitation
        if n_neighbors < 1:
            raise ValueError(
                "The parameter n_neighbors must be at least 1, but {self.n_neighbors} was inserted."
            )

    def _binary_classifier(lof: DNDarray, method="threshold", **kwargs):
        """
        Binary classification of the data points as outliers or inliers based on their non-binary lof. According to the method,
        the data points are classified as outliers if their lof is greater or equal to a specified threshold or if they have one
        of the topN largest lof scores.

        lof : float
            local outlier factor (non-binary) of the data points
        method : string
            defines which classification method should be used:
                - "threshold": everything greater or equal then specified threshold is considered as an outlier
                - "topN": the data points with the ``topN`` largest outlier scores as outliers
                Note that parameters for the methods use default values 1.5 and 10, respectively.

        Returns
        -------
        anomaly : DNDarray
            array with outlier classifiaction (1 -> outlier, -1 -> inlier)

        Raises
        ------
        ValueError
            If ``method`` is not "threshold" or "topN".
        """
        if method == "threshold":
            if "threshold" in kwargs:
                threshold = kwargs["threshold"]
            else:
                threshold = 1.5
        elif method == "topN":
            if "top_n" in kwargs:
                top_n = kwargs["top_n"]
            else:
                top_n = 10
            threshold = ht.sort(lof)[0][-top_n]
        anomaly = ht.where(lof >= threshold, 1, -1)
        return anomaly

    def fit(self, X: DNDarray):
        """
        Fit the model using X as training data.

        Parameters
        ----------
        X : DNDarray
            Training data.
        """
        self._fit_X = X
        # input sanitation
        if self.n_neighbors > X.shape[0]:
            self.n_neighbors = X.shape[0]
        # Implement fitting logic here

    def _k_distance(self, X: DNDarray):
        """
        Compute the k-distance for each point in X.

        Parameters
        ----------
        X : DNDarray
            Data points.

        Returns
        -------
        DNDarray
            k-distances for each point.
        """
        # Implement k-distance computation here

    def _reachability_distance(self, X: DNDarray):
        """
        Compute the reachability distance for each point in X.

        Parameters
        ----------
        X : DNDarray
            Data points.

        Returns
        -------
        DNDarray
            Reachability distances for each point.
        """
        # Implement reachability distance computation here

    def _local_reachability_density(self, X: DNDarray):
        """
        Compute the local reachability density for each point in X.

        Parameters
        ----------
        X : DNDarray
            Data points.

        Returns
        -------
        DNDarray
            Local reachability densities for each point.
        """
        # Implement local reachability density computation here

    def _local_outlier_factor(self, X: DNDarray):
        """
        Compute the local outlier factor for each point in X.

        Parameters
        ----------
        X : DNDarray
            Data points.

        Returns
        -------
        DNDarray
            Local outlier factors for each point.
        """
        # Implement local outlier factor computation here

    def predict(self, X: DNDarray):
        """
        Predict the LOF scores for X.

        Parameters
        ----------
        X : DNDarray
            Data points.

        Returns
        -------
        DNDarray
            LOF scores for each point.
        """
        # Implement prediction logic here

    def fit_predict(self, X: DNDarray):
        """
        Fit the model using X as training data and return LOF scores.

        Parameters
        ----------
        X : DNDarray
            Training data.

        Returns
        -------
        DNDarray
            LOF scores for each point.
        """
        self.fit(X)
        return self.predict(X)
