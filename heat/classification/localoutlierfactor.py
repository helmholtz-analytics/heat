"""Implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
from heat.core.dndarray import DNDarray
from heat.spatial.distance import cdist_small, _euclidian, _manhattan, _gaussian


class LOF:
    """
    Implementation of the Local Outlier Factor (LOF) algorithm based on [1].
    """

    def __init__(
        self,
        n_neighbors=20,
        metric="euclidian",
    ):
        """
        Initialize the LOF model.

        Parameters
        ----------
        n_neighbors : int, optional (default=20)
            Number of neighbors used to calculate the density of points in the lof algorithm. Denoted as MinPts in [1].
        metric : str, optional (default=_euclidian)
            The distance metric to use for the tree.

        Raises
        ------
        ValueError
            If ``n_neighbors`` is in a non-suitable range for the lof.

        References
        ----------
        [1] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.
        """
        # input sanitation
        if n_neighbors < 10:  # [1] suggests a minimum of 10 neighbors
            raise ValueError(
                "The parameter n_neighbors must be at least 10, but {self.n_neighbors} was inserted."
            )
        if metric == "gaussian":
            self.metric = _gaussian
        elif metric == "manhattan":
            self.metric = _manhattan
        elif metric == "euclidian":
            self.metric = _euclidian
        else:
            valid_metrics = ["euclidian", "gaussian", "manhattan"]
            raise ValueError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}.")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.lof_scores = None

    def fit_predict(self, X: DNDarray):
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

        Returns
        -------
        DNDarray
            LOF scores for each point.
        """
        # Implement prediction logic here

    def fit(self, X: DNDarray):
        """
        Compute the LOF for each sample in X.

        Parameters
        ----------
        X : DNDarray
            Data points.
        """
        # input sanitation
        # If n_neighbors is larger than or equal the number of samples, continue with the whole sample when evaluating the LOF
        if self.n_neighbors >= X.shape[0]:
            self.n_neighbors = X.shape[0] - 1  # n_neighbors + the point itself = X.shape[0]
        if X.shape[0] < 10:  # [1] suggests a minimum of 10 neighbors
            raise ValueError(
                f"The data set is too small for a reasonable LOF evaluation. The number of samples must be at least 10, but was {X.shape[0]}."
            )
        # Compute the distance matrix for the n_neighbors nearest neighbors of each point and the corresponding indices
        # (only these are needed for the LOF computation).
        # Note that cdist_small sorts from the lowest to the highest distance
        dist, idx = cdist_small(
            X, X, metric=self.metric, n_smallest=self.n_neighbors + 1
        )  # cdist_small stores also the distance of each point to itself, therefore use n_neighbors+1

        # Compute the k-distance for each point
        k_dist = dist[:, -1]  # k-distance = largest value in dist for each row
        idx_k_dist = idx[:, -1]  # indices corresponding to k_dist

        # Compute the reachability distance for each point by comparing the k-distance of the neighbors with the distance to the neighbors
        # Note:
        # - this implementation is simplified by assuming that k_dist fits into the memory of each process
        # - only the maximal values of dist are necessary to compute the reachability distance
        # ensure correct indexing across processes for later comparison with k_dist
        largest_dist_neighbor_unsplit = k_dist.resplit_(
            None
        )  # only the maximal values of dist are needed, thus use k_dist instead of dist
        largest_dist = largest_dist_neighbor_unsplit[idx_k_dist]
        largest_dist = largest_dist.resplit_(0)
        # evaluate reachability distance
        reachability_dist = ht.maximum(
            k_dist, largest_dist[idx_k_dist]
        )  # the second arguemt k_dist directly takes the largest distance of each row

        # Compute the local reachability density (lrd) for each point
        lrd = self.n_neighbors / (
            ht.sum(reachability_dist, axis=1) + 1e-10
        )  # add 1e-10 to avoid division by zero
        lrd_neighbors = lrd[idx[:, 1:]]

        # Compute the local outlier factor for each point
        lof = ht.sum(lrd_neighbors, axis=1) / (self.n_neighbors * lrd + 1e-10)

        # Store the LOF scores in the class object
        self.lof_scores = lof

    def _binary_classifier(self, method="threshold", **kwargs):
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
            threshold = ht.sort(self.lof_scores)[0][-top_n]
        anomaly = ht.where(self.lof_scores >= threshold, 1, -1)
        return anomaly

    def _local_outlier_factor(self, X: DNDarray):
        """
        Compute the local outlier factor for sample in X.

        Parameters
        ----------
        X : DNDarray
            Data points.

        Returns
        -------
        lof : DNDarray
            Local outlier factors for each point.
        idx : DNDarray
            Indices of the
        """
        # Implement local outlier factor computation here

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
