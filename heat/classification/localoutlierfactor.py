"""Implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
import torch
import warnings
from heat.core import types
from mpi4py import MPI
from heat.core.dndarray import DNDarray
from heat.spatial.distance import cdist, cdist_small, _euclidian, _manhattan, _gaussian

__all__ = ["LocalOutlierFactor"]


class LocalOutlierFactor:
    """
    Class for the Local Outlier Factor (LOF) algorithm. The LOF algorithm is a density-based outlier detection method.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors used to calculate the density of points in the lof algorithm. Denoted as MinPts in [1].
    metric : str, optional (default=_euclidian)
        The distance metric to use for the tree.
    binary_decision : string, optional
        Defines which classification method should be used:
        - "threshold": everything greater or equal to the specified threshold is considered an outlier.
        - "top_n": the data points with the ``top_n`` largest outlier scores are considered outliers.
        Default is "threshold".
    threshold : float, optional
        The threshold value for the "threshold" method. Default is 1.5.
    top_n : int, optional
        The number of top outliers for the "top_n" method. Default is 10.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors used to calculate the density of points in the lof algorithm. Denoted as MinPts in [1].
    binary_decision: string
        Method that converts lof score into a binary decision of outlier and non-outlier. Can be "threshold" or "top_n".
    metric : str
        The measure of the distance. Can be "euclidian", "manhattan", or "gaussian".
    threshold : float
        The threshold value for the "threshold" method used for binary classification.
    top_n : int
        The number of top outliers for the "top_n" method used for binary classification.
    lof_scores : DNDarray
        The local outlier factor for each sample in the data set.
    anomaly : DNDarray
        Array with binary outlier classification (1 -> outlier, -1 -> inlier).
    chunks : int
        Compute the distance matrix iteratively in chunks to reduce memory consumption (but with larger runtime).
        For ``chunks``= 2: first compute one half of the distance matrix and then the second half.
    fully_distributed : bool
        Decides whether to distribute auxiliary vectors during the computation among all MPI processes.
        Only set to True for a very large number of data points that may already cause memory issues on their own.
        True is more memory efficient, but much slower than False due to large communication overhead.
    idx_n_neighbors : DNDarray
        Indices of nearest neighbors for each sample in the data set.

    Raises
    ------
    ValueError
        If ``binary_decision`` is not "threshold" or "top_n".
        If ``metric`` is neither "euclidian", "manhattan", nor "gaussian".

    Warnings
    --------
        If ``n_neighbors`` is in a non-suitable range for the lof.

    References
    ----------
    [1] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers.
    """

    def __init__(
        self,
        n_neighbors=20,
        metric="euclidian",
        binary_decision="threshold",
        threshold=1.5,
        chunks=1,
        top_n=None,
        fully_distributed=False,
    ):
        self.n_neighbors = n_neighbors
        self.binary_decision = binary_decision
        self.threshold = threshold
        self.top_n = top_n
        self.lof_scores = None
        self.anomaly = None
        self.metric = metric
        self.chunks = chunks
        self.fully_distributed = fully_distributed
        self.idx_n_neighbors = None

        self._input_sanitation()

    def fit(self, X: DNDarray):
        """
        Fit the LOF model to the data.

        Parameters
        ----------
        X : DNDarray
            Data points.
        """
        # Compute the LOF for each sample in X
        self._local_outlier_factor(X)
        # Classifying the data points as outliers or inliers
        self._binary_classifier()

    def _local_outlier_factor(self, X: DNDarray):
        """
        Compute the LOF for each sample in X.

        Parameters
        ----------
        X : DNDarray
            Data points.
        """
        # number of data points
        length = X.shape[0]

        # input sanitation
        # If n_neighbors is larger than or equal the number of samples, continue with the whole sample when evaluating the LOF
        if self.n_neighbors >= length:
            self.n_neighbors = length - 1  # length of data is n_neighbors + the point itself
        # [1] suggests a minimum of 10 neighbors
        if length <= 10:
            raise ValueError(
                f"The data set is too small for a reasonable LOF evaluation. The number of samples should be larger than 10, but was {X.shape[0]}."
            )

        # Compute the distance matrix for the n_neighbors nearest neighbors of each point and the corresponding indices
        # (only these are needed for the LOF computation).
        # Note that cdist_small sorts from the lowest to the highest distance
        dist, idx = cdist_small(
            X, X, metric=self.metric, n_smallest=self.n_neighbors + 1, chunks=self.chunks
        )  # cdist_small stores also the distance of each point to itself, therefore use n_neighbors+1

        # Extract the k-distance and the indices of the k-nearest neighbors
        k_dist = dist[:, -1]
        idx_neighbors = idx[:, 1 : self.n_neighbors + 1]
        # Make the indices of the n-nearest neighbors available for a use outside this function
        self.idx_n_neighbors = idx_neighbors

        k_dist_neighbors = self._advanced_indexing(k_dist, idx_neighbors)

        # Compute the reachability distance for each point
        reachability_dist = ht.maximum(k_dist_neighbors, dist[:, 1 : self.n_neighbors + 1])

        # Compute the local reachability density (lrd) for each point
        lrd = 1 / (
            ht.mean(reachability_dist, axis=1) + 1e-10
        )  # add 1e-10 to avoid division by zero (important for many duplicates in data)

        # Calculate the local reachability distance for each point's neighbors
        lrd_neighbors = self._advanced_indexing(lrd, idx[:, 1 : self.n_neighbors + 1])

        lof = ht.mean(lrd_neighbors, axis=1) / lrd

        self.lof_scores = lof

    def _binary_classifier(self):
        """
        Binary classification of the data points as outliers or inliers based on their non-binary LOF. According to the method,
        the data points are classified as outliers if their LOF is greater or equal to a specified threshold or if they have one
        of the top_n largest LOF scores.

        Returns
        -------
        anomaly : DNDarray
            Array with outlier classification (1 -> outlier, -1 -> inlier).

        Raises
        ------
        ValueError
            If ``method`` is not "threshold" or "top_n".
        """
        if self.binary_decision == "threshold":
            # Use the provided threshold value
            threshold_value = self.threshold
        elif self.binary_decision == "top_n":
            # Determine the threshold based on the top_n largest LOF scores
            threshold_value = ht.topk(self.lof_scores, k=self.top_n, sorted=True, largest=True)[0][
                -1
            ]
        else:
            raise ValueError(
                f"Unknown method for binary decision: {self.binary_decision}. Use 'threshold' or 'top_n'."
            )

        # Classify anomalies based on the threshold value
        self.anomaly = ht.where(self.lof_scores >= threshold_value, 1, -1)

    def _advanced_indexing(self, A: DNDarray, idx: DNDarray) -> DNDarray:
        """
        Perform advanced indexing on a distributed DNDarray, allowing for optional runtime optimization.

        This function handles advanced indexing for distributed DNDarrays. It supports two modes:
        1. Fully distributed mode (`fully_distributed=True`): handles indexing in a completely distributed manner.
           This mode is memory safe but rather slow.
        2. Local mode (`fully_distributed=False`):uses local arrays (torch tensors) to perform indexing
           efficiently, assuming that local arrays of dimension (A.shape[0], `n_neighbors`) fit into memory.

        Parameters
        ----------
        A : DNDarray
            The input DNDarray to be indexed.
        idx : DNDarray
            The indices used for advanced indexing.

        Returns
        -------
        indexed_A : DNDarray
            The result of advanced indexing on the input array.
        """
        # Using heat's advanced indexing for large data set
        if self.fully_distributed is True:
            # TODO: currently, the required advanced indexing only works if k_dist=k_dist.resplit_(None).
            # Once the advanced indexing is implemented for all configurations, replace the following loop
            # by indexed_A=A[idx]
            indexed_A = ht.zeros(idx.shape, split=0)
            for i in range(A.shape[0]):
                indexed_A[i] = A[idx[i]]
        # Use local arrays, i.e., torch.tensors, to reduce runtime while indexing
        # (only possible if all local arrays defined below fit into memory)
        else:
            split = A.split
            type = A.dtype
            # Use none-split arrays to reduce communication overhead
            A_ = A.resplit_(None).larray.contiguous()
            idx_ = idx.resplit_(None).larray.contiguous()
            # Apply standard advanced indexing
            indexed_A_ = A_[idx_]
            # Convert the result back to a distributed DNDarray
            indexed_A = ht.array(indexed_A_, split=split, dtype=type)
        return indexed_A

    def _map_idx_to_proc(self, idx, comm):
        """
        Auxiliary function to map indices to the corresponding MPI process ranks.

        This function takes an array of indices and determines which MPI process
        each index belongs to, based on the distribution of data across processes.
        It returns an array where each index is replaced by the rank of the process
        that contains the corresponding data.

        Parameters
        ----------
        idx : DNDarray
            The array of indices to be mapped to MPI process ranks. The array should
            be distributed along the first axis (split=0).
        comm: MPI.COMM_WORLD
            The MPI communicator.

        Returns
        -------
        mapped_idx : DNDarray
            An array of the same shape as `idx`, where each index is replaced by the
            rank of the MPI process that contains the corresponding data.
        """
        size = comm.Get_size()
        _, displ, _ = comm.counts_displs_shape(idx.shape, idx.split)
        mapped_idx = ht.zeros_like(idx)
        for rank in range(size):
            lower_bound = displ[rank]
            if rank == size - 1:  # size-1 is the last rank
                upper_bound = idx.shape[0]
            else:
                upper_bound = displ[rank + 1]
            mask = (idx >= lower_bound) & (idx < upper_bound)
            mapped_idx[mask] = rank
        return mapped_idx

    def _input_sanitation(self):
        """
        Check if the input parameters are valid and raise warnings or exceptions.
        """
        # check number of neighbors, [1] suggests n_neighbors >= 10
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be great one. but was {self.n_neighbors}.")
        if self.n_neighbors < 10 and self.n_neighbors > 100:
            warnings.warn(
                f"For reasonable results n_neighbors is expected between 10 and 100, but was {self.n_neighbors}.",
                UserWarning,
            )

        # check for correctly binary decision method
        if self.binary_decision not in ["threshold", "top_n"]:
            raise ValueError(
                f"Unknown method for binary decision: {self.binary_decision}. Use 'threshold' or 'top_n'."
            )

        # check if the top_n parameter is specified when using the top_n method
        if self.binary_decision == "top_n":
            if self.top_n is None:
                raise ValueError(
                    "For binary decision='top_n', the parameter 'top_n' has to be specified."
                )
            elif self.top_n < 1:
                raise ValueError("The number of top outliers should be greater than one.")
            if self.threshold != 1.5:
                warnings.warn(
                    "You are specifying the parameter threshold, although binary_decision is set to 'top_n'. The threshold will be ignored.",
                    UserWarning,
                )

        if self.binary_decision == "threshold":
            if self.threshold <= 1 or self.threshold is None:
                raise ValueError("The threshold should be greater than one.")
            if self.top_n is not None:
                warnings.warn(
                    "You are specifying the parameter top_n, although binary_decision is set to 'threshold'. The value of top_n will be ignored.",
                    UserWarning,
                )

        # check for valid metric
        valid_metrics = ["euclidian", "gaussian", "manhattan"]
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{self.metric}'. Must be one of {valid_metrics}.")

        # replace the name of the metric with the corresponding function
        if self.metric == "gaussian":
            self.metric = _gaussian
        elif self.metric == "manhattan":
            self.metric = _manhattan
        elif self.metric == "euclidian":
            self.metric = _euclidian
