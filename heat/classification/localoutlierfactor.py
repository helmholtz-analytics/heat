"""Implementation of the Local Outlier Factor (LOF) algorithm"""

import warnings
import heat as ht
import torch
from heat.core.dndarray import DNDarray
from heat.spatial.distance import _euclidian, _gaussian, _manhattan, cdist_small

__all__ = ["LocalOutlierFactor"]


class LocalOutlierFactor:
    """
    Class for the Local Outlier Factor (LOF) algorithm. The LOF algorithm is a density-based outlier detection method.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors used to calculate the density of points in the LOF algorithm. Denoted as MinPts in [1].
    metric : str, optional (default="euclidian")
        The distance metric to use. Can be "euclidian", "manhattan", or "gaussian".
    binary_decision : string, optional
        Defines which classification method should be used:
        - "threshold": everything greater or equal to the specified threshold is considered an outlier.
        - "top_n": the data points with the ``top_n`` largest outlier scores are considered outliers.
        Default is "threshold".
    threshold : float, optional
        The threshold value for the "threshold" method. Default is 1.5.
    top_n : int, optional
        The number of top outliers for the "top_n" method. Default is None.
    chunks : int, optional (default=1)
        Compute the distance matrix iteratively in chunks to reduce memory consumption (but with larger runtime).
        For ``chunks``= 2: first compute one half of the distance matrix and then the second half.
    fully_distributed : bool, optional (default=False)
        Decides whether to distribute auxiliary vectors during the computation among all MPI processes.
        Only set to True for a very large number of data points that may already cause memory issues on their own.
        True is more memory efficient, but much slower than False due to large communication overhead.

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
            self.n_neighbors_ = length - 1  # length of data is n_neighbors + the point itself
        else:
            self.n_neighbors_ = self.n_neighbors

        # [1] suggests a minimum of 10 neighbors
        if length <= 10:
            raise ValueError(
                f"The data set is too small for a reasonable LOF evaluation. The number of samples should be larger than 10, but was {X.shape[0]}."
            )

        # Compute the distance matrix for the n_neighbors nearest neighbors of each point and the corresponding indices
        # (only these are needed for the LOF computation).
        dist, idx = cdist_small(
            X, X, metric=self._metric_fn, n_smallest=self.n_neighbors_ + 1, chunks=self.chunks
        )

        # Local views on axis 1 (column 0 is self-distance, 1: are the neighbors)
        dist_neighbors_l = dist.larray[:, 1:]
        k_dist_l = dist_neighbors_l[:, -1]

        # Direct instantiation with known metadata
        k_dist = DNDarray(
            k_dist_l,
            gshape=(length,),
            dtype=dist.dtype,
            split=dist.split,
            device=dist.device,
            comm=dist.comm,
            balanced=dist.balanced,
        )
        idx_neighbors = DNDarray(
            idx.larray[:, 1:],
            gshape=(length, self.n_neighbors_),
            dtype=idx.dtype,
            split=idx.split,
            device=idx.device,
            comm=idx.comm,
            balanced=idx.balanced,
        )
        self.idx_n_neighbors = idx_neighbors

        k_dist_neighbors = self._advanced_indexing(k_dist, idx_neighbors)

        # calculate reachability distance on local tensors
        reachability_dist_l = torch.maximum(k_dist_neighbors.larray, dist_neighbors_l)

        # calculate local reachability density (lrd)
        lrd_l = 1.0 / (reachability_dist_l.mean(dim=1) + 1e-10)

        # wrap lrd with known metadata to pass to indexing without collective calls
        lrd = DNDarray(
            lrd_l,
            gshape=(length,),
            dtype=dist.dtype,
            split=dist.split,
            device=dist.device,
            comm=dist.comm,
            balanced=dist.balanced,
        )

        # calculate neighbor density
        lrd_neighbors = self._advanced_indexing(lrd, idx_neighbors)

        # compute final LOF scores locally on device
        lof_l = lrd_neighbors.larray.mean(dim=1) / lrd_l

        self.lof_scores = DNDarray(
            lof_l,
            gshape=(length,),
            dtype=dist.dtype,
            split=dist.split,
            device=dist.device,
            comm=dist.comm,
            balanced=dist.balanced,
        )

    def _binary_classifier(self):
        """
        Binary classification of the data points as outliers (1) or inliers (-1) based on their non-binary LOF. According to the method,
        the data points are classified as outliers if their LOF is greater or equal to a specified threshold or if they have one
        of the top_n largest LOF scores.
        """
        if self.binary_decision == "top_n":
            # Determine the threshold based on the top_n largest LOF scores
            top_k_val = ht.topk(self.lof_scores, k=self.top_n, sorted=True, largest=True)[0][-1]
            self.threshold = float(top_k_val.item())
        # Classify anomalies based on the threshold value
        self.anomaly = ht.where(self.lof_scores >= self.threshold, 1, -1)

    def _advanced_indexing(self, A: DNDarray, idx: DNDarray) -> DNDarray:
        """
        Perform advanced indexing of neighbor values on a 1D array using a 2D index matrix.

        Supports two execution modes controlled by ``fully_distributed``:
        1. Fully distributed mode (``fully_distributed=True``): executes native Heat distributed
           indexing (``A[idx]``) via MPI all-to-all communication. This path is memory-safe when
           replicating ``A`` would exceed device memory.
        2. Replicated target mode (``fully_distributed=False``): replicates only the 1D vector ``A``
           across processes while keeping ``idx`` partitioned along axis 0. Indexing is performed
           locally on device tensors without all-to-all communication, and the output ``DNDarray``
           is instantiated directly using known distribution metadata.

        Parameters
        ----------
        A : DNDarray
            1D array of values to be indexed (e.g., ``k_dist`` or ``lrd``).
        idx : DNDarray
            2D array of neighbor indices with shape ``(n_samples, n_neighbors)``, partitioned
            along axis 0.

        Returns
        -------
        indexed_A : DNDarray
            2D array with indexed neighbor values, having shape ``(n_samples, n_neighbors)``
            and inheriting the partition layout of ``idx``.
        """
        if self.fully_distributed:
            return A[idx]

        # Replicate only 1D vector A; keep idx partitioned
        A_replicated = A.resplit(None) if A.is_distributed() else A
        local_result = A_replicated.larray[idx.larray]

        # Zero collective calls: reuse known metadata from idx
        return DNDarray(
            local_result,
            gshape=idx.shape,
            dtype=A.dtype,
            split=idx.split,
            device=A.device,
            comm=A.comm,
            balanced=idx.balanced,
        )

    def _input_sanitation(self):
        """
        Check if the input parameters are valid and raise warnings or exceptions.
        """
        # check number of neighbors, [1] suggests n_neighbors >= 10
        if self.n_neighbors < 1 or not isinstance(self.n_neighbors, int):
            raise ValueError(
                f"n_neighbors must be an integer greater than or equal to 1, but was {self.n_neighbors}."
            )
        if self.n_neighbors < 10 or self.n_neighbors > 100:
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
            if self.threshold != 1.5:
                warnings.warn(
                    "You are specifying the parameter threshold, although binary_decision is set to 'top_n'. The threshold will be ignored.",
                    UserWarning,
                )
            if self.top_n is None:
                raise ValueError(
                    "For binary decision='top_n', the parameter 'top_n' has to be specified."
                )
            if self.top_n < 1:
                raise ValueError("The number of top outliers should be greater than one.")

        if self.binary_decision == "threshold":
            if self.threshold is None or self.threshold <= 1:
                raise ValueError("The threshold should be greater than one.")
            if self.top_n is not None:
                warnings.warn(
                    "You are specifying the parameter top_n, although binary_decision is set to 'threshold'. The value of top_n will be ignored.",
                    UserWarning,
                )

        # check for valid metric
        metric_map = {
            "euclidian": _euclidian,
            "euclidean": _euclidian,
            "gaussian": _gaussian,
            "manhattan": _manhattan,
        }
        if self.metric not in metric_map:
            raise ValueError(
                f"Invalid metric '{self.metric}'. Must be one of {list(metric_map.keys())}."
            )

        # store the resolved callable separately without overwriting self.metric
        self._metric_fn = metric_map[self.metric]
