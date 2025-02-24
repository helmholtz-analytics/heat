"""Implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht
import torch
from heat.core.dndarray import DNDarray
from heat.spatial.distance import cdist_small, _euclidian, _manhattan, _gaussian

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
        - "topN": the data points with the ``topN`` largest outlier scores are considered outliers.
        Default is "threshold".
    threshold : float, optional
        The threshold value for the "threshold" method. Default is 1.5.
    top_n : int, optional
        The number of top outliers for the "topN" method. Default is 10.
    Attributes
    ----------
    n_neighbors : int
        Number of neighbors used to calculate the density of points in the lof algorithm. Denoted as MinPts in [1].
    metric : str
        The measure of the distance. Can be "euclidian", "manhattan", or "gaussian".
    threshold : float
        The threshold value for the "threshold" method used for binary classification.
    top_n : int
        The number of top outliers for the "topN" method used for binary classification.
    lof_scores : DNDarray
        The local outlier factor for each sample in the data set.
    anomaly : DNDarray
        Array with binary outlier classification (1 -> outlier, -1 -> inlier).
    Raises
    ------
    ValueError
        If ``n_neighbors`` is in a non-suitable range for the lof.
        If ``binary_decision`` is not "threshold" or "topN".
        If ``metric`` is neither "euclidian", "manhattan", nor "gaussian".
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
        top_n=10,
    ):
        # input sanitation
        if n_neighbors < 10 and n_neighbors > 1000:  # [1] suggests a minimum of 10 neighbors
            raise ValueError(
                f"For a reasonable results, the parameter n_neighbors should be between 10 and 1000, but was {self.n_neighbors}."
            )
        if binary_decision not in ["threshold", "topN"]:
            raise ValueError(
                f"Unknown method for binary decision: {self.binary_decision}. Use 'threshold' or 'topN'."
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
        self.threshold = threshold
        self.top_n = top_n
        self.lof_scores = None
        self.anomaly = None

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
        print(
            f"process: {ht.MPI_WORLD.rank}: \n ------------------------------ \n X.larray={X.larray}\n ------------------------------ \n"
        )
        # input sanitation
        # If n_neighbors is larger than or equal the number of samples, continue with the whole sample when evaluating the LOF
        if self.n_neighbors >= X.shape[0]:
            self.n_neighbors = X.shape[0] - 1  # n_neighbors + the point itself = X.shape[0]
        if X.shape[0] <= 10:  # [1] suggests a minimum of 10 neighbors
            raise ValueError(
                f"The data set is too small for a reasonable LOF evaluation. The number of samples should be larger than 10, but was {X.shape[0]}."
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
        # - this implementation is simplified by assuming that k_dist fits in the memory of each process

        comm = dist.comm
        rank = comm.Get_rank()
        _, displ, _ = comm.counts_displs_shape(dist.shape, dist.split)

        # TODO: add a type promotion to float32 or float64
        # promoted_type = types.promote_types(dist.dtype)
        # promoted_type = types.promote_types(promoted_type, types.float32)
        # X = X.astype(promoted_type)
        # Y = Y.astype(promoted_type)
        # if promoted_type == types.float32:
        #     torch_type = torch.float32
        #     # mpi_type = MPI.FLOAT
        # elif promoted_type == types.float64:
        #     torch_type = torch.float64
        #     # mpi_type = MPI.DOUBLE
        # else:
        #     raise NotImplementedError(f"Datatype {X.dtype} currently not supported as input")

        # map the indices in idx_k_dist to the corresponding MPI process that is responsible for
        # the corresponding sample in dist
        mapped_idx = self._map_idx_to_proc(idx_k_dist, comm)

        reachability_dist = ht.zeros_like(dist)
        for i in range(int(idx_k_dist.lshape[0])):
            # evaluate reachability distance for the current process
            if mapped_idx[i] == rank:
                reachability_dist[i] = ht.maximum(k_dist[i, None], dist[idx_k_dist[i], :])
            else:
                receiver = rank
                sender = mapped_idx[i]

                # select the distances to communicate between the processes according to the mapped inidces
                dist_comm = dist[idx_k_dist[i] - displ[sender], :]
                dist_comm = dist_comm.larray
                comm.Isend(dist_comm, dest=receiver, tag=i)

                # set a buffer to store the part of Y that is sent to the next process
                buffer = torch.zeros(
                    (dist_comm.lshape_map[sender, 0], dist_comm.lshape_map[sender, 1]),
                    # dtype=torch_type,
                    device=X.device.torch_device,
                )

                # receive the part of Y to the next process
                comm.Irecv(buffer, source=sender, tag=i)

                reachability_dist[i] = ht.maximum(k_dist[i], buffer[:])

        print(
            f"process: {ht.MPI_WORLD.rank}: \n ------------------------------ \n rd.larray={reachability_dist.larray}\n ------------------------------ \n"
        )
        # Compute the local reachability density (lrd) for each point
        lrd = self.n_neighbors / (
            ht.sum(reachability_dist, axis=1) + 1e-10
        )  # add 1e-10 to avoid division by zero
        lrd_neighbors = lrd[idx[:, 1:]]

        # Compute the local outlier factor for each point
        lof = ht.sum(lrd_neighbors, axis=1) / (self.n_neighbors * lrd + 1e-10)

        # Store the LOF scores in the class attribute
        self.lof_scores = lof

    def _binary_classifier(self):
        """
        Binary classification of the data points as outliers or inliers based on their non-binary LOF. According to the method,
        the data points are classified as outliers if their LOF is greater or equal to a specified threshold or if they have one
        of the topN largest LOF scores.

        Returns
        -------
        anomaly : DNDarray
            Array with outlier classification (1 -> outlier, -1 -> inlier).

        Raises
        ------
        ValueError
            If ``method`` is not "threshold" or "topN".
        """
        if self.binary_decision == "threshold":
            # Use the provided threshold value
            threshold_value = self.threshold
        elif self.binary_decision == "topN":
            # Determine the threshold based on the top_n largest LOF scores
            threshold_value = ht.sort(self.lof_scores)[0][-self.top_n]
        else:
            raise ValueError(
                f"Unknown method for binary decision: {self.binary_decision}. Use 'threshold' or 'topN'."
            )

        # Classify anomalies based on the threshold value
        self.anomaly = ht.where(self.lof_scores >= threshold_value, 1, -1)

    def _map_idx_to_proc(idx, comm):
        """
        Helper function to map indices to the corresponding MPI process ranks.

        This function takes an array of indices and determines which MPI process
        each index belongs to based on the distribution of data across processes.
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
