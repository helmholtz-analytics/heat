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
    fully_distributed : bool, optional
        If False, some auxiliary vectors are not distributed among the MPI processes, but kept as local ones.
        This can reduce communication overhead and thus speed up the computation, but can lead to memory issues,
        depending on the number of samples in the data. Default is True.

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
    fully_distributed : bool
        Decides whether to distribute every part of the computation among all MPI processes.

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
        top_n=None,
        fully_distributed=True,
    ):

        self.n_neighbors = n_neighbors
        self.binary_decision = binary_decision
        self.threshold = threshold
        self.top_n = top_n
        self.lof_scores = None
        self.anomaly = None
        self.metric = metric
        self.fully_distributed = fully_distributed

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
        if X.split == 0:
            # Note that cdist_small sorts from the lowest to the highest distance
            dist, idx = cdist_small(
                X, X, metric=self.metric, n_smallest=self.n_neighbors + 1
            )  # cdist_small stores also the distance of each point to itself, therefore use n_neighbors+1
        elif X.split == 1:
            dist, idx = cdist(X, X, metric=self.metric, n_smallest=self.n_neighbors + 1)
        else:
            raise ValueError(
                f"The data should be split among axis 0 or 1, but was split along axis {X.split}."
            )
        # Compute the reachability distance matrix
        reachability_dist = self._reach_dist(dist, idx)
        # Compute the local reachability density (lrd) for each point
        lrd = self.n_neighbors / (
            ht.sum(reachability_dist, axis=1) + 1e-10
        )  # add 1e-10 to avoid division by zero

        # define a matrix storing the lrd of all neighbors for each point
        lrd = lrd.resplit_(None)
        lrd_neighbors = ht.zeros((length, self.n_neighbors), split=None)

        # TODO: Once the advanced indexing is implemented in Heat, replace this loop by lrd_neighbors = lrd[idx[:, 1:]]
        for i in range(length):
            lrd_neighbors[i, :] = lrd[idx[i, 1:]]
        lrd = lrd.resplit_(X.split)
        lrd_neighbors = lrd_neighbors.resplit_(X.split)
        # Compute the local outlier factor for each point
        lof = ht.sum(lrd_neighbors, axis=1) / (self.n_neighbors * lrd + 1e-10)
        # Store the LOF scores in the class attribute
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

    def _reach_dist(self, dist, idx):
        """
        Computes the reachability distance matrix using MPI communication.

        The reachability distance is defined as [1]:
            reachability_dist(p, o) = max(k_dist(p), dist(p, o))
        where:
            - `p` is a reference point,
            - `o` is another data point,
            - `k_dist(p)` is the k-distance of `p`,
            - `dist(p, o)` is the pairwise distance between `p` and `o`.

        This function handles distributed computation by leveraging MPI communication.
        It ensures that each process retrieves the necessary distance rows, either locally
        or via communication with other processes, and then computes the maximum
        between `k_dist` and `dist`.

        Parameters:
        -----------
        dist : ht.DNDarray
            Pairwise distances between data points, calculated with the 'cdist_small' function in heat.
            It is expected to be split along the first axis (`split=0`).

        idx : ht.DNDarray
            Indices of the k-nearest neighbors from dist.
            Used to determine which rows of `dist` need to be accessed or communicated.

        Returns:
        --------
        reach_dist : ht.DNDarray
            Reachability distance matrix.

        Notes:
        ------
        - The auxiliary index arrays (`proc_id_global`, `k_dist_global`, `idx_k_dist_global`, `mapped_idx_global`)
          are assumed to fit into the memory of each process. This assumption helps to minimize
          communication overhead by storing global indices locally and speeds up the computation.
        - The MPI communication uses blocking send and receive commands. Non-blocking sending/receiving would
          mess up with functionality (overwriting the buffer)
        """
        # Compute the k-distance for each point
        k_dist = dist[:, -1]  # k-distance = largest value in dist for each row
        idx_k_dist = idx[:, -1]  # indices corresponding to k_dist

        # Set up communication parameters
        comm = dist.comm
        rank = comm.Get_rank()
        size = comm.Get_size()
        _, displ, _ = comm.counts_displs_shape(dist.shape, dist.split)

        reach_dist = ht.zeros_like(dist)
        reach_dist = reach_dist.larray
        dist_ = dist.larray

        # buffer to store one row of the distance matrix that is sent to the next process
        buffer = torch.zeros(
            (1, dist_.shape[1]),
            dtype=dist.dtype.torch_type(),
            device=dist.device.torch_device,
        )

        # map the indices of idx_k_dist to respective process, this serves as the list of senders
        senders = self._map_idx_to_proc(idx_k_dist, comm)
        # define list of receivers
        ones = ht.ones(int(idx_k_dist.shape[0]), split=0)
        receivers = ones * rank  # store the rank of each process

        # Store the senders and the respective receivers that shall communicate parts of the distance matrix
        communicators = ht.column_stack((receivers, senders))

        print(f"process: {rank}, idx_k_dist: {idx_k_dist}")

        # The fully distributed version requires two different communication steps:
        # 1. A cyclic communication of the array 'communicators' to all processes
        # 2. A point-to-point communication of the entries of the distance matrix according to 'communicators'
        if self.fully_distributed is True:
            pass
            # # type promotion
            # promoted_type = types.promote_types(communicators.dtype, types.float32)
            # communicators = communicators.astype(promoted_type)
            # if promoted_type == types.float32:
            #     torch_type = torch.float32
            #     mpi_type = MPI.FLOAT
            # elif promoted_type == types.float64:
            #     torch_type = torch.float64
            #     mpi_type = MPI.DOUBLE
            # else:
            #     raise NotImplementedError(f"Datatype {communicators.dtype} currently not supported as input")

            # Step 1 cyclic communication
            for i in range(size):
                if i != 0:
                    send_to = (rank + i) % size
                    recv_from = (rank - i) % size
                    # define a tag that does not overlap with the tags used in the point-to-point communication
                    cyclic_tag = communicators.shape[0] + i

                    # send
                    communicators.comm.Isend(communicators, dest=send_to, tag=cyclic_tag)
                    # define a dynamic buffer to receive the data (note the order: send->buffer->receive)
                    stat = MPI.Status()
                    communicators.comm.handle.Probe(source=recv_from, tag=cyclic_tag, status=stat)
                    count = int(stat.Get_count(MPI.INT) / communicators.shape[1])
                    buffer = torch.zeros(
                        (count, communicators.shape[1]),
                        dtype=communicators.dtype.torch_type(),
                        device=communicators.device.torch_device,
                    )
                    # receive
                    communicators.comm.Irecv(buffer, source=recv_from, tag=cyclic_tag)
                else:
                    buffer = communicators.larray
                # Step 2 point-to-point communication, i.e., start actual computation of the reachability distance
                for j in range(int(buffer.shape[0])):
                    receiver = int(buffer[j, 0].item())
                    sender = int(buffer[j, 1].item())
                    tag = j
                    idx_reach_dist = j
                    # assign
                    # idx_k_dist_ordered_ = idx_k_dist[
                    #     displ[rank] <= idx_k_dist < displ[rank + 1]
                    # ].larray

                    # check if current process needs to send the corresponding row of its distance matrix
                    if sender != receiver:
                        """# send
                        if rank == sender:
                            # select the row of the distance matrix to communicate between the processes
                            dist_row = dist_[int(idx_k_dist[j]), :]
                            sent_to_buffer = dist_row
                            # send the row to the next process
                            comm.Send(sent_to_buffer, dest=receiver, tag=tag)
                        # receive
                        if rank == receiver:
                            comm.Recv(buffer, source=sender, tag=tag)
                            dist_row = buffer
                            k_dist_compare = k_dist[j, None]
                            k_dist_compare = k_dist_compare.larray
                            reach_dist[idx_reach_dist] = torch.maximum(k_dist_compare, dist_row)"""
                        # print(f"process: {rank}, test 0")
                    # no communication required
                    elif sender == receiver:
                        # only take the row of the distance matrix that is already available
                        if rank == sender:
                            # TODO: The list idx_k_dist stores the global indices of the k-distances, which are not ordered,
                            # i.e., the index 110 can be in idx_k_dist on the first process, but the corresponding distance is stored on the second process.
                            dist_row = dist_[int(idx_k_dist[j]), :]
                            # k_dist_compare = k_dist[j, None]
                            # k_dist_compare = k_dist_compare.larray

                            # k_dist_compare = k_dist[1, None]
                            # k_dist_compare = k_dist_compare.larray
                            # print(f"process: {rank}, iteration: {j},  \n \n dist_row: {dist_row}, \n \n k_dist_compare: {k_dist_compare}")
                            # print(f"process: {rank}, iteration: {j},  test 03")
                            # reach_dist[idx_reach_dist] = torch.maximum(k_dist_compare, dist_row)
                        else:
                            pass
        print(f"process: {rank}, test 2")
        if self.fully_distributed is False:
            # use arrays as global ones to reduce communication overhead (assume they fit into memory of each process)
            receivers_global = receivers.resplit_(None)
            k_dist_global = k_dist.resplit_(None)
            idx_k_dist_global = idx_k_dist.resplit_(None)
            senders_global = senders.resplit_(None)
            for i in range(int(senders_global.shape[0])):
                receiver = receivers_global[i].item()
                sender = senders_global[i].item()
                tag = i
                # map the global index i to the local index of the reachability_dist array
                idx_reach_dist = i - displ[rank]
                # check if current process needs to send the corresponding row of its distance matrix
                if sender != receiver:
                    # send
                    if rank == sender:
                        if rank == size - 1:
                            upper_bound = senders_global.shape[0]
                        else:
                            upper_bound = displ[rank + 1]
                        if not displ[rank] <= i < upper_bound:
                            # select the row of the distance matrix to communicate between the processes
                            dist_row = dist_[int(idx_k_dist_global[i]) - displ[sender], :]
                            sent_to_buffer = dist_row
                            # send the row to the next process
                            comm.Send(sent_to_buffer, dest=receiver, tag=tag)
                    # receive
                    if rank == receiver:
                        comm.Recv(buffer, source=sender, tag=tag)
                        dist_row = buffer
                        k_dist_compare = k_dist_global[i, None]
                        k_dist_compare = k_dist_compare.larray
                        reach_dist[idx_reach_dist] = torch.maximum(k_dist_compare, dist_row)
                # no communication required
                elif sender == receiver:
                    # only take the row of the distance matrix that is already available
                    if rank == sender:
                        dist_row = dist_[int(idx_k_dist_global[i]) - displ[sender], :]
                        k_dist_compare = k_dist_global[i, None]
                        k_dist_compare = k_dist_compare.larray
                        reach_dist[idx_reach_dist] = torch.maximum(k_dist_compare, dist_row)
                    else:
                        pass
        reach_dist = ht.array(reach_dist, is_split=0)
        return reach_dist

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

        # if fully_distributed is not a boolean, raise an error
        if self.fully_distributed is not False and self.fully_distributed is not True:
            raise ValueError(
                f"The parameter fully_distributed should be either True or False, but was {self.fully_distributed}."
            )
