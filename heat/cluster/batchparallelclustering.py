"""
Module implementing some clustering algorithms that work in parallel on batches of data.
"""

import heat as ht
import torch
from heat.cluster._kcluster import _KCluster
from heat.core.dndarray import DNDarray
from warnings import warn
from math import log


from typing import Union, Tuple, TypeVar

self = TypeVar("self")

"""
Auxiliary single-process functions and base class for batch-parallel k-clustering
"""


def _initialize_plus_plus(X, n_clusters, p, random_state=None):
    """
    Auxiliary function: single-process k-means++/k-medians++ initialization in pytorch
    p is the norm used for computing distances
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    idxs = torch.zeros(n_clusters, dtype=torch.long, device=X.device)
    idxs[0] = torch.randint(0, X.shape[0], (1,))
    for i in range(1, n_clusters):
        dist = torch.cdist(X, X[idxs[:i]], p=p)
        dist = torch.min(dist, dim=1)[0]
        idxs[i] = torch.multinomial(dist, 1)
    return X[idxs]


def _kmex(X, p, n_clusters, init, max_iter, tol, random_state=None):
    """
    Auxiliary function: single-process k-means and k-medians in pytorch
    p is the norm used for computing distances: p=2 implies k-means, p=1 implies k-medians.
    p should be 1 (k-medians) or 2 (k-means). For other choice of p, we proceed as for p=2 and hope for the best.
    (note: kmex stands for kmeans and kmedians)
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    if isinstance(init, torch.Tensor):
        if init.shape != (n_clusters, X.shape[1]):
            raise ValueError("if a torch tensor, init must have shape (n_clusters, n_features).")
        centers = init
    elif init == "++":
        centers = _initialize_plus_plus(X, n_clusters, p, random_state)
    elif init == "random":
        idxs = torch.randint(0, X.shape[0], (n_clusters,))
        centers = X[idxs]
    else:
        raise ValueError(
            "init must be a torch tensor with initial centers, string '++', or 'random'."
        )
    for _ in range(max_iter):
        dist = torch.cdist(X, centers, p=p)
        labels = torch.argmin(dist, dim=1)
        # update centers
        centers_old = centers.clone()
        for i in range(n_clusters):
            if (labels == i).any():
                if p == 1:
                    centers[i] = torch.median(X[labels == i], dim=0)[0]
                else:
                    centers[i] = torch.mean(X[labels == i], dim=0)
            else:
                # if a cluster is empty, we leave its center unchanged
                pass
        # check if tolerance is reached
        if torch.allclose(centers, centers_old, atol=tol):
            break
    return centers, _ + 1


def _parallel_batched_kmex_predict(X, centers, p):
    """
    Auxiliary function: predict labels for parallel_batched_kmex
    """
    dist = torch.cdist(X, centers, p=p)
    return torch.argmin(dist, dim=1).reshape(-1, 1)


class _BatchParallelKCluster(ht.ClusteringMixin, ht.BaseEstimator):
    """
    Base class for batch parallel k-clustering
    """

    def __init__(
        self,
        p: int,
        n_clusters: int,
        init: str,
        max_iter: int,
        tol: float,
        random_state: Union[int, None],
        n_procs_to_merge: Union[int, None],
    ):  # noqa: D107
        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be int, but was {type(n_clusters)}")
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, but was {n_clusters}")
        if not isinstance(max_iter, int):
            raise TypeError(f"max_iter must be int, but was {type(max_iter)}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, but was {max_iter}")
        if not isinstance(tol, float):
            raise TypeError(f"tol must be float, but was {type(tol)}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, but was {tol}")

        if not isinstance(random_state, int) and random_state is not None:
            raise TypeError(f"random_state must be int or None, but was {type(random_state)}")
        if not isinstance(n_procs_to_merge, int) and n_procs_to_merge is not None:
            raise TypeError(f"procs_to_merge must be int or None, but was {type(n_procs_to_merge)}")
        if n_procs_to_merge is not None and n_procs_to_merge <= 1:
            raise ValueError(
                f"If an integer, procs_to_merge must be > 1, but was {n_procs_to_merge}."
            )

        self.n_clusters = n_clusters
        self._init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_procs_to_merge = n_procs_to_merge

        # in-place properties
        if not (p == 1 or p == 2):
            warn(
                "p should be 1 (k-Medians) or 2 (k-Means). For other choice of p, we proceed as for p=2 and hope for the best.",
                UserWarning,
            )
        self._p = p
        self._cluster_centers = None
        self._n_iter = None
        self._functional_value = None

    @property
    def cluster_centers_(self) -> DNDarray:
        """
        Returns the coordinates of the cluster centers. Returns None if fit has not been called yet.
        """
        return self._cluster_centers

    @property
    def n_iter_(self) -> Tuple[int]:
        """
        Returns the number of iterations run. Returns None if fit has not been called yet.
        The output is of the form (n_iter_local, n_iter_global), where n_iter_local is the number of iterations of the local k-means/k-medians,
        and n_iter_global is the number of iterations of the global k-means/k-medians;
        """
        return self._n_iter

    @property
    def functional_value_(self) -> float:
        """
        Returns the value of the K-Clustering functional. Returns None if fit and predict have not been called yet.
        """
        return self._functional_value

    def fit(self, x: DNDarray):
        """
        Computes the centroid of the clustering algorithm to fit the data ``x``.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features). It must hold x.split=0.

        """
        if not isinstance(x, DNDarray):
            raise TypeError(f"input needs to be a ht.DNDarray, but was {type(x)}")
        if not x.ndim == 2:
            raise ValueError(f"input needs to be 2D, but was {x.ndim}D")
        if x.split != 0:
            raise ValueError(
                f"input needs to be split along the sample axis, but was split along {x.split}"
            )

        # local k-clustering
        local_random_state = None if self.random_state is None else self.random_state + x.comm.rank
        centers_local, n_iters_local = _kmex(
            x.larray,
            self._p,
            self.n_clusters,
            self._init,
            self.max_iter,
            self.tol,
            local_random_state,
        )

        # hierarchical approach to obtail "global" cluster centers from the "local" centers
        procs_to_merge = self.n_procs_to_merge if self.n_procs_to_merge is not None else x.comm.size
        current_procs = [i for i in range(x.comm.size)]
        current_comm = x.comm
        local_comm = current_comm.Split(current_comm.rank // procs_to_merge, x.comm.rank)

        level = 1
        while len(current_procs) > 1:
            if x.comm.rank in current_procs and local_comm.size > 1:
                # create array to collect centers from all processes of the process group of at most n_procs_to_merge processes
                if local_comm.rank == 0:
                    gathered_centers_local = torch.zeros(
                        (local_comm.size, self.n_clusters, x.shape[1]),
                        device=x.larray.device,
                        dtype=x.larray.dtype,
                    )
                else:
                    gathered_centers_local = torch.empty(
                        0, device=x.larray.device, dtype=x.larray.dtype
                    )
                # gather centers from all processes of the process group of at most n_procs_to_merge processes
                local_comm.Gather(centers_local, gathered_centers_local, root=0, axis=0)
                # k-clustering on the gathered centers
                if local_comm.rank == 0:
                    gathered_centers_local = gathered_centers_local.reshape(-1, x.shape[1])
                    centers_local, n_iters_local_new = _kmex(
                        gathered_centers_local,
                        self._p,
                        self.n_clusters,
                        self._init,
                        self.max_iter,
                        self.tol,
                        local_random_state,
                    )
                    del gathered_centers_local
                    n_iters_local += n_iters_local_new

            # update: determine processes to be active at next "merging" level, create new communicator and split it into groups for gathering
            current_procs = [
                current_procs[i] for i in range(len(current_procs)) if i % procs_to_merge == 0
            ]
            if len(current_procs) > 1:
                new_group = x.comm.group.Incl(current_procs)
                current_comm = x.comm.Create_group(new_group)
                if x.comm.rank in current_procs:
                    local_comm = ht.communication.MPICommunication(
                        current_comm.Split(current_comm.rank // procs_to_merge, x.comm.rank)
                    )
            level += 1

        # broadcast the final centers to all processes
        x.comm.Bcast(centers_local, root=0)

        self._cluster_centers = DNDarray(
            centers_local,
            (self.n_clusters, x.shape[1]),
            dtype=x.dtype,
            device=x.device,
            comm=x.comm,
            split=None,
            balanced=True,
        )
        self._n_iter = n_iters_local
        return self

    def predict(self, x: DNDarray):
        """
        Predict the closest cluster each sample in ``x`` belongs to.

        In the vector quantization literature, :func:`cluster_centers_` is called the code book and each value returned by
        predict is the index of the closest code in the code book.

        Parameters
        ----------
        x : DNDarray
            New data to predict. Shape = (n_samples, n_features)
        """
        # input sanitation
        if not isinstance(x, DNDarray):
            raise TypeError(f"input needs to be a ht.DNDarray, but was {type(x)}")
        if not x.ndim == 2:
            raise ValueError(f"input needs to be 2D, but was {x.ndim}D")
        if x.split != 0:
            raise ValueError(
                f"input needs to be split along the sample axis, but was split along {x.split}"
            )
        if self._cluster_centers is None:
            raise RuntimeError("fit needs to be called before predict")
        if x.shape[1] != self._cluster_centers.shape[1]:
            raise ValueError(
                f"input needs to have {self._cluster_centers.shape[1]} features, but has {x.shape[1]}"
            )

        local_labels = _parallel_batched_kmex_predict(
            x.larray, self._cluster_centers.larray, self._p
        )
        labels = DNDarray(
            local_labels,
            gshape=(x.shape[0], 1),
            dtype=ht.int32,
            device=x.device,
            comm=x.comm,
            split=x.split,
            balanced=x.balanced,
        )
        if self._p == 2:
            self._functional_value = (
                torch.norm(
                    x.larray - self._cluster_centers.larray[local_labels, :].squeeze(), p="fro"
                )
                ** 2
            )
        else:
            self._functional_value = torch.norm(
                x.larray - self._cluster_centers.larray[local_labels, :].squeeze(), p=self._p, dim=1
            ).sum()
        x.comm.Allreduce(ht.communication.MPI.IN_PLACE, self._functional_value)
        self._functional_value = self._functional_value.item()
        return labels


"""
Actual classes for batch parallel K-means and K-medians
"""


class BatchParallelKMeans(_BatchParallelKCluster):
    r"""
    Batch-parallel K-Means clustering algorithm from Ref. [1].
    The input must be a ``DNDarray`` of shape `(n_samples, n_features)`, with split=0 (i.e. split along the sample axis).
    This method performs K-Means clustering on each batch (i.e. on each process-local chunk) of data individually and in parallel.
    After that, all centroids from the local K-Means are gathered and another instance of K-means is performed on them in order to determine the final centroids.
    To improve scalability of this approach also on a large number of processes, this procedure can be applied in a hierarchical manner using the parameter `n_procs_to_merge`.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    init : str
        Method for initialization for local and global k-means:
        - ‘k-means++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
        - ‘random’: choose k observations (rows) at random from data for the initial centroids. (Not implemented yet)
    max_iter : int
        Maximum number of iterations of the local/global k-means algorithms.
    tol : float
        Relative tolerance with regards to inertia to declare convergence, both for local and global k-means.
    random_state : int
        Determines random number generation for centroid initialization.
    n_procs_to_merge : int
        Number of processes to merge after each iteration of the local k-means. If None, all processes are merged after each iteration.


    References
    ----------
    [1] Rasim M. Alguliyev, Ramiz M. Aliguliyev, Lyudmila V. Sukhostat, Parallel batch k-means for Big data clustering, Computers & Industrial Engineering, Volume 152 (2021). https://doi.org/10.1016/j.cie.2020.107023.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = None,
        n_procs_to_merge: int = None,
    ):  # noqa: D107
        if not isinstance(init, str):
            raise TypeError(f"init must be str, but was {type(init)}")
        else:
            if init == "k-means++":
                _init = "++"
            elif init == "random":
                raise NotImplementedError(
                    "random initialization for batch parallel k-means is currently not supported due to instable behaviour of the algorithm. Use init='k-means++' instead."
                )
            else:
                raise ValueError(f"init must be 'k-means++' or 'random', but was {init}")
        super().__init__(
            p=2,
            n_clusters=n_clusters,
            init=_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_procs_to_merge=n_procs_to_merge,
        )
        self.init = init


class BatchParallelKMedians(_BatchParallelKCluster):
    r"""
    Batch-parallel K-Medians clustering algorithm, in analogy to the K-means algorithm from Ref. [1].
    This requires data to be given as DNDarray of shape (n_samples, n_features) with split=0 (i.e. split along the sample axis).
    The idea of the method is to perform the classical K-Medians on each batch of data (i.e. on each process-local chunk of data) individually and in parallel.
    After that, all centroids from the local K-Medians are gathered and another instance of K-Medians is performed on them in order to determine the final centroids.
    To improve scalability of this approach also on a range number of processes, this procedure can be applied in a hierarchical manor using the parameter n_procs_to_merge.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    init : str
        Method for initialization for local and global k-medians:
        - ‘k-medians++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
        - ‘random’: choose k observations (rows) at random from data for the initial centroids. (Not implemented yet)
    max_iter : int
        Maximum number of iterations of the local/global k-Medians algorithms.
    tol : float
        Relative tolerance with regards to inertia to declare convergence, both for local and global k-Medians.
    random_state : int
        Determines random number generation for centroid initialization.
    n_procs_to_merge : int
        Number of processes to merge after each iteration of the local k-Medians. If None, all processes are merged after each iteration.


    References
    ----------
    [1] Rasim M. Alguliyev, Ramiz M. Aliguliyev, Lyudmila V. Sukhostat, Parallel batch k-means for Big data clustering, Computers & Industrial Engineering, Volume 152 (2021). https://doi.org/10.1016/j.cie.2020.107023.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-medians++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = None,
        n_procs_to_merge: int = None,
    ):  # noqa: D107
        if not isinstance(init, str):
            raise TypeError(f"init must be str, but was {type(init)}")
        else:
            if init == "k-medians++":
                _init = "++"
            elif init == "random":
                raise NotImplementedError(
                    "random initialization for batch parallel k-medians is currently not supported due to instable behaviour of the algorithm. Use init='k-medians++' instead."
                )
            else:
                raise ValueError(f"init must be 'k-medians++' or 'random', but was {init}")
        super().__init__(
            p=1,
            n_clusters=n_clusters,
            init=_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_procs_to_merge=n_procs_to_merge,
        )
        self.init = init
