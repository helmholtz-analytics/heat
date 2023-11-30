"""
Module implementing some clustering algorithms that work in parallel on batches of data.
"""

import heat as ht
import torch
from heat.cluster._kcluster import _KCluster
from heat.core.dndarray import DNDarray

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
    p is the norm used for computing distances: p=2 implies k-means, p=1 implies k-medians
    (note: kmex stands for kmeans and kmedians)
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    if not (p == 1 or p == 2):
        raise Warning(
            "p should be 1 (k-medians) or 2 (k-means). For other choice of p, we proceed as for p=2 and hope for the best."
        )
    if isinstance(init, torch.Tensor):
        if init.shape != (n_clusters, X.shape[1]):
            raise ValueError("if a torch tensor, init must have shape (n_clusters, n_features).")
        centers = init
    elif init == "++":
        centers = _initialize_plus_plus(X, n_clusters, p)
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
            if p == 1:
                centers[i] = torch.median(X[labels == i], dim=0)[0]
            else:
                centers[i] = torch.mean(X[labels == i], dim=0)
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
        if not isinstance(init, str):
            raise TypeError(f"init must be str, but was {type(init)}")
        if not isinstance(random_state, int) and random_state is not None:
            raise TypeError(f"random_state must be int or None, but was {type(random_state)}")

        self.n_clusters = n_clusters
        self._init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # in-place properties
        if not (p == 1 or p == 2):
            raise Warning(
                "p should be 1 (k-Medians) or 2 (k-Means). For other choice of p, we proceed as for p=2 and hope for the best."
            )
        self._p = p
        self._cluster_centers = None
        self._n_iter = None

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

    def fit(self, x: DNDarray):
        """
        Computes the centroid of the clustering algorithm to fit the data ``x``.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features). It must hold x.split=0.

        """
        if not isinstance(x, DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")
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

        # collect set of all local centroids
        gathered_centers_local = torch.zeros(
            (x.comm.size, self.n_clusters, x.shape[1]), device=x.larray.device, dtype=x.larray.dtype
        )
        x.comm.Allgather(centers_local, gathered_centers_local)

        # "merge" local clusterings using k-clustering of the set of local centroids
        global_random_state = (
            None if self.random_state is None else self.random_state + x.comm.size + 1
        )
        centers, n_iters_global = _kmex(
            gathered_centers_local.view(-1, x.shape[1]),
            self._p,
            self.n_clusters,
            init=self._init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=global_random_state,
        )

        self._cluster_centers = DNDarray(
            centers,
            (self.n_clusters, x.shape[1]),
            dtype=x.dtype,
            device=x.device,
            comm=x.comm,
            split=None,
            balanced=True,
        )
        self._n_iter = (n_iters_local, n_iters_global)
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
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")
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

        labels = _parallel_batched_kmex_predict(x.larray, self._cluster_centers.larray, self._p)
        return ht.DNDarray(
            labels,
            (x.shape[0], 1),
            dtype=ht.int32,
            device=x.device,
            comm=x.comm,
            split=x.split,
            balanced=x.balanced,
        )


"""
Actual classes for batch parallel K-means and K-medians
"""


class BatchParallelKMeans(_BatchParallelKCluster):
    r"""
    Batch-parallel K-Means clustering algorithm from Ref. [1].
    This requires data to be given as DNDarray of shape (n_samples, n_features) with split=0 (i.e. split along the sample axis).
    The idea of the method is to perform the classical K-Means on each batch of data (i.e. on each process-local chunk of data) individually and in parallel.
    After that, all centroids from the local K-Means are gathered and another instance of K-means is performed on them in order to determine the final centroids.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    init : str
        Method for initialization for local and global k-means:
        - ‘k-means++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
        - ‘random’: choose k observations (rows) at random from data for the initial centroids.
    max_iter : int
        Maximum number of iterations of the local/global k-means algorithms.
    tol : float
        Relative tolerance with regards to inertia to declare convergence, both for local and global k-means.
    random_state : int
        Determines random number generation for centroid initialization.


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
    ):  # noqa: D107
        if not isinstance(init, str):
            raise TypeError(f"init must be str, but was {type(init)}")
        else:
            if init == "k-means++":
                _init = "++"
            elif init == "random":
                _init = "random"
            else:
                raise ValueError(f"init must be 'k-means++' or 'random', but was {init}")

        super().__init__(
            p=2,
            n_clusters=n_clusters,
            init=_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.init = init


class BatchParallelKMedians(_BatchParallelKCluster):
    r"""
    Batch-parallel K-Medians clustering algorithm, in analogy to the K-means algorithm from Ref. [1].
    This requires data to be given as DNDarray of shape (n_samples, n_features) with split=0 (i.e. split along the sample axis).
    The idea of the method is to perform the classical K-Medians on each batch of data (i.e. on each process-local chunk of data) individually and in parallel.
    After that, all centroids from the local K-Medians are gathered and another instance of K-Medians is performed on them in order to determine the final centroids.

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    init : str
        Method for initialization for local and global k-medians:
        - ‘k-medians++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
        - ‘random’: choose k observations (rows) at random from data for the initial centroids.
    max_iter : int
        Maximum number of iterations of the local/global k-Medians algorithms.
    tol : float
        Relative tolerance with regards to inertia to declare convergence, both for local and global k-Medians.
    random_state : int
        Determines random number generation for centroid initialization.


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
    ):  # noqa: D107
        if not isinstance(init, str):
            raise TypeError(f"init must be str, but was {type(init)}")
        else:
            if init == "k-medians++":
                _init = "++"
            elif init == "random":
                _init = "random"
            else:
                raise ValueError(f"init must be 'k-medians++' or 'random', but was {init}")

        super().__init__(
            p=1,
            n_clusters=n_clusters,
            init=_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.init = init
