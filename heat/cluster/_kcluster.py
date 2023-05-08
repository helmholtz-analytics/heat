"""
Base-module for k-clustering algorithms
"""

import heat as ht
from typing import Optional, Union, Callable
from heat.core.dndarray import DNDarray


class _KCluster(ht.ClusteringMixin, ht.BaseEstimator):
    """
    Base class for k-statistics clustering algorithms (kmeans, kmedians, kmedoids).
    The clusters are represented by centroids ci (we use the term from kmeans for simplicity)

    Parameters
    ----------
    metric : function
        One of the distance metrics in ht.spatial.distance. Needs to be passed as lambda function to take only two arrays as input
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    init : str or DNDarray, default: ‘random’
        Method for initialization:

        - ‘probability_based’ : selects initial cluster centers for the clustering in a smart way to speed up convergence (k-means++)
        - ‘random’: choose k observations (rows) at random from data for the initial centroids.
        - ``DNDarray``: gives the initial centers, should be of Shape = (n_clusters, n_features)
    max_iter : int
        Maximum number of iterations for a single run.
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int
        Determines random number generation for centroid initialization.
    """

    def __init__(
        self,
        metric: Callable,
        n_clusters: int,
        init: Union[str, DNDarray],
        max_iter: int,
        tol: float,
        random_state: int,
    ):  # noqa: D107
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # in-place properties
        self._metric = metric
        self._cluster_centers = None
        self._labels = None
        self._inertia = None
        self._n_iter = None

    @property
    def cluster_centers_(self) -> DNDarray:
        """
        Returns the coordinates of the cluster centers.
        If the algorithm stops before fully converging (see ``tol`` and ``max_iter``),
        these will not be consistent with :func:`labels_`.
        """
        return self._cluster_centers

    @property
    def labels_(self) -> DNDarray:
        """
        Returns the labels of each point
        """
        return self._labels

    @property
    def inertia_(self) -> float:
        """
        Returns the sum of squared distances of samples to their closest cluster center.
        """
        return self._inertia

    @property
    def n_iter_(self) -> int:
        """
        Returns the number of iterations run.
        """
        return self._n_iter

    def _initialize_cluster_centers(self, x: DNDarray):
        """
        Initializes the K-Means centroids.

        Parameters
        ----------
        x : DNDarray
            The data to initialize the clusters for. Shape = (n_samples, n_features)
        """
        # always initialize the random state
        if self.random_state is not None:
            ht.random.seed(self.random_state)

        # initialize the centroids by randomly picking some of the points
        if self.init == "random":
            # Samples will be equally distributed drawn from all involved processes
            _, displ, _ = x.comm.counts_displs_shape(shape=x.shape, axis=0)
            centroids = ht.empty(
                (self.n_clusters, x.shape[1]), split=None, device=x.device, comm=x.comm
            )
            if x.split is None or x.split == 0:
                for i in range(self.n_clusters):
                    samplerange = (
                        x.gshape[0] // self.n_clusters * i,
                        x.gshape[0] // self.n_clusters * (i + 1),
                    )
                    sample = ht.random.randint(samplerange[0], samplerange[1]).item()
                    proc = 0
                    for p in range(x.comm.size):
                        if displ[p] > sample:
                            break
                        proc = p
                    xi = ht.zeros(x.shape[1], dtype=x.dtype)
                    if x.comm.rank == proc:
                        idx = sample - displ[proc]
                        xi = ht.array(x.lloc[idx, :], device=x.device, comm=x.comm)
                    xi.comm.Bcast(xi, root=proc)
                    centroids[i, :] = xi

            else:
                raise NotImplementedError("Not implemented for other splitting-axes")

            self._cluster_centers = centroids

        # directly passed centroids
        elif isinstance(self.init, DNDarray):
            if len(self.init.shape) != 2:
                raise ValueError(
                    "passed centroids need to be two-dimensional, but are {}".format(len(self.init))
                )
            if self.init.shape[0] != self.n_clusters or self.init.shape[1] != x.shape[1]:
                raise ValueError("passed centroids do not match cluster count or data shape")
            self._cluster_centers = self.init.resplit(None)

        # Smart centroid guessing, random sampling with probability weight proportional to distance to existing centroids
        elif self.init == "probability_based":
            if x.split is None or x.split == 0:
                centroids = ht.zeros(
                    (self.n_clusters, x.shape[1]), split=None, device=x.device, comm=x.comm
                )
                sample = ht.random.randint(0, x.shape[0] - 1).item()
                _, displ, _ = x.comm.counts_displs_shape(shape=x.shape, axis=0)
                proc = 0
                for p in range(x.comm.size):
                    if displ[p] > sample:
                        break
                    proc = p
                x0 = ht.zeros(x.shape[1], dtype=x.dtype, device=x.device, comm=x.comm)
                if x.comm.rank == proc:
                    idx = sample - displ[proc]
                    x0 = ht.array(x.lloc[idx, :], device=x.device, comm=x.comm)
                x0.comm.Bcast(x0, root=proc)
                centroids[0, :] = x0
                for i in range(1, self.n_clusters):
                    distances = ht.spatial.distance.cdist(x, centroids, quadratic_expansion=True)
                    D2 = distances.min(axis=1)
                    D2.resplit_(axis=None)
                    prob = D2 / D2.sum()
                    random_position = ht.random.rand().item()
                    sample = 0
                    sum = 0
                    for j in range(len(prob)):
                        if sum > random_position:
                            break
                        sum += prob[j].item()
                        sample = j
                    proc = 0
                    for p in range(x.comm.size):
                        if displ[p] > sample:
                            break
                        proc = p
                    xi = ht.zeros(x.shape[1], dtype=x.dtype)
                    if x.comm.rank == proc:
                        idx = sample - displ[proc]
                        xi = ht.array(x.lloc[idx, :], device=x.device, comm=x.comm)
                    xi.comm.Bcast(xi, root=proc)
                    centroids[i, :] = xi

            else:
                raise NotImplementedError("Not implemented for other splitting-axes")
            self._cluster_centers = centroids

        else:
            raise ValueError(
                'init needs to be one of "random", ht.DNDarray or "kmeans++", but was {}'.format(
                    self.init
                )
            )

    def _assign_to_cluster(self, x: DNDarray):
        """
        Assigns the passed data points to the centroids based on the respective metric

        Parameters
        ----------
        x : DNDarray
            Data points, Shape = (n_samples, n_features)
        """
        # calculate the distance matrix and determine the closest centroid
        distances = self._metric(x, self._cluster_centers)
        matching_centroids = distances.argmin(axis=1, keepdims=True)

        return matching_centroids

    def _update_centroids(self, x: DNDarray, matching_centroids: DNDarray):
        """
        The Update strategy is algorithm specific (e.g. calculate mean of assigned points for kmeans, median for kmedians, etc.)

        Parameters
        ----------
        x : DNDarray
            Input Data
        matching_centroids: DNDarray
            Index array of assigned centroids

        """
        raise NotImplementedError()

    def fit(self, x: DNDarray):
        """
        Computes the centroid of the clustering algorithm to fit the data ``x``. The full pipeline is algorithm specific.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        """
        raise NotImplementedError()

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

        # determine the centroids
        return self._assign_to_cluster(x)
