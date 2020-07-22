from typing import Optional, Union, TypeVar

import heat as ht
from heat.core.dndarray import DNDarray


self = TypeVar("self")


class KMeans(ht.ClusteringMixin, ht.BaseEstimator):
    """
    K-Means clustering algorithm. An implementation of Lloyd's algorithm [1].

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    init : str or DNDarray
        Method for initialization:

        - ‘k-means++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].

        - ‘random’: choose k observations (rows) at random from data for the initial centroids.

        - DNDarray: it should be of shape (n_clusters, n_features) and gives the initial centers.
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int
        Determines random number generation for centroid initialization.

    Notes
    -----
    The average complexity is given by :math:`O(k \\cdot n \\cdot T)`, were n is the number of samples and :math:`T` is the number of iterations.
    In practice, the k-means algorithm is very fast, but it may fall into local minima. That is why it can be useful
    to restart it several times. If the algorithm stops before fully converging (because of ``tol`` or ``max_iter``),
    labels_ and cluster_centers_ will not be consistent, i.e. the cluster_centers_ will not be the means of the
    points in each cluster. Also, the estimator will reassign labels_ after the last iteration to make labels_
    consistent with predict on the training set.

    References
    ----------
    [1] Lloyd, Stuart P., "Least squares quantization in PCM", IEEE Transactions on Information Theory, 28 (2), pp.
    129–137, 1982. \n
    [2] Arthur, D., Vassilvitskii, S., "k-means++: The Advantages of Careful Seeding", Proceedings of the Eighteenth
    Annual ACM-SIAM Symposium on Discrete Algorithms, Society for Industrial and Applied Mathematics
    Philadelphia, PA, USA. pp. 1027–1035, 2007.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, DNDarray] = "random",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.init = init
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tol = tol

        # in-place properties
        self._cluster_centers = None
        self._labels = None
        self._inertia = None
        self._n_iter = None

    @property
    def cluster_centers_(self) -> DNDarray:
        """
        Returns the coordinates of the cluster centers.
        If the algorithm stops before fully converging (see tol and max_iter),
        these will not be consistent with labels_.
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

        # kmeans++, smart centroid guessing
        elif self.init == "kmeans++":
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

    def _fit_to_cluster(self, x: DNDarray) -> DNDarray:
        """
        Assigns the passed data points to

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

       """
        # calculate the distance matrix and determine the closest centroid
        distances = ht.spatial.distance.cdist(x, self._cluster_centers, quadratic_expansion=True)
        matching_centroids = distances.argmin(axis=1, keepdim=True)

        return matching_centroids

    def fit(self, x: DNDarray) -> self:
        """
        Computes the centroid of a k-means clustering.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        """
        # input sanitation
        if not isinstance(x, DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(x)))

        # initialize the clustering
        self._initialize_cluster_centers(x)
        self._n_iter = 0
        matching_centroids = ht.zeros((x.shape[0]), split=x.split, device=x.device, comm=x.comm)

        new_cluster_centers = self._cluster_centers.copy()

        # iteratively fit the points to the centroids
        for epoch in range(self.max_iter):
            # increment the iteration count
            self._n_iter += 1
            # determine the centroids
            matching_centroids = self._fit_to_cluster(x)

            # update the centroids
            # compute the new centroids
            for i in range(self.n_clusters):
                # points in current cluster
                selection = (matching_centroids == i).astype(ht.int64)
                # accumulate points and total number of points in cluster
                assigned_points = x * selection
                points_in_cluster = selection.sum(axis=0, keepdim=True).clip(
                    1.0, ht.iinfo(ht.int64).max
                )

                # compute the new centroids
                new_cluster_centers[i : i + 1, :] = (assigned_points / points_in_cluster).sum(
                    axis=0, keepdim=True
                )

            # check whether centroid movement has converged
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.copy()
            if self.tol is not None and self._inertia <= self.tol:
                break

        self._labels = matching_centroids

        return self

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Returns the index of the closest cluster each sample in X belongs to.

        In the vector quantization literature, cluster_centers_ is called the code book and each value returned by
        predict is the index of the closest code in the code book.

        Parameters
        ----------
        x : DNDarray
            New data to predict. Shape = (n_samples, n_features)

        """
        # input sanitation
        if not isinstance(x, DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(x)))

        # determine the centroids
        return self._fit_to_cluster(x)
