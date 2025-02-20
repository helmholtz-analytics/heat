"""
Base-module for k-clustering algorithms
"""

import heat as ht
import torch
from heat.cluster.batchparallelclustering import _kmex
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
        - 'batchparallel': use the batch parallel algorithm to initialize the centroids, only available for split=0 and KMeans or KMedians
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
        self._functional_value = None
        self._labels = None
        self._inertia = None
        self._n_iter = None
        self._p = None

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

    @property
    def functional_value_(self) -> DNDarray:
        """
        Returns the K-Clustering functional value of the clustering algorithm
        """
        return self._functional_value

    def _initialize_cluster_centers(self, x: DNDarray, oversampling: float, iter_multiplier: float):
        """
        Initializes the K-Means centroids.

        Parameters
        ----------
        x : DNDarray
            The data to initialize the clusters for. Shape = (n_samples, n_features)

        oversampling : float
            oversampling factor used in the k-means|| initializiation of centroids

        iter_multiplier : float
            factor that increases the number of iterations used in the initialization of centroids

        Raises
        ------
        TypeError
            If the input is not a DNDarray
        ValueError
            If the oversampling factor or the iteration multiplier is too small
        """
        # input sanitation
        if not isinstance(x, DNDarray):
            raise ValueError(f"Input x needs to be a ht.DNDarray, but was {type(x)}")
        if oversampling < 2:
            raise ValueError(f"Oversampling factor should be at least 2, but was {oversampling}")
        if iter_multiplier < 1:
            raise ValueError(
                f"Iteration multiplier should be at least 1, but was {iter_multiplier}"
            )

        # always initialize the random state
        if self.random_state is not None:
            ht.random.seed(self.random_state)

        # initialize the centroids by randomly picking some of the points
        if self.init == "random":
            idx = ht.random.randint(0, x.shape[0] - 1, size=(self.n_clusters,), split=None)
            centroids = x[idx, :]
            self._cluster_centers = centroids if x.split == 1 else centroids.resplit_(None)

        # directly passed centroids
        elif isinstance(self.init, DNDarray):
            if len(self.init.shape) != 2:
                raise ValueError(
                    f"passed centroids need to be two-dimensional, but are {len(self.init)}"
                )
            if self.init.shape[0] != self.n_clusters or self.init.shape[1] != x.shape[1]:
                raise ValueError("passed centroids do not match cluster count or data shape")
            self._cluster_centers = self.init.resplit(None)

        # Parallelized centroid guessing using the k-means|| algorithm
        elif self.init == "probability_based":
            # First, check along which axis the data is sliced
            if x.split is None or x.split == 0:
                # Define a list of random, uniformly distributed probabilities,
                # which is later used to sample the centroids
                sample = ht.random.rand(x.shape[0], split=x.split)
                # Define a random integer serving as a label to pick the first centroid randomly
                init_idx = ht.random.randint(0, x.shape[0] - 1).item()
                # Randomly select first centroid and organize it as a tensor, in order to use the function cdist later.
                # This tensor will be filled continously in the proceeding of this function
                # We assume that the centroids fit into the memory of a single GPU
                centroids = ht.expand_dims(x[init_idx, :].resplit_(None), axis=0)
                # Calculate the initial cost of the clustering after the first centroid selection
                # and use it as an indicator for the order of magnitude for the number of necessary iterations
                init_distance = ht.spatial.distance.cdist(x, centroids, quadratic_expansion=True)
                # --> init_distance calculates the Euclidean distance between data points x and initial centroids
                # output format: tensor
                init_min_distance = init_distance.min(axis=1)
                # --> Pick the minimal distance of the data points to each centroid
                # output format: vector
                init_cost = init_min_distance.sum()
                # --> Now calculate the cost
                # output format: scalar
                #
                # Iteratively fill the tensor storing the centroids
                for _ in range(0, int(iter_multiplier * ht.log(init_cost))):
                    # Calculate the distance between data points and the current set of centroids
                    distance = ht.spatial.distance.cdist(x, centroids, quadratic_expansion=True)
                    min_distance = distance.min(axis=1)
                    # Sample each point in the data to a new set of centroids
                    prob = oversampling * min_distance / min_distance.sum()
                    # -->   probability distribution with oversampling factor
                    #       output format: vector
                    idx = ht.where(sample <= prob)
                    # -->   choose indices to sample the data according to prob
                    #       output format: vector
                    local_data = x[idx].resplit_(centroids.split)
                    # -->   pick the data points that are identified as possible centroids and make sure
                    #       that data points and centroids are split in the same way
                    #       output format: vector
                    centroids = ht.row_stack((centroids, local_data))
                    # -->   stack the data points with these indices to the DNDarray of centroids
                    #       output format: tensor
                # Evaluate distance between final centroids and data points
                if centroids.shape[0] <= self.n_clusters:
                    raise ValueError(
                        f"The parameter oversampling={oversampling} and/or iter_multiplier={iter_multiplier} "
                        "are chosen too small for the initialization of cluster centers."
                    )
                # Evaluate the distance between data and the final set of centroids for the initialization
                final_distance = ht.spatial.distance.cdist(x, centroids, quadratic_expansion=True)
                # For each data point in x, find the index of the centroid that is closest
                final_idx = ht.argmin(final_distance, axis=1)
                # Introduce weights, i.e., the number of data points closest to each centroid
                # (count how often the same index in final_idx occurs)
                weights = ht.zeros(centroids.shape[0], split=centroids.split)
                for i in range(centroids.shape[0]):
                    weights[i] = ht.sum(final_idx == i)
                # Recluster the oversampled centroids using standard k-means ++ (here we use the
                # already implemented version in torch)
                centroids = centroids.resplit_(None)
                centroids = centroids.larray
                weights = weights.resplit_(None)
                weights = weights.larray
                # --> first transform relevant arrays into torch tensors
                if ht.MPI_WORLD.rank == 0:
                    batch_kmeans = _kmex(
                        centroids,
                        p=2,
                        n_clusters=self.n_clusters,
                        init="++",
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=None,
                        weights=weights,
                    )
                    # --> apply standard k-means ++
                    #     Note: as we only recluster the centroids for initialization with standard k-means ++,
                    #     this list of centroids can also be used to initialize k-medians and k-medoids
                    reclustered_centroids = batch_kmeans[0]
                    # --> access the reclustered centroids
                else:
                    # ensure that all processes have the same data
                    reclustered_centroids = torch.zeros(
                        (self.n_clusters, centroids.shape[1]),
                        dtype=x.dtype.torch_type(),
                        device=centroids.device,
                    )
                    # -->  tensor with zeros that has the same size as reclustered centroids, in order to to
                    #      allocate memory with the correct type in all processes(necessary for broadcast)
                ht.MPI_WORLD.Bcast(
                    reclustered_centroids, root=0
                )  # by default it is broadcasted from process 0
                reclustered_centroids = ht.array(reclustered_centroids, split=None)
                # --> transform back to DNDarray
                self._cluster_centers = reclustered_centroids
                # --> final result for initialized cluster centers
            else:
                raise NotImplementedError("Not implemented for other splitting-axes")

        elif self.init == "batchparallel":
            if x.split == 0:
                if self._p == 2:
                    batch_parallel_clusterer = ht.cluster.BatchParallelKMeans(
                        n_clusters=self.n_clusters,
                        init="k-means++",
                        max_iter=100,
                        random_state=self.random_state,
                    )
                elif self._p == 1:
                    batch_parallel_clusterer = ht.cluster.BatchParallelKMedians(
                        n_clusters=self.n_clusters,
                        init="k-medians++",
                        max_iter=100,
                        random_state=self.random_state,
                    )
                else:
                    raise ValueError(
                        "Batch parallel initialization only implemented for KMeans and KMedians"
                    )
                batch_parallel_clusterer.fit(x)
                self._cluster_centers = batch_parallel_clusterer.cluster_centers_
            else:
                raise NotImplementedError(
                    f"Batch parallel initalization only implemented for split = 0, but split was {x.split}"
                )

        else:
            raise ValueError(
                'init needs to be one of "random", ht.DNDarray, "kmeans++", or "batchparallel", but was {}'.format(
                    self.init
                )
            )

    def _assign_to_cluster(self, x: DNDarray, eval_functional_value: bool = False):
        """
        Assigns the passed data points to the centroids based on the respective metric

        Parameters
        ----------
        x : DNDarray
            Data points, Shape = (n_samples, n_features)
        eval_functional_value : bool, default: False
            If True, the current K-Clustering functional value of the clustering algorithm is evaluated
        """
        # calculate the distance matrix and determine the closest centroid
        distances = self._metric(x, self._cluster_centers)
        matching_centroids = distances.argmin(axis=1, keepdims=True)

        if eval_functional_value:
            self._functional_value = ht.norm(distances.min(axis=1), ord=self._p) ** self._p

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
        return self._assign_to_cluster(x, eval_functional_value=True)
