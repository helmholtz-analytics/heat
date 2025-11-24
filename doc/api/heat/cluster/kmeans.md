Module heat.cluster.kmeans
==========================
Module Implementing the Kmeans Algorithm

Classes
-------

`KMeans(n_clusters: int = 8, init: str | heat.core.dndarray.DNDarray = 'random', max_iter: int = 300, tol: float = 0.0001, random_state: int | None = None)`
:   K-Means clustering algorithm. An implementation of Lloyd's algorithm [1].

    Attributes
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    init : str or DNDarray
        Method for initialization:

        - ‘k-means++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
        - ‘random’: choose k observations (rows) at random from data for the initial centroids.
        - 'batchparallel': initialize by using the batch parallel algorithm (see BatchParallelKMeans for more information).
        - DNDarray: it should be of shape (n_clusters, n_features) and gives the initial centers.
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int
        Determines random number generation for centroid initialization.

    Notes
    -----
    The average complexity is given by :math:`O(k \cdot n \cdot T)`, were n is the number of samples and :math:`T` is the number of iterations.
    In practice, the k-means algorithm is very fast, but it may fall into local minima. That is why it can be useful
    to restart it several times. If the algorithm stops before fully converging (because of ``tol`` or ``max_iter``),
    ``labels_`` and ``cluster_centers_`` will not be consistent, i.e. the ``cluster_centers_`` will not be the means of the
    points in each cluster. Also, the estimator will reassign ``labels_`` after the last iteration to make ``labels_``
    consistent with predict on the training set.

    References
    ----------
    [1] Lloyd, Stuart P., "Least squares quantization in PCM", IEEE Transactions on Information Theory, 28 (2), pp.
    129–137, 1982.

    [2] Arthur, D., Vassilvitskii, S., "k-means++: The Advantages of Careful Seeding", Proceedings of the Eighteenth
    Annual ACM-SIAM Symposium on Discrete Algorithms, Society for Industrial and Applied Mathematics
    Philadelphia, PA, USA. pp. 1027–1035, 2007.

    ### Ancestors (in MRO)

    * heat.cluster._kcluster._KCluster
    * heat.core.base.ClusteringMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray, oversampling: float = 2, iter_multiplier: float = 1) ‑> ~self`
    :   Computes the centroid of a k-means clustering. Reduce the values of the parameters 'oversampling'
        and 'iter_multiplier' to speed up the computation, if necessary. However, for too low values the
        initialization of cluster centers might fail and raise a corresponding ValueError.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        oversampling : float
            oversampling factor used for the k-means|| initializiation of centroids

        iter_multiplier : float
            factor that increases the number of iterations used in the initialization of centroids
