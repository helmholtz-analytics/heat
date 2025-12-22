Module heat.cluster.kmedians
============================
Module Implementing the Kmedians Algorithm

Classes
-------

`KMedians(n_clusters: int = 8, init: str | heat.core.dndarray.DNDarray = 'random', max_iter: int = 300, tol: float = 0.0001, random_state: int = None)`
:   K-Medians clustering algorithm [1].
    Uses the Manhattan (City-block, :math:`L_1`) metric for distance calculations

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids to generate.
    init : str or DNDarray, default: ‘random’
        Method for initialization:

             - ‘k-medians++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
             - ‘random’: choose k observations (rows) at random from data for the initial centroids.
             - 'batchparallel': initialize by using the batch parallel algorithm (see BatchParallelKMedians for more information).
             - DNDarray: gives the initial centers, should be of Shape = (n_clusters, n_features)
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int
        Determines random number generation for centroid initialization.

    References
    ----------
    [1] Hakimi, S., and O. Kariv. "An algorithmic approach to network location problems II: The p-medians." SIAM Journal on Applied Mathematics 37.3 (1979): 539-560.

    ### Ancestors (in MRO)

    * heat.cluster._kcluster._KCluster
    * heat.core.base.ClusteringMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray, oversampling: float = 2, iter_multiplier: float = 1)`
    :   Computes the centroid of a k-medians clustering.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        oversampling : float
            oversampling factor used in the k-means|| initializiation of centroids

        iter_multiplier : float
            factor that increases the number of iterations used in the initialization of centroids
