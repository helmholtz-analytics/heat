Module heat.cluster.kmedoids
============================
Module Implementing the Kmedoids Algorithm

Classes
-------

`KMedoids(n_clusters: int = 8, init: str | heat.core.dndarray.DNDarray = 'random', max_iter: int = 300, random_state: int = None)`
:   Kmedoids with the Manhattan distance as fixed metric, calculating the median of the assigned cluster points as new cluster center
    and snapping the centroid to the the nearest datapoint afterwards.
    This is not the original implementation of k-medoids using PAM as originally proposed by in [1].

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids to generate.
    init : str or DNDarray, default: ‘random’
        Method for initialization:

            - ‘k-medoids++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
            - ‘random’: choose k observations (rows) at random from data for the initial centroids.
            - DNDarray: gives the initial centers, should be of Shape = (n_clusters, n_features)
    max_iter : int, default: 300
        Maximum number of iterations of the  algorithm for a single run.
    random_state : int
        Determines random number generation for centroid initialization.

    References
    ----------
    [1] Kaufman, L. and Rousseeuw, P.J. (1987), Clustering by means of Medoids, in Statistical Data Analysis Based on the L1 Norm and Related Methods, edited by Y. Dodge, North-Holland, 405416.

    ### Ancestors (in MRO)

    * heat.cluster._kcluster._KCluster
    * heat.core.base.ClusteringMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray, oversampling: float = 2, iter_multiplier: float = 1)`
    :   Computes the centroid of a k-medoids clustering.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)
        oversampling : float
            oversampling factor used in the k-means|| initializiation of centroids

        iter_multiplier : float
            factor that increases the number of iterations used in the initialization of centroids
