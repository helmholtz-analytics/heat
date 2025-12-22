Module heat.cluster.batchparallelclustering
===========================================
Module implementing some clustering algorithms that work in parallel on batches of data.

Variables
---------

`self`
:   Auxiliary single-process functions and base class for batch-parallel k-clustering

Classes
-------

`BatchParallelKMeans(n_clusters: int = 8, init: str = 'k-means++', max_iter: int = 300, tol: float = 0.0001, random_state: int = None, n_procs_to_merge: int = None)`
:   Batch-parallel K-Means clustering algorithm from Ref. [1].
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

    ### Ancestors (in MRO)

    * heat.cluster.batchparallelclustering._BatchParallelKCluster
    * heat.core.base.ClusteringMixin
    * heat.core.base.BaseEstimator

`BatchParallelKMedians(n_clusters: int = 8, init: str = 'k-medians++', max_iter: int = 300, tol: float = 0.0001, random_state: int = None, n_procs_to_merge: int = None)`
:   Batch-parallel K-Medians clustering algorithm, in analogy to the K-means algorithm from Ref. [1].
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

    ### Ancestors (in MRO)

    * heat.cluster.batchparallelclustering._BatchParallelKCluster
    * heat.core.base.ClusteringMixin
    * heat.core.base.BaseEstimator
