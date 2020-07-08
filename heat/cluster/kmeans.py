import heat as ht
from heat.cluster import KCluster


class KMeans(KCluster):
    def __init__(self, n_clusters=8, init="random", max_iter=300, tol=1e-4, random_state=None):
        """
        K-Means clustering algorithm. An implementation of Lloyd's algorithm [1].

        Parameters
        ----------
        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.
        init : {‘random’ or an ndarray}
            Method for initialization, defaults to ‘random’:
            ‘k-means++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
            ‘random’: choose k observations (rows) at random from data for the initial centroids.
            If an ht.DNDarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        max_iter : int, default: 300
            Maximum number of iterations of the k-means algorithm for a single run.
        tol : float, default: 1e-4
            Relative tolerance with regards to inertia to declare convergence.
        random_state : int
            Determines random number generation for centroid initialization.

        Notes
        -----
        The average complexity is given by O(k*n*T), were n is the number of samples and T is the number of iterations.

        In practice, the k-means algorithm is very fast, but it may fall into local minima. That is why it can be useful
        to restart it several times. If the algorithm stops before fully converging (because of tol or max_iter),
        labels_ and cluster_centers_ will not be consistent, i.e. the cluster_centers_ will not be the means of the
        points in each cluster. Also, the estimator will reassign labels_ after the last iteration to make labels_
        consistent with predict on the training set.

        References
        ----------
        [1] Lloyd, Stuart P., "Least squares quantization in PCM", IEEE Transactions on Information Theory, 28 (2), pp.
            129–137, 1982.
        [2] Arthur, D., Vassilvitskii, S., "k-means++: The Advantages of Careful Seeding", Proceedings of the Eighteenth
            Annual ACM-SIAM Symposium on Discrete Algorithms, Society for Industrial and Applied Mathematics
            Philadelphia, PA, USA. pp. 1027–1035, 2007.
        """
        if init == "k-means++":
            init = "probability_based"

        super().__init__(
            metric=lambda x, y: ht.spatial.distance.cdist(x, y, quadratic_expansion=True),
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def _update_centroids(self, X, matching_centroids):
        new_cluster_centers = self._cluster_centers.copy()
        for i in range(self.n_clusters):
            # points in current cluster
            selection = (matching_centroids == i).astype(ht.int64)
            # accumulate points and total number of points in cluster
            assigned_points = X * selection
            points_in_cluster = selection.sum(axis=0, keepdim=True).clip(
                1.0, ht.iinfo(ht.int64).max
            )

            # compute the new centroids
            # ATTENTION: ERST TEILEN; DANN SUMIEREN!!!
            new_cluster_centers[i : i + 1, :] = (assigned_points / points_in_cluster).sum(
                axis=0, keepdim=True
            )

        return new_cluster_centers

    def fit(self, X):
        """
        Computes the centroid of a k-means clustering.

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]:
            Training instances to cluster.
        """
        # input sanitation
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))

        # initialize the clustering
        self._initialize_cluster_centers(X)
        self._n_iter = 0
        matching_centroids = ht.zeros((X.shape[0]), split=X.split, device=X.device, comm=X.comm)

        # iteratively fit the points to the centroids
        for epoch in range(self.max_iter):
            # increment the iteration count
            self._n_iter += 1
            # determine the centroids
            matching_centroids = self._assign_to_cluster(X)

            # update the centroids
            new_cluster_centers = self._update_centroids(X, matching_centroids)
            # check whether centroid movement has converged
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.copy()
            if self.tol is not None and self._inertia <= self.tol:
                break

        self._labels = matching_centroids

        return self
