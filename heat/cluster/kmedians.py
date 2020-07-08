import heat as ht
from heat.cluster import KCluster


class KMedians(KCluster):
    def __init__(self, n_clusters=8, init="random", max_iter=300, tol=1e-4, random_state=None):
        """
        K-Medians clustering algorithm.

        Parameters
        ----------
        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.
        init : {‘random’ or an ndarray}
            Method for initialization, defaults to ‘random’:
            ‘k-medians++’ : selects initial cluster centers for the clustering in a smart way to speed up convergence [2].
            ‘random’: choose k observations (rows) at random from data for the initial centroids.
            If an ht.DNDarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        max_iter : int, default: 300
            Maximum number of iterations of the k-means algorithm for a single run.
        tol : float, default: 1e-4
            Relative tolerance with regards to inertia to declare convergence.
        random_state : int
            Determines random number generation for centroid initialization.

        """
        if init == "k-medians++":
            init = "probability_based"

        super().__init__(
            metric=lambda x, y: ht.spatial.distance.manhattan(x, y, expand=True),
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
            # Remove 0-element lines to avoid spoiling of median
            assigned_points = X * selection
            assigned_points = assigned_points[(assigned_points.abs()).sum(axis=1) != 0]
            median = ht.median(assigned_points, axis=0, keepdim=True)
            new_cluster_centers[i : i + 1, :] = median

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
