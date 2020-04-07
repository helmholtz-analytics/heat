import heat as ht


class KMeans(ht.ClusteringMixin, ht.BaseEstimator):
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
    def cluster_centers_(self):
        """
        Returns
        -------
        ht.DNDarray, shape =  [n_clusters, n_features]:
            Coordinates of cluster centers. If the algorithm stops before fully converging (see tol and max_iter),
            these will not be consistent with labels_.
        """
        return self._cluster_centers

    @property
    def labels_(self):
        """
        Returns
        -------
        ht.DNDarray, shape = [n_points]:
            Labels of each point.
        """
        return self._labels

    @property
    def inertia_(self):
        """
        Returns
        -------
        float:
            Sum of squared distances of samples to their closest cluster center.
        """
        return self._inertia

    @property
    def n_iter_(self):
        """
        Returns
        -------
        int:
            Number of iterations run.
        """
        return self._n_iter

    def _initialize_cluster_centers(self, X):
        """
        Initializes the K-Means centroids.

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_point, n_features]:
            The data to initialize the clusters for.
        """
        # always initialize the random state
        if self.random_state is not None:
            ht.random.seed(self.random_state)

        # initialize the centroids by randomly picking some of the points
        if self.init == "random":
            # Samples will be equally distributed drawn from all involved processes
            _, displ, _ = X.comm.counts_displs_shape(shape=X.shape, axis=0)
            centroids = ht.empty(
                (self.n_clusters, X.shape[1]), split=None, device=X.device, comm=X.comm
            )
            if X.split is None or X.split == 0:
                for i in range(self.n_clusters):
                    samplerange = (
                        X.gshape[0] // self.n_clusters * i,
                        X.gshape[0] // self.n_clusters * (i + 1),
                    )
                    sample = ht.random.randint(samplerange[0], samplerange[1]).item()
                    proc = 0
                    for p in range(X.comm.size):
                        if displ[p] > sample:
                            break
                        proc = p
                    xi = ht.zeros(X.shape[1], dtype=X.dtype)
                    if X.comm.rank == proc:
                        idx = sample - displ[proc]
                        xi = ht.array(X.lloc[idx, :], device=X.device, comm=X.comm)
                    xi.comm.Bcast(xi, root=proc)
                    centroids[i, :] = xi

            else:
                raise NotImplementedError("Not implemented for other splitting-axes")

            self._cluster_centers = centroids

        # directly passed centroids
        elif isinstance(self.init, ht.DNDarray):
            if len(self.init.shape) != 2:
                raise ValueError(
                    "passed centroids need to be two-dimensional, but are {}".format(len(self.init))
                )
            if self.init.shape[0] != self.n_clusters or self.init.shape[1] != X.shape[1]:
                raise ValueError("passed centroids do not match cluster count or data shape")
            self._cluster_centers = self.init.resplit(None)

        # kmeans++, smart centroid guessing
        elif self.init == "kmeans++":
            if X.split is None or X.split == 0:
                centroids = ht.zeros(
                    (self.n_clusters, X.shape[1]), split=None, device=X.device, comm=X.comm
                )
                sample = ht.random.randint(0, X.shape[0] - 1).item()
                _, displ, _ = X.comm.counts_displs_shape(shape=X.shape, axis=0)
                proc = 0
                for p in range(X.comm.size):
                    if displ[p] > sample:
                        break
                    proc = p
                x0 = ht.zeros(X.shape[1], dtype=X.dtype, device=X.device, comm=X.comm)
                if X.comm.rank == proc:
                    idx = sample - displ[proc]
                    x0 = ht.array(X.lloc[idx, :], device=X.device, comm=X.comm)
                x0.comm.Bcast(x0, root=proc)
                centroids[0, :] = x0
                for i in range(1, self.n_clusters):
                    distances = ht.spatial.distance.cdist(X, centroids, quadratic_expansion=True)
                    D2 = distances.min(axis=1)
                    D2.resplit_(axis=None)
                    prob = D2 / D2.sum()
                    x = ht.random.rand().item()
                    sample = 0
                    sum = 0
                    for j in range(len(prob)):
                        if sum > x:
                            break
                        sum += prob[j].item()
                        sample = j
                    proc = 0
                    for p in range(X.comm.size):
                        if displ[p] > sample:
                            break
                        proc = p
                    xi = ht.zeros(X.shape[1], dtype=X.dtype)
                    if X.comm.rank == proc:
                        idx = sample - displ[proc]
                        xi = ht.array(X.lloc[idx, :], device=X.device, comm=X.comm)
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

    def _fit_to_cluster(self, X):
        """
        Assigns the passed data points to

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]:
            Training instances to cluster.
        """
        # calculate the distance matrix and determine the closest centroid
        distances = ht.spatial.distance.cdist(X, self._cluster_centers, quadratic_expansion=True)
        matching_centroids = distances.argmin(axis=1, keepdim=True)

        return matching_centroids

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

        new_cluster_centers = self._cluster_centers.copy()

        # iteratively fit the points to the centroids
        for epoch in range(self.max_iter):
            # increment the iteration count
            self._n_iter += 1
            # determine the centroids
            matching_centroids = self._fit_to_cluster(X)

            # update the centroids
            for i in range(self.n_clusters):
                # points in current cluster
                selection = (matching_centroids == i).astype(ht.int64)

                # accumulate points and total number of points in cluster
                assigned_points = (X * selection).sum(axis=0, keepdim=True)
                points_in_cluster = selection.sum(axis=0, keepdim=True).clip(
                    1.0, ht.iinfo(ht.int64).max
                )

                # compute the new centroids
                new_cluster_centers[i : i + 1, :] = assigned_points / points_in_cluster

            # check whether centroid movement has converged
            self._inertia = ((self._cluster_centers - new_cluster_centers) ** 2).sum()
            self._cluster_centers = new_cluster_centers.copy()
            if self.tol is not None and self._inertia <= self.tol:
                break

        self._labels = matching_centroids

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, cluster_centers_ is called the code book and each value returned by
        predict is the index of the closest code in the code book.

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]:
            New data to predict.

        Returns
        -------
        labels : ht.DNDarray, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """
        # input sanitation
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))

        # determine the centroids
        return self._fit_to_cluster(X)
