import heat as ht


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
    init : str or DNDarray
        Method for initialization, defaults to ‘random’:
            - ‘probability_based’ : selects initial cluster centers for the clustering in a smart way to speed up convergence (k-means++) \n
            - ‘random’: choose k observations (rows) at random from data for the initial centroids. \n
            - ``DNDarray``: gives the initial centers, should be of Shape = (n_clusters, n_features)
    max_iter : int
        Maximum number of iterations for a single run.
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int
        Determines random number generation for centroid initialization.
    """

    def __init__(self, metric, n_clusters, init, max_iter, tol, random_state):
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
    def cluster_centers_(self):
        """
        Returns the coordinates of cluster centers.
        """
        return self._cluster_centers

    @property
    def labels_(self):
        """
        Returns the labels of each point.
        """
        return self._labels

    @property
    def inertia_(self):
        """
        Returns sum of squared distances of samples to their closest cluster center.
        """
        return self._inertia

    @property
    def n_iter_(self):
        """
        Returns the number of iterations run.
        """
        return self._n_iter

    def _initialize_cluster_centers(self, X):
        """
        Initializes the K-Means centroids.

        Parameters
        ----------
        X : DNDarray,
            The data to initialize the clusters for. Shape = (n_point, n_features)
        """
        # always initialize the random state
        if self.random_state is not None:
            ht.random.seed(self.random_state)

        # initialize the centroids by randomly picking some of the points
        if self.init == "random":
            # Samples will be equally distributed drawn from all involved processes
            _, displ, _ = X.comm.counts_displs_shape(shape=X.shape, axis=0)
            # At least float32 precision, if X has higher precision datatype, use that
            datatype = ht.promote_types(X.dtype, ht.float32)
            centroids = ht.empty(
                (self.n_clusters, X.shape[1]),
                split=None,
                dtype=datatype,
                device=X.device,
                comm=X.comm,
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

        # smart centroid guessing, random sampling with probability weight proportional to distance to existing centroids
        elif self.init == "probability_based":
            datatype = ht.promote_types(X.dtype, ht.float32)
            if X.split is None or X.split == 0:
                centroids = ht.zeros(
                    (self.n_clusters, X.shape[1]),
                    split=None,
                    dtype=datatype,
                    device=X.device,
                    comm=X.comm,
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
                    distances = self._metric(X, centroids)
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

    def _assign_to_cluster(self, X):
        """
        Assigns the passed data points to the centroids based on the respective metric

        Parameters
        ----------
        X : DNDarray
            Data points, Shape = (n_samples, n_features)
        """
        # calculate the distance matrix and determine the closest centroid
        distances = self._metric(X, self._cluster_centers)
        matching_centroids = distances.argmin(axis=1, keepdim=True)

        return matching_centroids

    def _update_centroids(self, X, matching_centroids):
        """
        The Update strategy is algorithm specific (e.g. calculate mean of assigned points for kmeans, median for kmedians, etc.)

        Parameters
        ----------
        X : DNDarray
            Input Data
        matching_centroids: DNDarray
            Index array of assigned centroids

        """
        raise NotImplementedError()

    def fit(self, X):
        """
       Computes the centroid of the clustering algorithm to fit the data X. The full pipeline is algorithm specific.

        Parameters
        ----------
        X : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        """
        raise NotImplementedError()

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, cluster_centers_ is called the code book and each value returned by
        predict is the index of the closest code in the code book.

        Parameters
        ----------
        X : DNDarray
            New data to predict. Shape = (n_samples, n_features)
       """
        # input sanitation
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))

        # determine the centroids
        return self._assign_to_cluster(X)
