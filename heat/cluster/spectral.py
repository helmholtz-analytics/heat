import heat as ht
import math
import torch


class Spectral(ht.ClusteringMixin, ht.BaseEstimator):
    def __init__(
        self,
        n_clusters=None,
        gamma=1.0,
        metric="rbf",
        laplacian="fully_connected",
        threshold=1.0,
        boundary="upper",
        n_lanczos=300,
        assign_labels="kmeans",
        **params
    ):
        """
        Spectral clustering

        Parameters
        ----------
        n_clusters : int, optional
        gamma : float, default=1.0
            Kernel coefficient sigma for 'rbf', ignored for affinity='euclidean'
        metric : string
            How to construct the similarity matrix.
            'rbf' : construct the similarity matrix using a radial basis function (RBF) kernel.
            'euclidean' : construct the similarity matrix as only euclidean distance
        laplacian : string
            How to calculate the graph laplacian (affinity)
            Currently supported : 'fully_connected', 'eNeighbour'
        threshold : float
            Threshold for affinity matrix if laplacian='eNeighbour'
            Ignorded for laplacian='fully_connected'
        boundary : string
            How to interpret threshold: 'upper', 'lower'
            Ignorded for laplacian='fully_connected'
        n_lanczos : int
            number of Lanczos iterations for Eigenvalue decomposition
        assign_labels: str, default = 'kmeans'
             The strategy to use to assign labels in the embedding space.
             'kmeans'
        **params: dict
              Parameter dictionary for the assign_labels estimator
        """
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.metric = metric
        self.laplacian = laplacian
        self.threshold = threshold
        self.boundary = boundary
        self.n_lanczos = n_lanczos
        self.assign_labels = assign_labels

        if metric == "rbf":
            sig = math.sqrt(1 / (2 * gamma))
            self._laplacian = ht.graph.Laplacian(
                lambda x: ht.spatial.rbf(x, sigma=sig, quadratic_expansion=True),
                definition="norm_sym",
                mode=laplacian,
                threshold_key=boundary,
                threshold_value=threshold,
            )

        elif metric == "euclidean":
            self._laplacian = ht.graph.Laplacian(
                lambda x: ht.spatial.cdist(x, quadratic_expansion=True),
                definition="norm_sym",
                mode=laplacian,
                threshold_key=boundary,
                threshold_value=threshold,
            )
        else:
            raise NotImplementedError("Other kernels currently not supported")

        if assign_labels == "kmeans":
            self._cluster = ht.cluster.KMeans(params)
        else:
            raise NotImplementedError(
                "Other Label Assignment Algorithms are currently not available"
            )

        # in-place properties
        self._labels = None

    @property
    def labels_(self):
        """
        Returns
        -------
        ht.DNDarray, shape=(n_points):
            Labels of each point.
        """
        return self._labels

    def _spectral_embedding(self, X):
        """
        Helper function to embed the dataset X into the eigenvectors of the graph Laplacian matrix
        Returns
        -------
        ht.DNDarray, shape=(m_lanczos):
            Eigenvalues of the graph's Laplacian matrix.
        ht.DNDarray, shape=(n, m_lanczos):
            Eigenvectors of the graph's Laplacian matrix.
        """
        L = self._laplacian.construct(X)
        # 3. Eigenvalue and -vector calculation via Lanczos Algorithm
        v0 = ht.ones((L.shape[0],), dtype=L.dtype, split=0, device=L.device) / math.sqrt(L.shape[0])
        V, T = ht.lanczos(L, self.n_lanczos, v0)

        # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
        eval, evec = torch.eig(T._DNDarray__array, eigenvectors=True)
        # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
        eval, idx = torch.sort(eval[:, 0], dim=0)
        eigenvalues = ht.array(eval)
        eigenvectors = ht.matmul(V, ht.array(evec))[:, idx]

        return eigenvalues, eigenvectors

    def fit(self, X):
        """
        Computes the low-dim representation by calculation of eigenspectrum (eigenvalues and eigenvectors) of the graph
        laplacian from the similarity matrix and fits the eigenvectors that correspond to the k lowest eigenvalues with
        a seperate clustering algorithm (currently only kmeans is supported). Similarity metrics for adjacency
        calculations are supported via spatial.distance. The eigenvalues and eigenvectors are computed by reducing the
        Laplacian via lanczos iterations and using the torch eigenvalue solver on this smaller matrix. If other
        eigenvalue decompostion methods are supported, this will be expanded.

        Parameters
        ----------
        X : ht.DNDarray, shape=(n_samples, n_features)
            Training instances to cluster.
        """
        # 1. input sanitation
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))
        if X.split is not None and X.split != 0:
            raise NotImplementedError("Not implemented for other splitting-axes")
        # 2. Embed Dataset into lower-dimensional Eigenvector space
        eigenvalues, eigenvectors = self._spectral_embedding(X)

        # 3. Find the spectral gap, if number of clusters is not defined from the outside
        if self.n_clusters is None:
            diff = eigenvalues[1:] - eigenvalues[:-1]
            tmp = ht.where(diff == diff.max()).item()
            self.n_clusters = tmp + 1

        components = eigenvectors[:, : self.n_clusters].copy()

        params = self._cluster.get_params()
        params["n_clusters"] = self.n_clusters
        self._cluster.set_params(**params)
        self._cluster.fit(components)
        self._labels = self._cluster.labels_
        self._cluster_centers = self._cluster.cluster_centers_

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        X is transformed to the low-dim representation by calculation of eigenspectrum (eigenvalues and eigenvectors) of
        the graph laplacian from the similarity matrix. Inference of lables is done by extraction of the closest
        centroid of the n_clusters eigenvectors from the previously fitted clustering algorithm (kmeans).

        Caution: Calculation of the low-dim representation requires some time!

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : ht.DNDarray, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """
        # input sanitation
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))
        if X.split is not None and X.split != 0:
            raise NotImplementedError("Not implemented for other splitting-axes")

        _, eigenvectors = self._spectral_embedding(X)

        components = eigenvectors[:, : self.n_clusters].copy()

        return self._cluster.predict(components)
