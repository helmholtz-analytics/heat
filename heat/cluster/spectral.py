import torch
import math
import numpy as np
import heat as ht
import time


def laplacian(X, similarity, gamma=1.0, norm=True, mode="fc", upper=None, lower=None):
    """
    Construct the graph Laplacian from a dataset

    Parameters
    ----------
    X : ht.DNDarray
        Dataset of dimensions (samples x features)
    similarity:
        Similarity metrices s_ij between data samples i and j. Can be (currently) 'rbf' or 'euclidean'. For 'rbf', the kernel parameter gamma has to be set
    gamma:
        Similarity kernel Parameter. Ignored for Similarity="euclidean"
    norm : bool
        Whether to calculate the normalized graph Laplacian
        The unnormalized graph Laplacian is defined as L = D - A, where where D is the diagonal degree matrix and A the adjacency matrix
        The normalized graph Laplacian is defined as L_norm = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2)
    mode : "fc", "eNeighbour"
        How to calculate adjacency from the similarity matrix
        "fc" is fully-connected, so A = S
        "eNeighbour" is the epsilon neighbourhood, with A_ji = 1 if S_ij >/< lower/upper else 0; for eNeighbour an upper or lower boundary needs to be set
    upper : float
        upper boundary for adjacency calculation, using A_ji = 1 if S_ij < upper else 0
    lower : float
        upper boundary for adjacency calculation, using A_ji = 1 if S_ij > lower else 0
    Returns
    -------
    L : ht.DNDarray

    """
    if similarity == "rbf":
        sig = math.sqrt(1 / (2 * gamma))
        S = ht.spatial.rbf(X, sigma=sig, quadratic_expansion=True)
    elif similarity == "euclidean":
        S = ht.spatial.cdist(X, quadratic_expansion=True)
    else:
        raise NotImplementedError("Other kernels currently not implemented")

    if mode == "eNeighbour":
        if (upper is not None) and (lower is not None):
            raise ValueError(
                "Epsilon neighborhood with either upper or lower threshold, both is not supported"
            )
        if (upper is None) and (lower is None):
            raise ValueError("Epsilon neighborhood requires upper or lower threshold to be defined")
        if upper is not None:
            S = ht.int(S < upper)
        elif lower is not None:
            S = ht.int(S > lower)
    elif mode == "fc":
        S = S - ht.eye(S.shape)
    else:
        raise NotImplementedError(
            "Only eNeighborhood and fully-connected graphs supported at the moment."
        )
    degree = ht.sum(S, axis=1)

    if norm:
        degree.resplit_(axis=None)
        temp = torch.ones(degree.shape, dtype=degree._DNDarray__array.dtype)
        degree._DNDarray__array = torch.where(
            degree._DNDarray__array == 0, temp, degree._DNDarray__array
        )
        w = S / ht.sqrt(ht.expand_dims(degree, axis=1))
        w = w / ht.sqrt(ht.expand_dims(degree, axis=0))
        L = ht.eye(S.shape, dtype=S.dtype, split=S.split, device=S.device, comm=S.comm) - w

    else:
        L = ht.diag(degree) - S

    return L


class Spectral:
    def __init__(
        self,
        n_clusters=None,
        gamma=1.0,
        metric="rbf",
        laplacian="fully_connected",
        threshold=1.0,
        boundary="upper",
        normalize=True,
        n_lanczos=300,
        assign_labels="kmeans",
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
            'fully_connected'
            'eNeighborhood'
        theshold : float
            Threshold for affinity matrix if laplacian='eNeighborhood'
            Ignorded for laplacian='fully_connected'
        boundary : string
            How to interpret threshold
            'upper'
            'lower'
        normalize : bool, default = True
            normalized vs. unnormalized Laplacian
        n_lanczos : int
            number of Lanczos iterations for Eigenvalue decomposition
        assign_labels: str, default = 'kmeans'
             The strategy to use to assign labels in the embedding space.
             'kmeans'

        """
        self.n_clusters = n_clusters
        self.n_lanczos = n_lanczos
        self.metric = metric
        self.gamma = gamma
        self.normalize = normalize
        self.laplacian = laplacian
        self.epsilon = (threshold, boundary)
        if assign_labels == "kmeans":
            self._cluster = ht.cluster.KMeans(init="random", max_iter=30, tol=-1.0)
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

    def fit(self, X):
        """
        Computes the low-dim representation by calculation of eigenspectrum (eigenvalues and eigenvectors) of the graph laplacian from the similarity matrix and fits the eigenvectors that correspond to the k lowest eigenvalues with a seperate clustering algorithm (currently only kemans is supported)
        Similarity metrics for adjacency calculations are supported via spatial.distance. The eigenvalues and eigenvectors are computed by reducing the Laplacian via lanczos iterations and using the torch eigenvalue solver on this smaller matrix. If other eigenvalue decompostion methods are supported, this will be expanded.

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
        # 2. Construct Laplacian
        if self.laplacian == "eNeighborhood":
            if self.epsilon[1] == "upper":
                L = laplacian(
                    X,
                    similarity=self.metric,
                    gamma=self.gamma,
                    norm=self.normalize,
                    mode="eNeighbour",
                    upper=self.epsilon[0],
                )
            elif self.epsilon[1] == "lower":
                L = laplacian(
                    X,
                    similarity=self.metric,
                    gamma=self.gamma,
                    norm=self.normalize,
                    mode="eNeighbour",
                    lower=self.epsilon[0],
                )
            else:
                raise ValueError(
                    "Boundary needs to be 'upper' or 'lower' and threshold needs to be set, if laplacian = eNeighborhood"
                )

        elif self.laplacian == "fully_connected":
            L = laplacian(
                X, similarity=self.metric, gamma=self.gamma, norm=self.normalize, mode="fc"
            )
        else:
            raise NotImplementedError("Other approaches currently not implemented")

        # 3. Eigenvalue and -vector calculation via Lanczos Algorithm
        v0 = ht.ones((L.shape[0],), dtype=L.dtype, split=L.split, device=L.device) / math.sqrt(
            L.shape[0]
        )

        V, T = ht.lanczos(L, self.n_lanczos, v0)
        # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
        eval, evec = torch.eig(T._DNDarray__array, eigenvectors=True)
        # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
        eval, idx = torch.sort(eval[:, 0], dim=0)
        eigenvalues = ht.array(eval)
        eigenvectors = ht.matmul(V, ht.array(evec))[:, idx]

        # 5. Find the spectral gap, if number of clusters is not defined from the outside
        if self.n_clusters is None:
            diff = eigenvalues[1:] - eigenvalues[:-1]
            self.n_clusters = np.where(diff == diff.max())[0][0] + 1
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

        X is transformed to the low-dim representation by calculation of eigenspectrum (eigenvalues and eigenvectors) of the graph laplacian from the similarity matrix.
        Inference of lables is done by extraction of the closest centroid of the n_clusters eigenvectors from the previously fitted clustering algorithm (kmeans)
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
        # 2. Construct Laplacian
        if self.laplacian == "eNeighborhood":
            if self.epsilon[1] == "upper":
                L = laplacian(
                    X,
                    similarity=self.metric,
                    gamma=self.gamma,
                    norm=self.normalize,
                    mode="eNeighbour",
                    upper=self.epsilon[0],
                )
            elif self.epsilon[1] == "lower":
                L = laplacian(
                    X,
                    similarity=self.metric,
                    gamma=self.gamma,
                    norm=self.normalize,
                    mode="eNeighbour",
                    lower=self.epsilon[0],
                )
            else:
                raise ValueError(
                    "Boundary needs to be 'upper' or 'lower' and threshold needs to be set, if laplacian = eNeighborhood"
                )

        elif self.laplacian == "fully_connected":
            L = laplacian(
                X, similarity=self.metric, gamma=self.gamma, norm=self.normalize, mode="fc"
            )
        else:
            raise NotImplementedError("Other approaches currently not implemented")

        # 3. Eigenvalue and -vector calculation via Lanczos Algorithm
        v0 = ht.ones((L.shape[0],), dtype=L.dtype, split=L.split, device=L.device) / math.sqrt(
            L.shape[0]
        )

        V, T = ht.lanczos(L, self.n_lanczos, v0)
        # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
        eval, evec = torch.eig(T._DNDarray__array, eigenvectors=True)
        # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
        eval, idx = torch.sort(eval[:, 0], dim=0)
        eigenvectors = ht.matmul(V, ht.array(evec))[:, idx]

        components = eigenvectors[:, : self.n_clusters].copy()

        return self._cluster.predict(components)

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        This method should be preferred to to calling fit(X) followed by predict(X), since predict(X) requires recomputation of the low-dim eigenspectrum representation of X

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]
            Input data to be clustered.

        Returns
        -------
        labels : ht.DNDarray, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self._labels
