import torch
import math
import numpy as np
import heat as ht


def laplacian(S, norm=True, mode="fc", upper=None, lower=None):
    """
    Calculate the graph Laplacian from a similarity matrix

    Parameters
    ----------
    S : ht.DNDarray
        quadrdatic, positive semidefinite similarity matrix, encoding similarity metrices s_ij between data samples i and j
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

    if mode == "eNeighbour":
        if (upper is not None) and (lower is not None):
            raise ValueError(
                "Epsilon neighborhood with either upper or lower threshold, both is not supported"
            )
        if (upper is None) and (lower is None):
            raise ValueError("Epsilon neighborhood requires upper or lower threshold to be defined")
        if upper is not None:
            A = ht.int(S < upper)
            A = A - ht.eye(S.shape, dtype=ht.int, split=S.split, device=S.device, comm=S.comm)
        elif lower is not None:
            print("Here")
            A = ht.int(S > lower)
            A = A - ht.eye(S.shape, dtype=ht.int, split=S.split, device=S.device, comm=S.comm)
    elif mode == "fc":
        A = S
    else:
        raise NotImplementedError(
            "Only eNeighborhood and fully-connected graphs supported at the moment."
        )
    if norm:
        degree = ht.sqrt(1.0 / ht.sum(A, axis=1))
    else:
        degree = ht.sum(A, axis=1)

    D = ht.diag(degree)

    if norm:
        L = ht.eye(A.shape, split=A.split, device=A.device, comm=A.comm) - ht.matmul(
            D, ht.matmul(S, D)
        )
    else:
        L = D - A

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
            ‘rbf’ : construct the similarity matrix using a radial basis function (RBF) kernel.
            'euclidean' : construct the similarity matrix as only euclidean distance
            ‘precomputed’ : interpret X as a precomputed similarity matrix.
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
            self._cluster = ht.cluster.KMeans(init="kmeans++")
        else:
            raise NotImplementedError(
                "Other Label Assignment Algorithms are currently not available"
            )

        # in-place properties
        self._eigenvectors = None
        self._eigenvalues = None
        self._labels = None
        self._similarity = None

    @property
    def eigenvectors_(self):
        """
        Returns
        -------
        ht.DNDarray, shape = (n_samples, n_lanzcos):
            Eigenvectors of the graph Laplacian
        """
        return self._eigenvectors

    @property
    def eigenvalues_(self):
        """
        Returns
        -------
        ht.DNDarray, shape = (n_lanzcos):
            Eigenvalues of the graph Laplacian
        """
        return self._eigenvalues

    @property
    def labels_(self):
        """
        Returns
        -------
        ht.DNDarray, shape=(n_points):
            Labels of each point.
        """
        return self._labels

    @property
    def similarity_(self):
        """
        Returns
        -------
        ht.DNDarray:
            The similarity matrix of the fitted dataset
        """
        return self._similarity

    def _calc_adjacency(self, X):
        """
        Internal utility function for calculating of the similarity matrix according to the configuration set in __init__

        Parameters
        ----------
            X :  ht.DNDarray, shape=(n_samples, n_features)

        Returns
        -------
            S : ht.DNDarray, shape = (n_samples, n_samples)
            The similarity matrix

        """
        if self.metric == "rbf":
            sig = math.sqrt(1 / (2 * self.gamma))
            s = ht.spatial.rbf(X, sigma=sig)

        elif self.metric == "euclidean":
            s = ht.spatial.cdist(X)
        else:
            raise NotImplementedError("Other kernels currently not implemented")
        return s

    def _calc_laplacian(self, S):
        """
        Internal utility function for calculating of the graph Laplacian according to the configuration set in __init__

        Parameters
        ----------
        S : ht.DNDarray
            The similarity matrix

        Returns
        -------
        ht.DNDarray

        """

        if self.laplacian == "eNeighborhood":
            if self.epsilon[1] == "upper":
                return laplacian(S, norm=self.normalize, mode="eNeighbour", upper=self.epsilon[0])
            elif self.epsilon[1] == "lower":
                return laplacian(S, norm=self.normalize, mode="eNeighbour", lower=self.epsilon[0])
            else:
                raise ValueError(
                    "Boundary needs to be 'upper' or 'lower' and threshold needs to be set, if laplacian = eNeighborhood"
                )

        elif self.laplacian == "fully_connected":
            return laplacian(self._similarity, norm=self.normalize, mode="fc")
        else:
            raise NotImplementedError("Other approaches currently not implemented")

    def _calculate_eigendecomposition(self, L):
        """
        Internal utility function for calculating eigenvalues and eigenvectors of the graph Laplacian using the Lanczos algorithm

        Parameters
        ----------
        L : ht.DNDarray
            The graph Laplacian matrix

        Returns
        -------
        eigenvalues : ht.DNDarray, shape = (n_lanczos,)
        eigenvectors: ht.DNDarray, shape = (n_samples, n_lanczos)

        """
        vr = ht.random.rand(L.shape[0], split=L.split, dtype=ht.float32)
        v0 = vr / ht.norm(vr)
        Vg_norm, Tg_norm = ht.lanczos(L, self.n_lanczos, v0)

        # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
        eval, evec = torch.eig(Tg_norm._DNDarray__array, eigenvectors=True)
        # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
        eigenvalues, idx = torch.sort(eval[:, 0], dim=0)
        eigenvalues = ht.array(eigenvalues)
        eigenvectors = ht.matmul(Vg_norm, ht.array(evec))[:, idx]

        return eigenvalues, eigenvectors

    def fit(self, X):
        """
        Computes the low-dim representation by calculation of eigenspectrum (eigenvalues and eigenvectors) of the graph laplacian from the similarity matrix and fits the eigenvectors that correspond to the k lowest eigenvalues with a seperate clustering algorithm (currently only kemans is supported)
        Similarity metrics for adjacency calculations are supported via spatial.distance. The eigenvalues and eigenvectors are computed by reducing the Laplacian via lanczos iterations and using the torch eigenvalue solver on this smaller matrix. If other eigenvalue decompostion methods are supported, this will be expanded.


        Parameters
        ----------
        X : ht.DNDarray, shape=(n_samples, n_features)
            Training instances to cluster.
        """
        # input sanitation
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))
        if X.split is not None and X.split != 0:
            raise NotImplementedError("Not implemented for other splitting-axes")

        # 1. Calculation of Adjacency Matrix
        self._similarity = self._calc_adjacency(X)

        # 2. Calculation of Laplacian
        L = self._calc_laplacian(self._similarity)

        # 3. Eigenvalue and -vector calculation via Lanczos Algorithm
        self._eigenvalues, self._eigenvectors = self._calculate_eigendecomposition(L)

        # 5. Find the spectral gap, if number of clusters is not defined from the outside
        if self.n_clusters is None:
            temp = np.diff(self._eigenvalues.numpy())
            self.n_clusters = np.where(temp == temp.max())[0][0] + 1
        components = self._eigenvectors[:, : self.n_clusters].copy()

        params = self._cluster.get_params()
        params["n_clusters"] = self.n_clusters
        self._cluster.set_params(**params)
        self._cluster.fit(components)
        self._labels = self._cluster.labels_

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
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))

        # 1. Calculation of Adjacency Matrix
        S = self._calc_adjacency(X)

        # 2. Calculation of Laplacian
        L = self._calc_laplacian(S)

        # 3. Eigenvalue and -vector calculation via Lanczos Algorithm
        eval, evec = self._calculate_eigendecomposition(L)

        return self._cluster.predict(evec[:, : self.n_clusters].copy())

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
