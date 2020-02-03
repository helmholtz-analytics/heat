import torch
import numpy as np

from .. import factories
from .. import manipulations
from .. import types
from .. import operations
from ..spatial import distances
from .. import linalg
from .. import random
from . import KMeans


def laplacian(S, norm=True, mode="fc", upper=None, lower=None):
    if mode == "eNeighbour":
        if (upper is not None) and (lower is not None):
            raise ValueError(
                "Epsilon neighborhood with either upper or lower threshold, both is not supported"
            )

        if upper is not None:
            A = types.int(S < upper) - factories.eye(S.shape, dtype=types.int, split=S.split)
        elif lower is not None:
            A = types.int(S > upper) - factories.eye(S.shape, dtype=types.int, split=S.split)
    elif mode == "fc":
        A = S
    else:
        raise NotImplementedError(
            "Only eNeighborhood and fully-connected graphs supported at the moment."
        )
    if norm:
        degree = operations.sqrt(1.0 / operations.sum(A, axis=1))
    else:
        degree = sum(A, axis=1)

    D = manipulations.diag(degree, dtype=S.dtype, split=S.split, device=S.device, comm=S.comm)

    if norm:
        L = factories.eye(A.shape, split=A.split) - linalg.matmul(D, linalg.matmul(S, D))
    else:
        L = D - A

    return L


class spectral:
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

        """
        self.n_clusters = n_clusters
        self.n_lanczos = n_lanczos
        self.metric = metric
        self.gamma = gamma
        self.normalize = normalize
        self.laplacian = laplacian
        self.epsilon = (threshold, boundary)

        # in-place properties
        self._labels = None
        self._similarity = None
        self._eigenvectors = None
        self._eigenvalues = None

    def fit(self, X):
        # 1. Calculation of Adjacency Matrix
        if self.kernel == "rbf":
            if self.sigma is None:
                raise ValueError(
                    "For usage of RBF kernel please specify the scaling factor sigma in class instantiation"
                )
            self._similarity = distances.similarity(
                X, lambda x, y: distances._gaussian(x, y, self.sigma)
            )
        elif self.kernel == "euclidean":
            self._similarity = distances.similarity(X, lambda x, y: distances._euclidian(x, y))
        else:
            raise NotImplementedError("Other kernels currently not implemented")

        # 2. Calculation of Laplacian
        if self.laplacian == "eNeighborhood":
            if self.epsilon[1] == "upper":
                L = laplacian(
                    self._similarity, norm=self.normalize, mode="eNeighbour", upper=self.epsilon[0]
                )
            elif self.epsilon[1] == "lower":
                L = laplacian(
                    self._similarity, norm=self.normalize, mode="eNeighbour", lower=self.epsilon[0]
                )
            else:
                raise ValueError(
                    "Boundary needs to be 'upper' or 'lower' and threshold needs to be set, if laplacian = eNeighborhood"
                )

        elif self.laplacian == "fully_connected":
            L = laplacian(self._similarity, norm=self.normalize, mode="fc")
        else:
            raise NotImplementedError("Other approaches currently not implemented")

        # 3. Eigenvalue and -vector Calculation
        vr = random.rand(L.shape[0], split=L.split, dtype=types.float64)
        v0 = vr / linalg.norm(vr)
        Vg_norm, Tg_norm = linalg.lanczos(L, self.n_lanczos, v0)

        # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
        eval, evec = torch.eig(Tg_norm._DNDarray__array, eigenvectors=True)
        # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
        self._eigenvectors = linalg.matmul(Vg_norm, factories.array(evec))
        self._eigenvalues, idx = torch.sort(eval[:, 0], dim=0)
        self._eigenvalues = self._eigenvectors[:, idx]

        if self.n_clusters is None:
            temp = np.diff(self._eigenvalues.numpy())
            self.n_clusters = np.where(temp == temp.max())[0][0] + 1

        components = self._eigenvectors[:, : self.n_clusters].copy()

        kmeans = KMeans(n_clusters=self.n_clusters, init="kmeans++")
        kmeans.fit(components)
        # cluster = kmeans.predict(self.n_clusters)
        # centroids = kmeans.cluster_centers_

        return
