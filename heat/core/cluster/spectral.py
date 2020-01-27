import torch
import numpy as np

from .. import factories
from .. import types
from .. import operations
from .. import linalg
from .. import metrics
from .. import random
from . import KMeans


def laplacian(X, kernel, adjacency=None, norm=True, mode="fc", upper=None, lower=None):
    if adjacency is not None:
        S = adjacency
    else:
        S = metrics.similarity(X, kernel)
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

    D = factories.diagonal(degree, dtype=S.dtype, split=S.split, device=S.device, comm=S.comm)

    if norm:
        L = factories.eye(A.shape, split=A.split) - linalg.matmul(D, linalg.matmul(S, D))
    else:
        L = D - A

    return L


class spectral:
    def __init__(self, kernel=metrics.GaussianDistance(sigma=1.0), n_clusters=None, iter=300):
        """
        K-Means clustering algorithm. An implementation of Lloyd's algorithm [1].

        Parameters
        ----------
        n_clusters : int, optional, default: 8
            The number of clusters to form as well as the number of centroids to generate.
        iter : int, default: 300
            Number of Lanczos iterations

        """
        self.iter = iter
        self.kernel = kernel
        self.n_clusters = n_clusters

        # in-place properties
        self._labels = None
        self._adjacency = None
        self._eigenvectors = None
        self._eigenvalues = None

    def fit(self, X):
        # 2. Calculation of Laplacian
        L = laplacian(X, self.kernel)

        # 3. Eigenvalue and -vector Calculation
        vr = random.rand(L.shape[0], split=L.split, dtype=types.float64)
        v0 = vr / linalg.norm(vr)
        Vg_norm, Tg_norm = linalg.lanczos(L, self.iter, v0)

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
