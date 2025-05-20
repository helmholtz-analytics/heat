"""
Module for Spectral Clustering, a graph-based machine learning algorithm
"""

import heat as ht
import math
import torch
from typing import Tuple, Union
from heat.core.dndarray import DNDarray
from heat.core.linalg import eigh


class SpectralClustering(ht.ClusteringMixin, ht.BaseEstimator):
    """
    Spectral clustering of large memory-distributed datasets.

    Attributes
    ----------
    n_clusters : int, default=8
        Number of clusters to fit
    eigen_solver : str, default='lanczos'
        Eigenvalue decomposition strategy to use. Supported: 'lanczos', 'zolotarev'.
    n_components : int, default=None
        Number of components to use for the embedding. If None, n_clusters is used
    gamma : float, default=1.0
        Kernel coefficient sigma for 'rbf', ignored for affinity='euclidean'
    affinity : str, default='rbf'
        How to construct the similarity (affinity) matrix.

            - 'rbf' : construct the similarity matrix using a radial basis function (RBF) kernel.
            - 'euclidean' : construct the similarity matrix as only euclidean distance.
            - 'precomputed' : interpret ``X`` as precomputed affinity matrix.
    laplacian : str, default='fully_connected'
        How to calculate the graph laplacian (affinity)
        Currently supported : 'fully_connected', 'eNeighbour'
    threshold : float
        Threshold for affinity matrix if laplacian='eNeighbour'
        Ignored for laplacian='fully_connected'
    boundary : str
        How to interpret threshold: 'upper', 'lower'
        Ignorded for laplacian='fully_connected'
    n_lanczos : int, default=300
        number of Lanczos iterations for Eigenvalue decomposition
    assign_labels: str, default='kmeans'
         The strategy to use to assign labels in the embedding space.
    **params: dict
          Parameter dictionary for the assign_labels estimator
    """

    def __init__(
        self,
        n_clusters: int = None,
        eigen_solver: str = "lanczos",
        n_components: int = None,
        gamma: float = 1.0,
        affinity: str = "rbf",
        laplacian: str = "fully_connected",
        threshold: float = 1.0,
        boundary: str = "upper",
        n_lanczos: int = 300,
        assign_labels: str = "kmeans",
        **params,
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.gamma = gamma
        self.affinity = affinity
        self.laplacian = laplacian
        self.threshold = threshold
        self.boundary = boundary
        self.n_lanczos = n_lanczos
        self.assign_labels = assign_labels

        if affinity == "rbf":
            sig = math.sqrt(1 / (2 * gamma))
            self._laplacian = ht.graph.Laplacian(
                lambda x: ht.spatial.rbf(x, sigma=sig, quadratic_expansion=True),
                definition="norm_sym",
                mode=laplacian,
                threshold_key=boundary,
                threshold_value=threshold,
            )

        elif affinity == "euclidean":
            self._laplacian = ht.graph.Laplacian(
                lambda x: ht.spatial.cdist(x, quadratic_expansion=True),
                definition="norm_sym",
                mode=laplacian,
                threshold_key=boundary,
                threshold_value=threshold,
            )
        elif affinity == "precomputed":
            self._laplacian = ht.graph.Laplacian(
                lambda x: x,
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
    def labels_(self) -> DNDarray:
        """
        Returns labels of each point.
        """
        return self._labels

    def __set_diag(self, x: DNDarray, value: Union[int, float], norm_laplacian: bool) -> DNDarray:
        """
        Set the diagonal of the matrix to a specific value.
        """
        # TODO see scikit learn
        n_nodes = x.shape[0]
        # for now we assume that the matrix is dense
        if norm_laplacian:
            # set diagonal to `value`
            # tensor.view(-1) == ndarray.flat
            x.larray.view(-1)[:: n_nodes + 1] = value
        return x

    def spectral_embedding(
        self,
        x: DNDarray,
        n_components: int = 8,
        eigen_solver: str = "zolotarev",
        norm_laplacian: bool = True,
        drop_first: bool = True,
    ) -> Tuple[DNDarray, DNDarray]:
        """
        Returns Tuple(Eigenvalues, Eigenvectors) of the graph's Laplacian matrix.

        Parameters
        ----------
        x : DNDarray
            Sample Matrix for which the embedding should be calculated
        n_components : int, default=8
            Number of components to use for the embedding
        eigen_solver : str
            Eigenvalue decomposition strategy to use. Default is 'zolotarev' #TODO expand
        norm_laplacian : bool, default=True
            Whether to use the normalized Laplacian
        drop_first : bool, default=True
            Whether to drop the first component (the smallest eigenvalue)

        See Also
        --------
        :func:`heat.linalg.lanczos`
        :func:`heat.linalg.eigh`
        :func:`heat.linalg.polar`

        Notes
        -----
        The imaginary part of the eigenvalues is discarded, as the Laplacian matrix is symmetric and the eigenvectors
        are real. TODO: check if this is still correct
        """
        L = self._laplacian.construct(x)
        # 3. Eigenvalue and -vector calculation
        if eigen_solver == "zolotarev":
            L = self.__set_diag(L, 1, norm_laplacian)  # TODO should it be a method?
            # extract the diagonal
            dd = L.diagonal()
            #            try:
            # we skip multiplying by -1 as we are not using ARPACK
            # keeping comment below as historical record for now
            ####
            # # We are computing the opposite of the laplacian inplace so as
            # # to spare a memory allocation of a possibly very large array
            # L *= -1
            ####
            # compute the Zolo-PD eigenvalue decomposition
            _, diffusion_map = eigh(L)
            # ht.linalg.eigh returns the eigenvalues in descending order
            # select the first n_components, no need to reverse order
            diffusion_map = diffusion_map[:, :n_components]
            embedding = diffusion_map.T
            if norm_laplacian:
                # REVERT FROM DIVISION TO MULTIPLICATION
                # (user requirement, see https://codebase.helmholtz.cloud/helmholtz-analytics/SCIMES/-/blob/master/scimes/old_spectral_embedding.py?ref_type=heads#L62)
                # TODO: generalize / adapt to current scikit-learn
                embedding = embedding * dd
            # except RuntimeError:
            #     # When submatrices are exactly singular, an LU decomposition
            #     # in arpack fails. We fallback to lobpcg
            #     eigen_solver = "lobpcg"
            #     # Revert the laplacian to its opposite to have lobpcg work
            #     L *= -1

        #  Lanczos Algorithm
        elif eigen_solver == "lanczos":
            v0 = ht.full(
                (L.shape[0],),
                fill_value=1.0 / math.sqrt(L.shape[0]),
                dtype=L.dtype,
                split=0,
                device=L.device,
            )
            V, T = ht.lanczos(L, self.n_lanczos, v0)

            # 4. Calculate and Sort Eigenvalues and Eigenvectors of tridiagonal matrix T
            eval, evec = torch.linalg.eig(T.larray)

            # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
            eval, idx = torch.sort(eval.real, dim=0)
            eigenvalues = ht.array(eval)
            eigenvectors = ht.matmul(V, ht.array(evec))[:, idx]
            # TODO: spectral_embedding should only return the embedding (eigenvectors)
            # TODO: move components calculation from fit() to here and only return n_components eigenvalues as embedding
            return eigenvalues.real, eigenvectors.real
        else:
            raise NotImplementedError(
                "Other Eigenvalue Decomposition methods are not yet supported"
            )

    def fit(self, x: DNDarray):
        """
        Clusters dataset X via spectral embedding.
        Computes the low-dim representation by calculation of eigenspectrum (eigenvalues and eigenvectors) of the graph
        laplacian from the similarity matrix and fits the eigenvectors that correspond to the k lowest eigenvalues with
        a seperate clustering algorithm (currently only kmeans is supported). Similarity metrics for adjacency
        calculations are supported via spatial.distance. The eigenvalues and eigenvectors are computed by reducing the
        Laplacian via lanczos iterations and using the torch eigenvalue solver on this smaller matrix. If other
        eigenvalue decompostion methods are supported, this will be expanded.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)
        """
        # 1. input sanitation
        if not isinstance(x, DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")
        if x.split is not None and x.split != 0:
            raise NotImplementedError("Not implemented for other splitting-axes")
        # 2. Embed Dataset into lower-dimensional Eigenvector space
        eigenvalues, eigenvectors = self.spectral_embedding(x, self.eigen_solver)

        # 3. Find the spectral gap, if number of clusters is not defined from the outside
        if self.n_clusters is None:
            diff = eigenvalues[1:] - eigenvalues[:-1]
            tmp = ht.argmax(diff).item()
            self.n_clusters = tmp + 1

        components = eigenvectors[:, : self.n_clusters].copy()

        params = self._cluster.get_params()
        params["n_clusters"] = self.n_clusters
        self._cluster.set_params(**params)
        self._cluster.fit(components)
        self._labels = self._cluster.labels_
        self._cluster_centers = self._cluster.cluster_centers_

        return self

    def fit_predict(self, x: DNDarray) -> DNDarray:
        """
        Fit the model to the data and return the labels.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        Returns
        -------
        DNDarray
            Labels of each point.
        """
        self.fit(x)
        return self.labels_
        # return self._cluster.predict(components)

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Return the label each sample in X belongs to.
        X is transformed to the low-dim representation by calculation of eigenspectrum (eigenvalues and eigenvectors) of
        the graph laplacian from the similarity matrix. Inference of lables is done by extraction of the closest
        centroid of the n_clusters eigenvectors from the previously fitted clustering algorithm (kmeans).

        Parameters
        ----------
        x : DNDarray
            New data to predict. Shape = (n_samples, n_features)

        Warning
        -------
        Caution: Calculation of the low-dim representation requires some time!

        """
        # input sanitation
        if not isinstance(x, DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")
        if x.split is not None and x.split != 0:
            raise NotImplementedError("Not implemented for other splitting-axes")

        _, eigenvectors = self.spectral_embedding(x, self.eigen_solver)

        components = eigenvectors[:, : self.n_clusters].copy()

        return self._cluster.predict(components)
