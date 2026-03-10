"""
Module for Spectral Clustering, a graph-based machine learning algorithm
"""

import heat as ht
import math
import torch
from typing import Tuple, Union
from heat.core.dndarray import DNDarray
from heat.core.linalg import reigh


class SpectralClustering(ht.ClusteringMixin, ht.BaseEstimator):
    """
    Spectral clustering of large memory-distributed arrays.

    Attributes
    ----------
    n_clusters : int, optional
        Number of clusters to fit
    eigen_solver : str, default='randomized'
        The eigenvalue decomposition strategy to use.
            - 'randomized' : Use a randomized algorithm to compute the approximate eigenvalues and eigenvectors.
            - 'lanczos' : Use Lanczos iterations to reduce the Laplacian matrix size before applying the torch eigenvalue solver.
    n_components : int, optional
        Number of components to use for the embedding. If None, it is set to ``n_clusters``
    random_state : int, optional
        Random seed for reproducibility. If None, no random seed is set.
    gamma : float, default=1.0
        Kernel coefficient sigma for 'rbf', ignored for affinity='euclidean'
    affinity : str, default='rbf'
        How to construct the similarity (affinity) matrix.
            - 'rbf' : construct the similarity matrix using a radial basis function (RBF) kernel.
            - 'euclidean' : construct the similarity matrix as euclidean distance.
            - 'precomputed' : interpret ``X`` as precomputed affinity matrix.
    laplacian : str, default='fully_connected'
        How to calculate the graph laplacian (affinity)
        Currently supported : 'fully_connected', 'eNeighbour'
    threshold : float
        Threshold for affinity matrix if laplacian='eNeighbour'
        Ignored for laplacian='fully_connected'
    boundary : str
        How to interpret threshold: 'upper', 'lower'
        Ignored for laplacian='fully_connected'
    reigh_rank : int, default: 100
        number of samples for randomized eigenvalue decomposition. Only used if eigen_solver='randomized'.
        It must hold :math:`reigh_rank >= n_clusters`. If ``n_clusters`` is None (automatic selection of number of clusters),
        ``reigh_rank`` gives an upper bound on the number of clusters that can be found. Therefore, reigh_rank should
        be set high enough to capture the expected number of clusters in that case.
    reigh_n_oversamples : int, default: 10
        number of oversamples for randomized eigenvalue decomposition. Only used if ``eigen_solver``='randomized'. Default is 10.
    reigh_power_iter : int, default: 0
        number of power iterations for randomized eigenvalue decomposition. Only used if eigen_solver='randomized'.
        Consider increasing this value if the eigen-spectrum of the Laplacian decays slowly.
    lanczos_n_iter : int, default: 300
        number of Lanczos iterations for Eigenvalue decomposition. Only used if eigen_solver='lanczos'.
    assign_labels: str, default='kmeans'
         The strategy to use to assign labels in the embedding space.
    **params: dict
          Parameter dictionary for the assign_labels estimator
    """

    def __init__(
        self,
        n_clusters: Union[int, None] = None,
        eigen_solver: str = "randomized",
        n_components: Union[int, None] = None,
        random_state: Union[int, None] = None,
        gamma: float = 1.0,
        affinity: str = "rbf",
        laplacian: str = "fully_connected",
        threshold: float = 1.0,
        boundary: str = "upper",
        reigh_rank: int = 100,
        reigh_n_oversamples: int = 10,
        reigh_power_iter: int = 0,
        lanczos_n_iter: int = 300,
        assign_labels: str = "kmeans",
        **params,
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components if n_components is not None else n_clusters
        self.random_state = random_state
        if self.random_state is not None:
            ht.random.seed(self.random_state)
        self.gamma = gamma
        self.affinity = affinity
        self.laplacian = laplacian
        self.threshold = threshold
        self.boundary = boundary
        self.lanczos_n_iter = lanczos_n_iter
        self.assign_labels = assign_labels
        self.eigen_solver = eigen_solver
        self.reigh_n_oversamples = reigh_n_oversamples
        self.reigh_power_iter = reigh_power_iter
        self.reigh_rank = reigh_rank

        if eigen_solver not in ["lanczos", "randomized"]:
            raise NotImplementedError(
                f"Currently only 'lanczos' and 'randomized' eigen_solver are supported, but got '{eigen_solver}' as input."
            )
        if eigen_solver == "randomized" and reigh_rank < (
            n_clusters if n_clusters is not None else 1
        ):
            raise ValueError("reigh_rank must be at least equal to n_clusters")

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
        Modified for dense PyTorch tensors from scikit-learn.manifold._spectral_embedding._set_diag
        https://github.com/scikit-learn/scikit-learn/blob/031d2f83b7c9d1027d1477abb2bf34652621d603/sklearn/manifold/_spectral_embedding.py#L108
        """
        n_nodes = x.shape[0]
        # we assume that the matrix is dense
        if norm_laplacian:
            # set diagonal to `value`
            # NB: tensor.view(-1) == ndarray.flat
            x.larray.view(-1)[:: n_nodes + 1] = value
        return x

    def __spectral_embedding(
        self,
        x: DNDarray,
        n_components: Union[int, None] = None,
        eigen_solver: str = "randomized",
        norm_laplacian: bool = True,
        drop_first: bool = True,
    ) -> DNDarray:
        """
        Returns the embedding (eigenvectors) of the graph's Laplacian matrix.

        Parameters
        ----------
        x : DNDarray
            Sample Matrix for which the embedding should be calculated
        n_components : int, optional
            Number of components to use for the embedding. If ``n_components`` is None, it will be set to ``SpectralClustering.n_clusters``.
        eigen_solver : str, default: `randomized`
            Eigenvalue decomposition strategy to use.
        norm_laplacian : bool, default=True
            Whether to use the normalized Laplacian
        drop_first : bool, default=True
            Whether to drop the the smallest eigenvalue and corresponding eigenvector.

        See Also
        --------
        :func:`heat.linalg.lanczos`
        :func:`heat.linalg.reigh`
        """
        L = self._laplacian.construct(x)

        # After sklearn.manifold._spectral_embedding.py
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_spectral_embedding.py#L295

        # Eigenvalue and -vector calculation
        if eigen_solver == "randomized":
            L = self.__set_diag(L, 1, norm_laplacian)
            # extract the diagonal
            dd = L.diagonal()

            # assess n_components = n_clusters if not specified by user
            if self.n_clusters is None:
                # set n_clusters to spectral gap
                diff = eval[1:] - eval[:-1]
                tmp = ht.argmax(diff).item()
                self.n_clusters = tmp + 1
            if n_components is None:
                n_components = self.n_clusters
            if drop_first:
                n_components += 1

            # compute the randomized eigenvalue decomposition
            # NB:  ht.linalg.reigh returns the eigenvalues in descending order
            eval, evec = reigh(
                L,
                n_eigenvalues=n_components,
                n_oversamples=self.reigh_n_oversamples,
                power_iter=self.reigh_power_iter,
            )

            # select the largest n_components
            embedding = evec.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
            # Drop smallest component if requested
            if drop_first:
                embedding = embedding[:-1]

        #  Lanczos Algorithm
        elif eigen_solver == "lanczos":
            v0 = ht.full(
                (L.shape[0],),
                fill_value=1.0 / math.sqrt(L.shape[0]),
                dtype=L.dtype,
                split=0,
                device=L.device,
            )
            V, T = ht.lanczos(L, self.lanczos_n_iter, v0)

            # Calculate and sort eigenvalues and eigenvectors of tridiagonal matrix T
            # T is non-distributed
            eval, evec = torch.linalg.eig(T.larray)

            # If x is an Eigenvector of T, then y = V@x is the corresponding Eigenvector of L
            # sort local eval, and evec accordingly
            eval, idx = torch.sort(eval.real, dim=0)
            evec = evec[:, idx]
            # Calculate global eigenvectors of L
            eigenvectors = ht.matmul(V, ht.array(evec))
            # transpose to have the eigenvectors as rows
            eigenvectors = eigenvectors.T
            # Set n_clusters to spectral gap, if it is not defined by the user
            if self.n_clusters is None:
                diff = eval[1:] - eval[:-1]
                tmp = torch.argmax(diff).item()
                self.n_clusters = tmp + 1
            # n_components = n_clusters if not specified
            if n_components is None:
                n_components = self.n_clusters
            # the Laplacian is symmetric and the eigenvectors are real
            # TODO: check the returned order of evec, eval here
            embedding = eigenvectors[:n_components].real
            # Drop smallest component if requested
            if drop_first:
                embedding = embedding[1:]
        else:
            raise NotImplementedError(
                f"Eigenvalue decomposition method {eigen_solver} is not supported. Supported methods are 'randomized' and 'lanczos'."
            )
        # Return the embedding transposed back to original shape
        return embedding.T

    def fit(self, x: DNDarray):
        """
        Clusters dataset X via spectral embedding.
        Computes the low-dim representation by calculation of the spectral embedding (eigenvectors corresponding to the lowest `n_clusters` eigenvalues) of the
        graph laplacian computed from the similarity matrix. Similarity metrics for adjacency
        calculations are supported via :func:`heat.spatial.distance`.

        See :func:`__spectral_embedding` for more details on the decomposition of the graph laplacian.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        See Also
        --------
        :func:`__spectral_embedding`
        """
        # input sanitation
        if not isinstance(x, DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")
        if x.is_distributed() and x.split != 0:
            raise NotImplementedError(f"Distribution along axis {x.split} is not supported yet.")
        # Embed dataset into lower-dimensional eigenvector space
        components = self.__spectral_embedding(
            x, n_components=self.n_components, eigen_solver=self.eigen_solver
        )

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

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Return the label each sample in X belongs to.
        X is transformed to the low-dim representation by calculation of the embedding (eigenvectors corresponding to the lowest `n_clusters` eigenvalues) of
        the graph laplacian from the similarity matrix. Inference of labels is done by extraction of the closest
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
        if x.is_distributed() and x.split != 0:
            raise NotImplementedError(f"Distribution along axis {x.split} is not supported yet.")

        # TODO is copy necessary?
        components = self.__spectral_embedding(x, self.eigen_solver).copy()

        return self._cluster.predict(components)
