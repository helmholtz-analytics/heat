Module heat.cluster.spectral
============================
Module for Spectral Clustering, a graph-based machine learning algorithm

Classes
-------

`Spectral(n_clusters: int = None, gamma: float = 1.0, metric: str = 'rbf', laplacian: str = 'fully_connected', threshold: float = 1.0, boundary: str = 'upper', n_lanczos: int = 300, assign_labels: str = 'kmeans', **params)`
:   Spectral clustering

    Attributes
    ----------
    n_clusters : int
        Number of clusters to fit
    gamma : float
        Kernel coefficient sigma for 'rbf', ignored for metric='euclidean'
    metric : string
        How to construct the similarity matrix.

            - 'rbf' : construct the similarity matrix using a radial basis function (RBF) kernel.
            - 'euclidean' : construct the similarity matrix as only euclidean distance.
    laplacian : str
        How to calculate the graph laplacian (affinity)
        Currently supported : 'fully_connected', 'eNeighbour'
    threshold : float
        Threshold for affinity matrix if laplacian='eNeighbour'
        Ignorded for laplacian='fully_connected'
    boundary : str
        How to interpret threshold: 'upper', 'lower'
        Ignorded for laplacian='fully_connected'
    n_lanczos : int
        number of Lanczos iterations for Eigenvalue decomposition
    assign_labels: str
         The strategy to use to assign labels in the embedding space.
    **params: dict
          Parameter dictionary for the assign_labels estimator

    ### Ancestors (in MRO)

    * heat.core.base.ClusteringMixin
    * heat.core.base.BaseEstimator

    ### Instance variables

    `labels_: heat.core.dndarray.DNDarray`
    :   Returns labels of each point.

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray)`
    :   Clusters dataset X via spectral embedding.
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

    `predict(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Return the label each sample in X belongs to.
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
