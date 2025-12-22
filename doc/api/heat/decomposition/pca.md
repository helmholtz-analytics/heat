Module heat.decomposition.pca
=============================
Module implementing decomposition techniques, such as PCA.

Classes
-------

`IncrementalPCA(n_components: int | None = None, copy: bool = True, whiten: bool = False, batch_size: int | None = None)`
:   Incremental Principal Component Analysis (PCA).

    This class allows for incremental updates of the PCA model. This is especially useful for large data sets that do not fit into memory.

    An example how to apply this class is given in, e.g., `benchmarks/cb/decomposition.py`.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If `n_components` is not set all components are kept (default).
    copy : bool, default=True
        In-place operations are not yet supported. Please set `copy=True`.
    whiten : bool, default=False
        Not yet supported.
    batch_size : int, optional
        Currently not needed and only added for API consistency and possible future extensions.

    Attributes
    ----------
    components_ : DNDarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by `explained_variance_.
    singular_values_ : DNDarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
    mean_ : DNDarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    n_components_ : int
        The estimated number of components.
    n_samples_seen_ : int
        Number of samples processed so far.

    ### Ancestors (in MRO)

    * heat.core.base.TransformMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X, y=None) ‑> Self`
    :   Not yet implemented; please use `.partial_fit` instead.
        Please open an issue on GitHub if you would like to see this method implemented and make a suggestion on how you would like to see it implemented.

    `inverse_transform(self, X: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Transform data back to its original space.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_components)
            Data set to be transformed back.

    `partial_fit(self, X: heat.core.dndarray.DNDarray, y=None)`
    :   One single step of incrementally building up the PCA.
        Input X is the current batch of data that needs to be added to the existing PCA.

    `transform(self, X: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Apply dimensionality based on PCA to X.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_features)
            Data set to be transformed.

`PCA(n_components: int | float | None = None, copy: bool = True, whiten: bool = False, svd_solver: str = 'hierarchical', tol: float | None = None, iterated_power: int | str = 0, n_oversamples: int = 10, power_iteration_normalizer: str = 'qr', random_state: int | None = None)`
:   Pricipal Component Analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
    The input data is centered but not scaled for each feature before applying the SVD.

    Parameters
    ----------
    n_components : int, float, None, default=None
        Number of components to keep. If n_components is not set all components are kept.
        If n_components is an integer, it specifies the number of components to keep.
        If n_components is a float between 0 and 1, it specifies the fraction of variance explained by the components to keep.
    copy : bool, default=True
        In-place operations are not yet supported. Please set copy=True.
    whiten : bool, default=False
        Not yet supported.
    svd_solver : {'full', 'hierarchical'}, default='hierarchical'
        'full' : Full SVD is performed. In general, this is more accurate, but also slower. So far, this is only supported for tall-skinny or short-fat data.
        'hierarchical' : Hierarchical SVD, i.e., an algorithm for computing an approximate, truncated SVD, is performed. Only available for data split along axis no. 0.
        'randomized' : Randomized SVD is performed.
    tol : float, default=None
        Not yet necessary as iterative methods for PCA are not yet implemented.
    iterated_power :  int, default=0
        if svd_solver='randomized', this parameter is the number of iterations for the power method.
        Choosing `iterated_power > 0` can lead to better results in the case of slowly decaying singular values but is computationally more expensive.
    n_oversamples : int, default=10
        if svd_solver='randomized', this parameter is the number of additional random vectors to sample the range of X so that the range of X can be approximated more accurately.
    power_iteration_normalizer : {'qr'}, default='qr'
        if svd_solver='randomized', this parameter is the normalization form of the iterated power method. So far, only QR is supported.
    random_state : int, default=None
        if svd_solver='randomized', this parameter allows to set the seed for the random number generator.

    Attributes
    ----------
    components_ : DNDarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.
    explained_variance_ : DNDarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
        Not supported by svd_solver='hierarchical' and svd_solver='randomized'.
    explained_variance_ratio_ : DNDarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        Not supported by svd_solver='hierarchical' and svd_solver='randomized'.
    total_explained_variance_ratio_ : float
        The percentage of total variance explained by the selected components together.
        For svd_solver='hierarchical', an lower estimate for this quantity is provided; see :func:`ht.linalg.hsvd_rtol` and :func:`ht.linalg.hsvd_rank` for details.
        Not supported by svd_solver='randomized'.
    singular_values_ : DNDarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        Not supported by svd_solver='hierarchical' and svd_solver='randomized'.
    mean_ : DNDarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    n_components_ : int
        The estimated number of components.
    n_samples_ : int
        Number of samples in the training data.
    noise_variance_ : float
        not yet implemented

    Notes
    -----
    Hierarchical SVD (`svd_solver = "hierarchical"`) computes an approximate, truncated SVD. Thus, the results are not exact, in general, unless the
    truncation rank chosen is larger than the actual rank (matrix rank) of the underlying data; see :func:`ht.linalg.hsvd_rank` and :func:`ht.linalg.hsvd_rtol` for details.
    Randomized SVD (`svd_solver = "randomized"`) is a stochastic algorithm that computes an approximate, truncated SVD.

    ### Ancestors (in MRO)

    * heat.core.base.TransformMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray, y=None) ‑> Self`
    :   Fit the PCA model with data X.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_features)
            Data set of which PCA has to be computed.
        y : Ignored
            Not used, present for API consistency by convention.

    `inverse_transform(self, X: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Transform data back to its original space.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_components)
            Data set to be transformed back.

    `transform(self, X: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Apply dimensionality based on PCA to X.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_features)
            Data set to be transformed.
