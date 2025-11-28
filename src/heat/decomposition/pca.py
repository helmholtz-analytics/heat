"""
Module implementing decomposition techniques, such as PCA.
"""

import heat as ht
from typing import Optional, Tuple, Union
from ..core.linalg.svdtools import _isvd

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

"""
The implementation is heavily inspired by the corresponding routines in scikit-learn (https://scikit-learn.org/stable/modules/decomposition.html).
Please note that sometimes deviations cannot be avoided due to different algorithms and the distributed nature of the heat package.
"""


class PCA(ht.TransformMixin, ht.BaseEstimator):
    """
    Pricipal Component Analysis (PCA).

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
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = "hierarchical",
        tol: Optional[float] = None,
        iterated_power: Union[str, int] = 0,
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "qr",
        random_state: Optional[int] = None,
    ):
        # check correctness of inputs
        if not copy:
            raise NotImplementedError(
                "In-place operations for PCA are not supported at the moment. Please set copy=True."
            )
        if whiten:
            raise NotImplementedError("Whitening is not yet supported. Please set whiten=False.")
        if not (svd_solver == "full" or svd_solver == "hierarchical" or svd_solver == "randomized"):
            raise ValueError(
                "At the moment, only svd_solver='full' (for tall-skinny or short-fat data), svd_solver='hierarchical', and svd_solver='randomized' are supported. \n An implementation of the 'full' option for arbitrarily shaped data is already planned."
            )
        if not isinstance(iterated_power, int):
            raise TypeError(
                "iterated_power must be an integer. The option 'auto' is not yet supported."
            )
        if isinstance(iterated_power, int) and iterated_power < 0:
            raise ValueError("if an integer, iterated_power must be greater or equal to 0.")
        if power_iteration_normalizer != "qr":
            raise ValueError("Only power_iteration_normalizer='qr' is supported yet.")
        if not isinstance(n_oversamples, int) or n_oversamples < 0:
            raise ValueError("n_oversamples must be a non-negative integer.")
        if tol is not None:
            raise ValueError(
                "Argument tol is not yet necessary as iterative methods for PCA are not yet implemented. Please set tol=None."
            )
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(f"random_state must be None or an integer, was {type(random_state)}.")
        if (
            n_components is not None
            and not (isinstance(n_components, int) and n_components >= 1)
            and not (isinstance(n_components, float) and n_components > 0.0 and n_components < 1.0)
        ):
            raise ValueError(
                "n_components must be None, in integer greater or equal to 1 or a float in (0,1). Option 'mle' is not supported at the moment."
            )

        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        if self.random_state is not None:
            # set random seed accordingly
            ht.random.seed(self.random_state)

        # set future attributes to None to initialize those that will not be computed later on with None (e.g., explained_variance_ for svd_solver='hierarchical')
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.total_explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_samples_ = None
        self.noise_variance_ = None

    def fit(self, X: ht.DNDarray, y=None) -> Self:
        """
        Fit the PCA model with data X.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_features)
            Data set of which PCA has to be computed.
        y : Ignored
            Not used, present for API consistency by convention.
        """
        ht.sanitize_in(X)
        if y is not None:
            raise ValueError(
                "Argument y is ignored and just present for API consistency by convention."
            )

        # center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # set n_components
        # note: n_components is an argument passed by the user to the PCA object and encodes the TARGET number of components
        #       n_components_ is an attribute of the PCA object that stores the ACTUAL number of components
        if self.n_components is None:
            self.n_components_ = min(X.shape)
        else:
            self.n_components_ = self.n_components

        # compute SVD via "full" SVD
        if self.svd_solver == "full" or not X.is_distributed():
            _, S, V = ht.linalg.svd(X_centered, full_matrices=False)
            total_variance = (S**2).sum() / (X.shape[0] - 1)
            if not isinstance(self.n_components_, int):
                # truncation w.r.t. prescribed bound on explained variance
                # determine n_components_ accordingly
                explained_variance_threshold = self.n_components_ * total_variance.larray.item()
                explained_variance_cumsum = (S**2).larray.cumsum(0) / (X.shape[0] - 1)
                self.n_components_ = (
                    len(
                        explained_variance_cumsum[
                            explained_variance_cumsum <= explained_variance_threshold
                        ]
                    )
                    + 1
                )
            self.components_ = V[:, : self.n_components_].T
            self.singular_values_ = S[: self.n_components_]
            self.explained_variance_ = (S**2)[: self.n_components_] / (X.shape[0] - 1)
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
            self.total_explained_variance_ratio_ = self.explained_variance_ratio_.sum().item()
        # compute SVD via "hierarchical" SVD
        elif self.svd_solver == "hierarchical":
            if X.split != 0:
                raise ValueError(
                    "PCA with hierarchical SVD is only available for data split along axis 0."
                )
            if isinstance(self.n_components_, float):
                # hierarchical SVD with prescribed upper bound on relative error
                # note: "upper bound on relative error" (hsvd_rtol) is "1 - lower bound" (PCA)
                _, S, V, info = ht.linalg.hsvd_rtol(
                    X_centered, (1 - self.n_components_) ** 0.5, compute_sv=True, safetyshift=0
                )
            else:
                # hierarchical SVD with prescribed, fixed rank
                _, S, V, info = ht.linalg.hsvd_rank(
                    X_centered, self.n_components_, compute_sv=True, safetyshift=0
                )
            self.n_components_ = V.shape[1]
            self.components_ = V.T
            self.total_explained_variance_ratio_ = 1 - info.larray.item() ** 2

        else:
            # compute SVD via "randomized" SVD
            _, S, V = ht.linalg.rsvd(
                X_centered,
                self.n_components_,
                n_oversamples=self.n_oversamples,
                power_iter=self.iterated_power,
            )
            self.components_ = V.T
            self.n_components_ = V.shape[1]

        self.n_samples_ = X.shape[0]
        self.noise_variance_ = None  # not yet implemented

        return self

    def transform(self, X: ht.DNDarray) -> ht.DNDarray:
        """
        Apply dimensionality based on PCA to X.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_features)
            Data set to be transformed.
        """
        ht.sanitize_in(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"X must have the same number of features as the training data. Expected {self.mean_.shape[0]} but got {X.shape[1]}."
            )

        # center data and apply PCA
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, X: ht.DNDarray) -> ht.DNDarray:
        """
        Transform data back to its original space.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_components)
            Data set to be transformed back.
        """
        ht.sanitize_in(X)
        if X.shape[1] != self.n_components_:
            raise ValueError(
                f"Dimension mismatch. Expected input of shape n_points x {self.n_components_} but got {X.shape}."
            )

        return X @ self.components_ + self.mean_


class IncrementalPCA(ht.TransformMixin, ht.BaseEstimator):
    """
    Incremental Principal Component Analysis (PCA).

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
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: bool = True,
        whiten: bool = False,
        batch_size: Optional[int] = None,
    ):
        if not copy:
            raise NotImplementedError(
                "In-place operations for PCA are not supported at the moment. Please set copy=True."
            )
        if whiten:
            raise NotImplementedError("Whitening is not yet supported. Please set whiten=False.")
        if n_components is not None:
            if not isinstance(n_components, int):
                raise TypeError(
                    f"n_components must be None or an integer, but is {type(n_components)}."
                )
            else:
                if n_components < 1:
                    raise ValueError("if an integer, n_components must be greater or equal to 1.")
        self.whiten = whiten
        self.n_components = n_components
        self.batch_size = batch_size
        self.components_ = None
        # self.explained_variance_ = None            # not yet supported
        # self.explained_variance_ratio_ = None      # not yet supported
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.batch_size_ = None
        self.n_samples_seen_ = 0

    def fit(self, X, y=None) -> Self:
        """
        Not yet implemented; please use `.partial_fit` instead.
        Please open an issue on GitHub if you would like to see this method implemented and make a suggestion on how you would like to see it implemented.
        """
        raise NotImplementedError(
            f"You have called IncrementalPCA's `.fit`-method with an argument of type {type(X)}. \n So far, we have only implemented the method `.partial_fit` which performs a single-step update of incremental PCA. \n Please consider using `.partial_fit` for the moment, and open an issue on GitHub in which we can discuss what you would like to see implemented for the `.fit`-method."
        )

    def partial_fit(self, X: ht.DNDarray, y=None):
        """
        One single step of incrementally building up the PCA.
        Input X is the current batch of data that needs to be added to the existing PCA.
        """
        ht.sanitize_in(X)
        if y is not None:
            raise ValueError(
                "Argument y is ignored and just present for API consistency by convention."
            )
        if self.n_samples_seen_ == 0:
            # this is the first batch of data, hence we need to initialize everything
            if self.n_components is None:
                self.n_components_ = min(X.shape)
            else:
                self.n_components_ = min(X.shape[0], X.shape[1], self.n_components)

            self.mean_ = X.mean(axis=0)
            X_centered = X - self.mean_
            _, S, V = ht.linalg.svd(X_centered)
            self.components_ = V[:, : self.n_components_].T
            self.singular_values_ = S[: self.n_components_]
            self.n_samples_seen_ = X.shape[0]

        else:
            # if already batches of data have been seen before, only an update is necessary
            U, S, mean = _isvd(
                X.T,
                self.components_.T,
                self.singular_values_,
                V_old=None,
                maxrank=self.n_components,
                old_matrix_size=self.n_samples_seen_,
                old_rowwise_mean=self.mean_,
            )
            self.components_ = U.T
            self.singular_values_ = S
            self.mean_ = mean
            self.n_samples_seen_ += X.shape[0]
            self.n_components_ = self.components_.shape[0]

    def transform(self, X: ht.DNDarray) -> ht.DNDarray:
        """
        Apply dimensionality based on PCA to X.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_features)
            Data set to be transformed.
        """
        ht.sanitize_in(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"X must have the same number of features as the training data. Expected {self.mean_.shape[0]} but got {X.shape[1]}."
            )

        # center data and apply PCA
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, X: ht.DNDarray) -> ht.DNDarray:
        """
        Transform data back to its original space.

        Parameters
        ----------
        X : DNDarray of shape (n_samples, n_components)
            Data set to be transformed back.
        """
        ht.sanitize_in(X)
        if X.shape[1] != self.n_components_:
            raise ValueError(
                f"Dimension mismatch. Expected input of shape n_points x {self.n_components_} but got {X.shape}."
            )

        return X @ self.components_ + self.mean_
