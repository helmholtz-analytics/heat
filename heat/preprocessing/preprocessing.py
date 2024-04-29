"""
Module implementing basic data preprocessing techniques
"""

import heat as ht
from typing import Optional, Tuple, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

"""
The implementation is heavily inspired by the corresponding routines in scikit-learn (https://scikit-learn.org/stable/modules/preprocessing.html).
"""


# auxiliary function that checks if input array is appropriate to serve as data set for fitting or transforming
def _is_2D_float_DNDarray(input):
    if not isinstance(input, ht.DNDarray):
        raise TypeError(f"Input of preprocessing routines must be DNDarray, but is {type(input)}.")
    if not input.ndim == 2:
        raise ValueError(
            f"Input of preprocessing routines must be a 2D DNDarray of shape (n_datapoints, n_features), but dimension is {input.ndim}."
        )
    if ht.heat_type_is_exact(input.dtype):
        raise TypeError(
            f"Supported data types for preprocessing routines are float32 and float64, but dtype of input is {input.dtype}.",
        )


# auxiliary function that checks whether parameter of a Scaler and data to be transformed with this Scaler have matching shapes
def _has_n_features(param, inputdata):
    if param.shape[0] != inputdata.shape[1]:
        raise ValueError(
            f"Scaler has been fitted on a data set with {param.shape[0]} features, but shall now be applied to data with {inputdata.shape[1]} features."
        )


# auxiliary function that returns expected precision depending on input data type
# this is used to determine whether a feature is almost constant (w.r.t. machine precision) and should therefore not be scaled
def _tol_wrt_dtype(inputdata):
    if inputdata.dtype == ht.float32:
        return 1e-7
    if inputdata.dtype == ht.float64:
        return 1e-14


class StandardScaler(ht.TransformMixin, ht.BaseEstimator):
    """Standardization of features to mean 0 and variance 1 by affine linear transformation; similar to `sklearn.preprocessing.StandardScaler`.
    The data set to be scaled must be stored as 2D-`DNDarray` of shape (n_datapoints, n_features).
    Shifting to mean 0 and scaling to variance 1 is applied to each feature independently.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.

    with_mean : bool, default=True
        If True, center the data (i.e. mean = 0) before scaling.

    with_std : bool, default=True
        If True, scale the data to variance = 1.

    Attributes
    ----------
    scale_ : DNDarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve unit
        variance. Set to ``None`` (no variance scaling applied) if ``var = None`` or ``var`` below machine precision.

    mean_ : DNDarray of shape (n_features,) or None
        The mean value for each feature. Equal to ``None`` when ``with_mean=False``.

    var_ : DNDarray of shape (n_features,) or None
        Featurewise variance of the given data. Equal to ``None`` when ``with_std=False``.
    """

    def __init__(self, *, copy: bool = True, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X: ht.DNDarray, sample_weight: Optional[ht.DNDarray] = None) -> Self:
        """
        Fit ``StandardScaler`` to the given data ``X``, i.e. compute mean and standard deviation of ``X`` to be used for later scaling.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features).
            Data used to compute the mean and standard deviation used for later featurewise scaling.

        sample_weight : Not yet supported.
            Raises ``NotImplementedError``.
        """
        if sample_weight is not None:
            NotImplementedError(
                "Standard scaler with sample weights is not yet implemented. You can open an issue to request this feature on  https://github.com/helmholtz-analytics/heat/issues/new/choose."
            )
        _is_2D_float_DNDarray(X)

        # determine mean and variance of the input data X and store them in self.mean_ and self.var_
        self.mean_ = ht.mean(X, axis=0)
        self.var_ = ht.var(X, axis=0)

        # check if var_ is below machine precision for some features, set scaling factor to 1 for these features if so and print warning
        self.scale_ = self.var_
        tol = _tol_wrt_dtype(X)
        if self.scale_.min() < tol:
            self.scale_ = ht.where(
                ht.abs(self.scale_) >= tol, self.scale_, ht.ones_like(self.scale_)
            )
            print(
                "At least one of the features is almost constant (w.r.t. machine precision) and will not be scaled for this reason."
            )
        self.scale_ = 1.0 / (self.scale_) ** 0.5
        return self

    def transform(self, X: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """Applies standardization to input data ``X`` by centering and scaling w.r.t. mean and std previously computed and saved in ``StandardScaler`` with :meth:``fit``.

        Parameters
        ----------
        X : DNDarray (n_datapoints, n_features)
            The data set to be standardized.
        copy : bool, default=None
            Copy the input ``X`` or not.
        """
        _is_2D_float_DNDarray(X)
        _has_n_features(self.mean_, X)
        if self.copy:
            return (X - self.mean_) * self.scale_
        # else in-place:
        X -= self.mean_
        X *= self.scale_
        return self

    def inverse_transform(self, Y: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Scale back the data to the original representation, i.e. apply the inverse of :meth:``transform`` to the input ``Y``.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            Data to be scaled back.
        copy : bool, default=None
            Copy the input ``Y`` or not.
        """
        _is_2D_float_DNDarray(Y)
        _has_n_features(self.mean_, Y)
        if self.copy:
            return Y / self.scale_ + self.mean_
        # else in-place:
        Y /= self.scale_
        Y += self.mean_
        return self


class MinMaxScaler(ht.TransformMixin, ht.BaseEstimator):
    """
    Min-Max-Scaler: transforms the features by scaling each feature (affine) linearly to the prescribed range;
    similar to `sklearn.preprocessing.MinMaxScaler`.
    The data set to be scaled must be stored as 2D-`DNDarray` of shape (n_datapoints, n_features).

    Each feature is scaled and translated individually such that it is in the given range on the input data set,
    e.g. between zero and one (default).

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed features.

    copy : bool, default=True
        ``copy = False`` means in-place transformations whenever possible.

    clip : Not yet supported.
        raises ``NotImplementedError``.

    Attributes
    ----------
    min_ : DNDarray of shape (n_features,)
        translation required per feature

    scale_ : DNDarray of shape (n_features,)
        scaling required per feature

    data_min_ : DNDarray of shape (n_features,)
        minimum per feature in the input data set

    data_max_ : DNDarray of shape (n_features,)
        maximum per feature in the input data set

    data_range_ : DNDarray of shape (n_features,)
        range per feature in the input data set
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        *,
        copy: bool = True,
        clip: bool = False,
    ):
        self.copy = copy
        self.feature_range = feature_range
        self.clip = clip
        if clip:
            raise NotImplementedError(
                "Clipped MinMaxScaler is not yet implemented. You can open an issue to request this feature on  https://github.com/helmholtz-analytics/heat/issues/new/choose."
            )
        if feature_range[1] <= feature_range[0]:
            raise ValueError(
                f"Upper bound of feature_range must be strictly larger than lower bound, but provided bounds are {self.feature_range[0]} and {self.feature_range[1]}."
            )

    def fit(self, X: ht.DNDarray) -> Self:
        """
        Fit the MinMaxScaler: i.e. compute the parameters required for later scaling.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            data set to which scaler shall be fitted.
        """
        _is_2D_float_DNDarray(X)
        self.data_min_ = ht.min(X, axis=0)
        self.data_max_ = ht.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        # if data_range is below machine precision for a feature, set scaling factor to 1 for this feature
        tol = _tol_wrt_dtype(X)
        self.scale_ = self.data_range_ / (self.feature_range[1] - self.feature_range[0])
        if ht.abs(self.data_range_).min() < tol:
            self.scale_ = ht.where(
                ht.abs(self.data_range_) >= tol, self.scale_, ht.ones_like(self.data_range_)
            )
            print(
                "At least one of the features is almost constant (w.r.t. machine precision) and will not be scaled for this reason."
            )
        self.scale_ = 1.0 / self.scale_
        self.min_ = -self.data_min_ * self.scale_ + self.feature_range[0]
        return self

    def transform(self, X: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Transform input data with MinMaxScaler: i.e. scale features of ``X`` according to feature_range.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            Data set to be transformed.
        """
        _is_2D_float_DNDarray(X)
        _has_n_features(self.data_min_, X)
        if self.copy:
            Y = (X - self.data_min_) * self.scale_ + self.feature_range[0]
            return Y
        # else in-place:
        X -= self.data_min_
        X *= self.scale_
        X += self.feature_range[0]
        return self

    def inverse_transform(self, Y: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Apply the inverse of :meth:``fit``.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            Data set to be transformed back.
        """
        _is_2D_float_DNDarray(Y)
        _has_n_features(self.data_min_, Y)
        if self.copy:
            X = (Y - self.feature_range[0]) / self.scale_ + self.data_min_
            return X
        # else in-place:
        Y -= self.feature_range[0]
        Y /= self.scale_
        Y += self.data_min_
        return self


class Normalizer(ht.TransformMixin, ht.BaseEstimator):
    """
    Normalizer: each data point of a data set is scaled to unit norm independently.
    The data set to be scaled must be stored as 2D-`DNDarray` of shape (n_datapoints, n_features); therefore
    the Normalizer scales each row to unit norm. This object is similar to `sklearn.preprocessing.Normalizer`.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize the data points. ``norm='max'`` refers to the :math:`\\ell^\\infty`-norm.

    copy : bool, default=True
        ``copy=False`` enables in-place normalization.

    Attributes
    ----------
    None


    Notes
    -----
    Normalizer is :term:`stateless` and, consequently, :meth:``fit`` is only a dummy that does not need to be called before :meth:``transform``.
    Since :meth:``transform`` is not bijective, there is no back-transformation :meth:``inverse_transform``.
    """

    def __init__(self, norm: str = "l2", *, copy: bool = True):
        self.norm_ = norm
        self.copy = copy
        if norm == "l2":
            self.ord_ = 2
        elif norm == "l1":
            self.ord_ = 1
        elif norm == "max":
            self.ord_ = ht.inf
        else:
            raise NotImplementedError(
                "Normalization with respect to norms other than l2, l1 or linfty not yet implemented. You can open an issue to request this feature on  https://github.com/helmholtz-analytics/heat/issues/new/choose."
            )

    def fit(self, X: ht.DNDarray) -> Self:
        """Since :object:``Normalizer`` is stateless, this function is only a dummy."""
        return self

    def transform(self, X: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Apply Normalizer trasformation: scales each data point of the input data set ``X`` to unit norm (w.r.t. to ``norm``).

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            The data set to be normalized.

        copy : bool, default=None
            ``copy=False`` enables in-place transformation.
        """
        _is_2D_float_DNDarray(X)
        X_norms = ht.norm(X, axis=1, ord=self.ord_).reshape((-1, 1))

        # if norm of data point is close to zero (w.r.t. machine precision), do not scale this data point
        tol = _tol_wrt_dtype(X)
        if X_norms.min() < tol:
            X_norms = ht.where(X_norms >= tol, X_norms, ht.ones_like(X_norms))
            print(
                "At least one of the data points has almost zero norm (w.r.t. machine precision) and will not be scaled for this reason."
            )
        if self.copy:
            Y = X / X_norms
            return Y
        # else in-place:
        X /= X_norms
        del X_norms
        return self


class MaxAbsScaler(ht.TransformMixin, ht.BaseEstimator):
    """
    MaxAbsScaler: scale each feature of a given data set linearly by its maximum absolute value. The underyling data set to be scaled is
    assumed to be stored as a 2D-`DNDarray` of shape (n_datapoints, n_features); this routine is similar to
    `sklearn.preprocessing.MaxAbsScaler`.

    Each feature is scaled individually such that the maximal absolute value of each feature after transformation will be 1.0.
    No shifting/centering is applied.

    Parameters
    ----------
    copy : bool, default=True
        ``copy=False`` enables in-place transformation.

    Attributes
    ----------
    scale_ : DNDarray of shape (n_features,)
        Per feature relative scaling of the data.

    max_abs_ : DNDarray of shape (n_features,)
        Per feature maximum absolute value of the input data.
    """

    def __init__(self, *, copy: bool = True):
        self.copy = copy

    def fit(self, X: ht.DNDarray) -> Self:
        """
        Fit MaxAbsScaler to input data ``X``: compute the parameters to be used for later scaling.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            The data set to which the scaler shall be fitted.
        """
        _is_2D_float_DNDarray(X)
        self.max_abs_ = ht.norm(X, axis=0, ord=ht.inf)

        # if max abs is close to machine precision for some feature, do not scale this feature
        tol = _tol_wrt_dtype(X)
        self.scale_ = self.max_abs_
        if self.scale_.min() < tol:
            self.scale_ = ht.where(self.scale_ >= tol, self.scale_, ht.ones_like(self.scale_))
            print(
                "At least one of the features is almost constant zero (w.r.t. machine precision) and will not be scaled for this reason."
            )
        self.scale_ = 1.0 / self.scale_
        return self

    def transform(self, X: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Scale the data with the MaxAbsScaler.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            The data set to be scaled.
        """
        _is_2D_float_DNDarray(X)
        _has_n_features(self.scale_, X)
        if self.copy:
            Y = X * self.scale_
            return Y
        # else in-place:
        X *= self.scale_
        return self

    def inverse_transform(self, Y: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Apply the inverse of :meth:``transform``, i.e. scale the input data ``Y`` back to the original representation.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            The data set to be transformed back.
        """
        _is_2D_float_DNDarray(Y)
        _has_n_features(self.scale_, Y)
        if self.copy:
            X = Y / self.scale_
            return X
        # else in-place:
        Y /= self.scale_
        return self


class RobustScaler(ht.TransformMixin, ht.BaseEstimator):
    """
    This scaler transforms the features of a given data set making use of statistics
    that are robust to outliers: it removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range); this routine is similar
    to ``sklearn.preprocessing.RobustScaler``.

    The underyling data set to be scaled must be stored as a 2D-`DNDarray` of shape (n_datapoints, n_features).
    Each feature is centered and scaled independently.

    Parameters
    ----------
    with_centering : bool, default=True
        If `True`, data are centered before scaling.

    with_scaling : bool, default=True
        If `True`, scale the data to prescribed interquantile range.

    quantile_range : tuple (q_min, q_max), 0.0 <= q_min < q_max <= 100.0, \
        default=(25.0, 75.0)
        Quantile range used to calculate `scale_`; default is the so-called
        the IQR given by ``q_min=25`` and ``q_max=75``.

    copy : bool, default=True
        ``copy=False`` enable in-place transformations.

    unit_variance : not yet supported.
        raises ``NotImplementedError``

    Attributes
    ----------
    center_ : DNDarray of shape (n_features,)
        Feature-wise median value of the given data set.

    iqr_ : DNDarray of shape (n_features,)
        length of the interquantile range for each feature.

    scale_ : array of floats
        feature-wise inverse of ``iqr_``.
    """

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        copy: bool = True,
        unit_variance: bool = False,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        if not with_centering and not with_scaling:
            raise ValueError(
                "Both centering and scaling are disabled, thus RobustScaler could do nothing. At least one of with_scaling or with_centering must be True."
            )
        if (
            self.quantile_range[0] >= self.quantile_range[1]
            or self.quantile_range[1] > 100.0
            or self.quantile_range[0] < 0.0
        ):
            raise ValueError(
                f"Lower bound of quantile range must be strictly smaller than uppert bound and both bounds need to be between 0.0 and 100.0. Inputs however are {self.quantile_range[0]} and {self.quantile_range[1]}."
            )
        if unit_variance:
            raise NotImplementedError(
                "Robust Scaler with additional unit variance scaling is not yet implemented. You can open an issue to request this feature on  https://github.com/helmholtz-analytics/heat/issues/new/choose."
            )
        else:
            self.unit_variance = unit_variance

    def fit(self, X: ht.DNDarray) -> Self:
        """
        Fit RobustScaler to given data set, i.e. compute the parameters required for transformation.

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            Data to which the Scaler should be fitted.
        """
        _is_2D_float_DNDarray(X)
        if self.with_centering:
            self.center_ = ht.median(X, axis=0)
        if self.with_scaling:
            self.iqr_ = ht.percentile(X, self.quantile_range[1], axis=0) - ht.percentile(
                X, self.quantile_range[0], axis=0
            )

            # if length of iqr is close to zero, do not scale this feature
            self.scale_ = self.iqr_
            tol = _tol_wrt_dtype(X)
            if ht.abs(self.scale_).min() < tol:
                self.scale_ = ht.where(
                    ht.abs(self.scale_) >= tol, self.scale_, ht.ones_like(self.scale_)
                )
                print(
                    "At least one of the features is almost constant (w.r.t. machine precision) and will not be scaled for this reason."
                )
            self.scale_ = 1.0 / self.scale_
        return self

    def transform(self, X: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Transform given data with RobustScaler

        Parameters
        ----------
        X : DNDarray of shape (n_datapoints, n_features)
            Data set to be transformed.
        """
        _is_2D_float_DNDarray(X)
        if self.with_centering:
            _has_n_features(self.center_, X)
        if self.with_scaling:
            _has_n_features(self.scale_, X)
        if self.copy:
            Y = X.copy()
            if self.with_centering:
                Y -= self.center_
            if self.with_scaling:
                Y *= self.scale_
            return Y
        # else in-place:
        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X *= self.scale_
        return X

    def inverse_transform(self, Y: ht.DNDarray) -> Union[Self, ht.DNDarray]:
        """
        Apply inverse of :meth:``transform``.

        Parameters
        ----------
        Y : DNDarray of shape (n_datapoints, n_features)
            Data to be back-transformed
        """
        _is_2D_float_DNDarray(Y)
        if self.with_centering:
            _has_n_features(self.center_, Y)
        if self.with_scaling:
            _has_n_features(self.scale_, Y)
        if self.copy:
            X = Y.copy()
            if self.with_scaling:
                X /= self.scale_
            if self.with_centering:
                X += self.center_
            return X
        # else in-place:
        if self.with_scaling:
            Y /= self.scale_
        if self.with_centering:
            Y += self.center_
        return Y
