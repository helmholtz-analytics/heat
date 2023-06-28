"""
Module implementing basic data preprocessing techniques
"""

import heat as ht


class StandardScaler(ht.TransformMixin, ht.BaseEstimator):
    """Standard Scaler"""

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X, sample_weight=None):
        """
        Fit standard scaler
        """
        if sample_weight is not None:
            NotImplementedError(
                "Standard scaler with sample weights is not yet implemented. We apologize for the inconvenience."
            )

        # determine mean and variance of the input data X
        self.mean_ = ht.mean(X, axis=0)
        self.var_ = ht.var(X, axis=0)

        # # if variance is below machine precision, do not use it for transformation
        # if X.dtype == ht.float32:
        #     tol = 1e-8
        # else:
        #     tol = 1e-16
        # if var_ < tol:
        #     self.with_std = None
        # else:
        #     self.var_ = var_
        return self

    def transform(self, X):
        """
        Apply standard scaler in order to transform data
        """
        # transform the input data X according to mean_ and var_ from the scaler...
        # ... either with copy...
        if self.copy:
            return (X - self.mean_) / (self.var_) ** 0.5
        # ... or in-place:
        else:
            X -= self.mean_
            X /= (self.var_) ** 0.5
            return X

    def inverse_transform(self, Y):
        """
        Apply the inverse transformation associated with the standard scaler
        """
        if self.copy:
            return Y * (self.var_) ** 0.5 + self.mean_
        # ... or in-place:
        else:
            Y *= (self.var_) ** 0.5
            Y += self.mean_
            return Y


class MinMaxScaler(ht.TransformMixin, ht.BaseEstimator):
    """
    Min-Max-Scaler
    """

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.copy = copy
        self.feature_range = feature_range
        if clip:
            raise NotImplementedError(
                "Clipped MinMaxScaler is not yet implemented. We apologize for the inconvenience."
            )
        if feature_range[1] <= feature_range[0]:
            raise ValueError(
                "Upper bound of feature_range must be strictly larger than lower bound."
            )

    def fit(self, X):
        """
        Fit MinMaxScaler
        """
        self.data_min_ = ht.min(X, axis=0)
        self.data_max_ = ht.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        return self

    def transform(self, X):
        """
        Transform according to MinMaxScaler
        """
        if self.copy:
            Y = (X - self.data_min_) * self.scale_ + self.feature_range[0]
            return Y
        else:
            X -= self.data_min_
            X *= self.scale_
            X += self.feature_range[0]
            return X

    def inverse_transform(self, Y):
        """
        Apply inverse of MinMaxScaler
        """
        if self.copy:
            X = (Y - self.feature_range[0]) / self.scale_ + self.data_min_
            return X
        else:
            Y -= self.feature_range[0]
            Y /= self.scale_
            Y += self.data_min_
            return Y


class Normalizer(ht.TransformMixin, ht.BaseEstimator):
    """
    Normalizer
    """

    def __init__(self, norm="l2", *, copy=True):
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
                "Normalization with respect to norms other than l2, l1 or linfty not yet implemented. We apologize for the inconvenience."
            )

    def fit(self, X):
        """Only a dummy in this case"""
        return self

    def transform(self, X):
        """
        Normalize all data entries
        """
        if self.copy:
            Y = X / ht.norm(X, axis=1, ord=self.ord_).reshape((-1, 1))
            return Y
        else:
            X /= ht.norm(X, axis=1, ord=self.ord_).reshape((-1, 1))
            return X


class MaxAbsScaler(ht.TransformMixin, ht.BaseEstimator):
    """
    MaxAbsScaler
    """

    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X):
        """
        Fit MaxAbsScaler to data
        """
        self.max_abs_ = ht.norm(X, axis=0, ord=ht.inf)
        self.scale_ = 1.0 / self.max_abs_
        return self

    def transform(self, X):
        """
        Transform data with MaxAbsScaler
        """
        if self.copy:
            Y = X * self.scale_
            return Y
        else:
            X *= self.scale_
            return X

    def inverse_transform(self, Y):
        """
        Apply inverse MaxAbsScaler
        """
        if self.copy:
            X = Y * self.max_abs_
            return X
        else:
            Y *= self.max_abs_
            return Y


class RobustScaler(ht.TransformMixin, ht.BaseEstimator):
    """
    Robust Scaler
    """

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        if (
            self.quantile_range[0] >= self.quantile_range[1]
            or self.quantile_range[1] > 100.0
            or self.quantile_range[0] < 0.0
        ):
            raise ValueError(
                "Lower bound of quantile range must be strictly smaller than uppert bound; both bounds need to be between 0.0 and 100.0."
            )
        if unit_variance:
            raise NotImplementedError(
                "Robust Scaler with additional unit variance scaling is not yet implemented. We apologize for the inconvenience."
            )
        else:
            self.unit_variance = unit_variance

    def fit(self, X):
        """
        Fit RobustScaler
        """
        if self.with_centering:
            self.center_ = ht.median(X, axis=0)
        if self.with_scaling:
            self.iqr_ = ht.percentile(X, self.quantile_range[1], axis=0) - ht.percentile(
                X, self.quantile_range[0], axis=0
            )
            self.scale_ = 1.0 / self.iqr_
        return self

    def transform(self, X):
        """
        Transform with RobustScaler
        """
        if self.copy:
            Y = X
            if self.with_centering:
                Y -= self.center_
            if self.with_scaling:
                Y *= self.scale_
            return Y
        else:
            if self.with_centering:
                X -= self.center_
            if self.with_scaling:
                X *= self.scale_
            return X

    def inverse_transform(self, Y):
        """
        Inverse RobustScaler
        """
        if self.copy:
            X = Y
            if self.with_scaling:
                X *= self.iqr_
            if self.with_centering:
                X += self.center_
            return X
        else:
            if self.with_scaling:
                Y *= self.iqr_
            if self.with_centering:
                Y += self.center_
            return Y
