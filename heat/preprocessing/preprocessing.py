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
            NotImplementedError("Standard scaler with sample weights is not yet implemented.")

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
            return (X - self.mean_) / self.var_
        # ... or in-place:
        else:
            X -= self.mean_
            X /= self.var_
            return X

    def inverse_transform(self, Y):
        """
        Apply the inverse transformation associated with the standard scaler
        """
        if self.copy:
            return Y * self.var_ + self.mean_
        # ... or in-place:
        else:
            Y *= self.var_
            Y += self.mean_
            return Y
