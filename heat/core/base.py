import inspect
import json
from typing import List, Dict, Any, TypeVar
self = TypeVar('self')

class BaseEstimator:
    @classmethod
    def _parameter_names(cls) -> List[str]:
        """
        Get the names of all parameters that can be set inside the constructor of the estimator.
        """
        init = cls.__init__
        if init is object.__init__:
            return []

        # introspect the constructor arguments to find the model parameters
        init_signature = inspect.signature(init)

        # consider the constructor parameters excluding 'self'
        return [
            p.name
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind == p.POSITIONAL_OR_KEYWORD
        ]

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args
        ----------
        deep : bool
            If True, will return the parameters for this estimator and contained sub-objects that are estimators.
        """
        params = dict()

        for key in self._parameter_names():
            value = getattr(self, key)

            if deep and hasattr(value, "get_params"):
                value = value.get_params()

            params[key] = value
        return params

    def __repr__(self, indent=1):
        return "{}({})".format(self.__class__.__name__, json.dumps(self.get_params(), indent=4))

    def set_params(self, **params) -> self:
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects (such
        as pipelines). The latter have to be nested dictionaries.

        Args
        ----------
        **params : Dict[str, Any]
            Estimator parameters to bet set.

        """
        if not params:
            return self

        parameter_names = self._parameter_names()
        for key, value in params.items():
            if key not in parameter_names:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. Check the list of available parameters "
                    "with `estimator.get_params().keys()`.".format(key, self)
                )

            if isinstance(value, dict):
                getattr(self, key).set_params(value)
            else:
                setattr(self, key, value)

        return self


class ClassificationMixin:
    """
    Mixin for all classifiers in Heat.
    """

    def fit(self, X, Y):
        """
        Fits the classification model.

        Args
        ----------
        X : ht.DNDarray
            Training instances to train on. Shape = (n_samples, n_features)

        Y : ht.DNDarray
            Class values to fit. Shape = (n_samples, )

        """
        raise NotImplementedError()

    def fit_predict(self, X, Y) -> ht.DNDarray:
        """
        Fits model and returns classes for each input sample

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Args
        ----------
        X : ht.DNDarray
            Input data to be predicted. Shape = (n_samples, n_features)
        Y : ht.DNDarray
            Class values to fit. Shape = (n_samples, )
        """
        self.fit(X, Y)
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the class labels for each sample.

        Args
        ----------
        X : ht.DNDarray
            Values to predict the classes for. Shape = (n_samples, n_features)

        """
        raise NotImplementedError()


class ClusteringMixin:
    """
    Clustering mixin for all clusterers in HeAT.
    """

    def fit(self, X):
        """
        Computes the clustering.

        Args
        ----------
        X : ht.DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

        """
        raise NotImplementedError()

    def fit_predict(self, X) -> ht.DNDarray:
        """
        Compute clusters and returns the predicted cluster assignment for each sample.

        Returns index of the cluster each sample belongs to.
        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : ht.DNDarray
            Input data to be clustered. Shape = (n_samples, n_features)

        """
        self.fit(X)
        return self.predict(X)


class RegressionMixin:
    """
    Mixin for all regression estimators in Heat.
    """

    def fit(self, X, Y):
        """
        Fits the regression model.

        Args
        ----------
        X : ht.DNDarray
            Training instances to train on. Shape = (n_samples, n_features)

        Y : ht.DNDarray
            Continuous values to fit. Shape = (n_samples,)

        """
        raise NotImplementedError()

    def fit_predict(self, X, Y) -> ht.DNDarray:
        """
        Fits model and returns regression predictions for each input sample

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Args
        ----------
        X : ht.DNDarray,
            Input data to be predicted. Shape = (n_samples, n_features)
        Y : ht.DNDarray
            Continuous values to fit. Shape = (n_samples,)

        """
        self.fit(X, Y)
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the continuous labels for each sample.

        Args
        ----------
        X : ht.DNDarray
            Values to let the model predict. Shape = (n_samples, n_features)
        """
        raise NotImplementedError()


def is_classifier(estimator) -> bool:
    """
    Return True if the given estimator is a classifier, False otherwise.

    Args
    ----------
    estimator : object
        Estimator object to test.
    """
    return isinstance(estimator, ClassificationMixin)


def is_estimator(estimator) -> bool:
    """
    Return True if the given estimator is an estimator, False otherwise.

    Args
    ----------
    estimator : object
        Estimator object to test.
    """
    return isinstance(estimator, BaseEstimator)


def is_clusterer(estimator) -> bool:
    """
    Return True if the given estimator is a clusterer, False otherwise.

    Args
    ----------
    estimator : object
        Estimator object to test.

    """
    return isinstance(estimator, ClusteringMixin)


def is_regressor(estimator) -> bool:
    """Return True if the given estimator is a regressor, False otherwise.

    Args
    ----------
    estimator : object
        Estimator object to test.
    """
    return isinstance(estimator, RegressionMixin)
