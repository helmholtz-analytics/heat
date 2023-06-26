"""Provides mixins for high-level algorithms, e.g. classifiers or clustering algorithms."""

import inspect
import json

from typing import Dict, List, TypeVar

from .dndarray import DNDarray

self = TypeVar("self")


class BaseEstimator:
    """
    Abstract base class for all estimators, i.e. parametrized analysis algorithms, in HeAT. Can be used as mixin.
    """

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

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default: True
            If ``True``, will return the parameters for this estimator and contained sub-objects that are estimators.
        """
        params = {}

        for key in self._parameter_names():
            value = getattr(self, key)

            if deep and hasattr(value, "get_params"):
                value = value.get_params()

            params[key] = value
        return params

    def __repr__(self, indent: int = 1) -> str:
        """
        Returns a printable representation of the object.

        Parameters
        ----------
        indent : int, default: 1
            Indicates the indentation for the top-level output.
        """
        return "{}({})".format(self.__class__.__name__, json.dumps(self.get_params(), indent=4))

    def set_params(self, **params: Dict[str, object]) -> self:
        """
        Set the parameters of this estimator. The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have to be nested dictionaries.

        Parameters
        ----------
        **params : dict[str, object]
            Estimator parameters to bet set.
        """
        if not params:
            return self

        parameter_names = self._parameter_names()
        for key, value in params.items():
            if key not in parameter_names:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self}. Check the list of available parameters with `estimator.get_params().keys()`."
                )

            if isinstance(value, dict):
                getattr(self, key).set_params(value)
            else:
                setattr(self, key, value)

        return self


class ClassificationMixin:
    """
    Mixin for all classifiers in HeAT.
    """

    def fit(self, x: DNDarray, y: DNDarray):
        """
        Fits the classification model.

        Parameters
        ----------
        x : DNDarray
            Training instances to train on. Shape = (n_samples, n_features)

        y : DNDarray
            Class values to fit. Shape = (n_samples, )

        """
        raise NotImplementedError()

    def fit_predict(self, x: DNDarray, y: DNDarray) -> DNDarray:
        """
        Fits model and returns classes for each input sample
        Convenience method; equivalent to calling :func:`fit` followed by :func:`predict`.

        Parameters
        ----------
        x : DNDarray
            Input data to be predicted. Shape = (n_samples, n_features)
        y : DNDarray
            Class values to fit. Shape = (n_samples, )
        """
        self.fit(x, y)
        return self.predict(x)

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Predicts the class labels for each sample.

        Parameters
        ----------
        x : DNDarray
            Values to predict the classes for. Shape = (n_samples, n_features)
        """
        raise NotImplementedError()


class ClusteringMixin:
    """
    Clustering mixin for all clusterers in HeAT.
    """

    def fit(self, x: DNDarray):
        """
        Computes the clustering.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)
        """
        raise NotImplementedError()

    def fit_predict(self, x: DNDarray) -> DNDarray:
        """
        Compute clusters and returns the predicted cluster assignment for each sample.
        Returns index of the cluster each sample belongs to.
        Convenience method; equivalent to calling :func:`fit` followed by :func:`predict`.

        Parameters
        ----------
        x : DNDarray
            Input data to be clustered. Shape = (n_samples, n_features)
        """
        self.fit(x)
        return self.predict(x)


class RegressionMixin:
    """
    Mixin for all regression estimators in HeAT.
    """

    def fit(self, x: DNDarray, y: DNDarray):
        """
        Fits the regression model.

        Parameters
        ----------
        x : DNDarray
            Training instances to train on. Shape = (n_samples, n_features)
        y : DNDarray
            Continuous values to fit. Shape = (n_samples,)
        """
        raise NotImplementedError()

    def fit_predict(self, x: DNDarray, y: DNDarray) -> DNDarray:
        """
        Fits model and returns regression predictions for each input sample
        Convenience method; equivalent to calling :func:`fit` followed by :func:`predict`.

        Parameters
        ----------
        x : DNDarray
            Input data to be predicted. Shape = (n_samples, n_features)
        y : DNDarray
            Continuous values to fit. Shape = (n_samples,)
        """
        self.fit(x, y)
        return self.predict(x)

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Predicts the continuous labels for each sample.

        Parameters
        ----------
        x : DNDarray
            Values to let the model predict. Shape = (n_samples, n_features)
        """
        raise NotImplementedError()


def is_classifier(estimator: object) -> bool:
    """
    Return ``True`` if the given estimator is a classifier, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.
    """
    return isinstance(estimator, ClassificationMixin)


def is_estimator(estimator: object) -> bool:
    """
    Return ``True`` if the given estimator is an estimator, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.
    """
    return isinstance(estimator, BaseEstimator)


def is_clusterer(estimator: object) -> bool:
    """
    Return ``True`` if the given estimator is a clusterer, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    """
    return isinstance(estimator, ClusteringMixin)


def is_regressor(estimator: object) -> bool:
    """
    Return ``True`` if the given estimator is a regressor, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.
    """
    return isinstance(estimator, RegressionMixin)
