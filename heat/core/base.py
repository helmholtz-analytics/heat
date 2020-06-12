import inspect
import json


class BaseEstimator:
    @classmethod
    def _parameter_names(cls):
        """
        Get the names of all parameters that can be set inside the constructor of the estimator.

        Returns
        -------
        parameter_names : list of str
            The names of the estimator's parameters.
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

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool
            If True, will return the parameters for this estimator and contained sub-objects that are estimators.
        Returns
        -------
        params : dict of str to any
            Parameter names mapped to their values.
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

    def set_params(self, **params):
        """
        Set the parameters of this estimator. The method works on simple estimators as well as on nested objects (such
        as pipelines). The latter have to be nested dictionaries.

        Parameters
        ----------
        **params : dict of str to any
            Estimator parameters to bet set.

        Returns
        -------
        self : object
            Estimator instance for chaining.
        """
        if not params:
            return self

        parameter_names = self._parameter_names()
        for key, value in params.items():
            if key not in parameter_names:
                raise ValueError(
                    "Invalid parameter {} for estimator {}. Check the list of available parameters with `estimator.get_params().keys()`.".format(
                        key, self
                    )
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

        Parameters
        ----------
        X : ht.DNDarray, shape=(n_samples, n_features)
            Training instances to train on.

        Y : ht.DNDarray, shape=(n_samples)
            Class values to fit.
        """
        raise NotImplementedError()

    def fit_predict(self, X, Y):
        """
        Fits model and computes classesfor each input sample

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]
            Input data to be predicted.

        Returns
        -------
        labels : ht.DNDarray, shape = [n_samples]
            Predicted classes.
        """
        self.fit(X, Y)
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the class labels for each sample.

        Parameters
        ----------
        X : ht.DNDarray, shape=[n_samples, n_features)
            Values to predict the classes for.
        """
        raise NotImplementedError()


class ClusteringMixin:
    """
    Clustering mixin for all clusterers in HeAT.
    """

    def fit(self, X):
        """
        Computes the clustering.

        Parameters
        ----------
        X : ht.DNDarray, shape=[n_samples, n_features)
            Training instances to cluster.
        """
        raise NotImplementedError()

    def fit_predict(self, X):
        """
        Compute clusters and predict cluster assignment for each sample.

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]
            Input data to be clustered.

        Returns
        -------
        labels : ht.DNDarray, shape = [n_samples]
            Index of the cluster each sample belongs to.
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

        Parameters
        ----------
        X : ht.DNDarray, shape=(n_samples, n_features)
            Training instances to train on.

        Y : ht.DNDarray, shape=(n_samples)
            Continuous values to fit.
        """
        raise NotImplementedError()

    def fit_predict(self, X, Y):
        """
        Fits model and computes regression predictions for each input sample

        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : ht.DNDarray, shape = [n_samples, n_features]
            Input data to be predicted.

        Returns
        -------
        labels : ht.DNDarray, shape = [n_samples,]
            Predicted value.
        """
        self.fit(X, Y)
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the continuous labels for each sample.

        Parameters
        ----------
        X : ht.DNDarray, shape=[n_samples, n_features)
            Values to let the model predict.
        """
        raise NotImplementedError()


def is_classifier(estimator):
    """
    Return True if the given estimator is a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return isinstance(estimator, ClassificationMixin)


def is_estimator(estimator):
    """
    Return True if the given estimator is an estimator.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an estimator and False otherwise.
    """
    return isinstance(estimator, BaseEstimator)


def is_clusterer(estimator):
    """
    Return True if the given estimator is a clusterer.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a clusterer and False otherwise.
    """
    return isinstance(estimator, ClusteringMixin)


def is_regressor(estimator):
    """Return True if the given estimator is a regressor.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return isinstance(estimator, RegressionMixin)
