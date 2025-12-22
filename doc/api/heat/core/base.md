Module heat.core.base
=====================
Provides mixins for high-level algorithms, e.g. classifiers or clustering algorithms.

Functions
---------

`is_classifier(estimator: object) ‑> bool`
:   Return ``True`` if the given estimator is a classifier, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

`is_clusterer(estimator: object) ‑> bool`
:   Return ``True`` if the given estimator is a clusterer, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

`is_estimator(estimator: object) ‑> bool`
:   Return ``True`` if the given estimator is an estimator, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

`is_regressor(estimator: object) ‑> bool`
:   Return ``True`` if the given estimator is a regressor, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

`is_transformer(estimator: object) ‑> bool`
:   Return ``True`` if the given estimator is a transformer, ``False`` otherwise.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

Classes
-------

`BaseEstimator()`
:   Abstract base class for all estimators, i.e. parametrized analysis algorithms, in Heat. Can be used as mixin.

    ### Descendants

    * heat.classification.kneighborsclassifier.KNeighborsClassifier
    * heat.cluster._kcluster._KCluster
    * heat.cluster.batchparallelclustering._BatchParallelKCluster
    * heat.cluster.spectral.Spectral
    * heat.decomposition.dmd.DMD
    * heat.decomposition.dmd.DMDc
    * heat.decomposition.pca.IncrementalPCA
    * heat.decomposition.pca.PCA
    * heat.naive_bayes.gaussianNB.GaussianNB
    * heat.preprocessing.preprocessing.MaxAbsScaler
    * heat.preprocessing.preprocessing.MinMaxScaler
    * heat.preprocessing.preprocessing.Normalizer
    * heat.preprocessing.preprocessing.RobustScaler
    * heat.preprocessing.preprocessing.StandardScaler
    * heat.regression.lasso.Lasso

    ### Methods

    `get_params(self, deep: bool = True) ‑> Dict[str, object]`
    :   Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default: True
            If ``True``, will return the parameters for this estimator and contained sub-objects that are estimators.

    `set_params(self, **params: Dict[str, object]) ‑> ~self`
    :   Set the parameters of this estimator. The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have to be nested dictionaries.

        Parameters
        ----------
        **params : dict[str, object]
            Estimator parameters to bet set.

`ClassificationMixin()`
:   Mixin for all classifiers in Heat.

    ### Descendants

    * heat.classification.kneighborsclassifier.KNeighborsClassifier
    * heat.naive_bayes.gaussianNB.GaussianNB

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray)`
    :   Fits the classification model.

        Parameters
        ----------
        x : DNDarray
            Training instances to train on. Shape = (n_samples, n_features)

        y : DNDarray
            Class values to fit. Shape = (n_samples, )

    `fit_predict(self, x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Fits model and returns classes for each input sample
        Convenience method; equivalent to calling :func:`fit` followed by :func:`predict`.

        Parameters
        ----------
        x : DNDarray
            Input data to be predicted. Shape = (n_samples, n_features)
        y : DNDarray
            Class values to fit. Shape = (n_samples, )

    `predict(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Predicts the class labels for each sample.

        Parameters
        ----------
        x : DNDarray
            Values to predict the classes for. Shape = (n_samples, n_features)

`ClusteringMixin()`
:   Clustering mixin for all clusterers in Heat.

    ### Descendants

    * heat.cluster._kcluster._KCluster
    * heat.cluster.batchparallelclustering._BatchParallelKCluster
    * heat.cluster.spectral.Spectral

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray)`
    :   Computes the clustering.

        Parameters
        ----------
        x : DNDarray
            Training instances to cluster. Shape = (n_samples, n_features)

    `fit_predict(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Compute clusters and returns the predicted cluster assignment for each sample.
        Returns index of the cluster each sample belongs to.
        Convenience method; equivalent to calling :func:`fit` followed by :func:`predict`.

        Parameters
        ----------
        x : DNDarray
            Input data to be clustered. Shape = (n_samples, n_features)

`RegressionMixin()`
:   Mixin for all regression estimators in Heat.

    ### Descendants

    * heat.decomposition.dmd.DMD
    * heat.decomposition.dmd.DMDc
    * heat.regression.lasso.Lasso

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray)`
    :   Fits the regression model.

        Parameters
        ----------
        x : DNDarray
            Training instances to train on. Shape = (n_samples, n_features)
        y : DNDarray
            Continuous values to fit. Shape = (n_samples,)

    `fit_predict(self, x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Fits model and returns regression predictions for each input sample
        Convenience method; equivalent to calling :func:`fit` followed by :func:`predict`.

        Parameters
        ----------
        x : DNDarray
            Input data to be predicted. Shape = (n_samples, n_features)
        y : DNDarray
            Continuous values to fit. Shape = (n_samples,)

    `predict(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Predicts the continuous labels for each sample.

        Parameters
        ----------
        x : DNDarray
            Values to let the model predict. Shape = (n_samples, n_features)

`TransformMixin()`
:   Mixin for all transformations in Heat.

    ### Descendants

    * heat.decomposition.pca.IncrementalPCA
    * heat.decomposition.pca.PCA
    * heat.preprocessing.preprocessing.MaxAbsScaler
    * heat.preprocessing.preprocessing.MinMaxScaler
    * heat.preprocessing.preprocessing.Normalizer
    * heat.preprocessing.preprocessing.RobustScaler
    * heat.preprocessing.preprocessing.StandardScaler

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray)`
    :   Fits the transformation model.

        Parameters
        ----------
        x : DNDarray
            Training instances to train on. Shape = (n_samples, n_features)

    `fit_transform(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Fits model and returns transformed data for each input sample
        Convenience method; equivalent to calling :func:`fit` followed by :func:`transform`.

        Parameters
        ----------
        x : DNDarray
            Input data to be transformed. Shape = (n_samples, n_features)

    `transform(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Transforms the input data.

        Parameters
        ----------
        x : DNDarray
            Values to transform. Shape = (n_samples, n_features)
