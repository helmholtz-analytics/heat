Module heat.naive_bayes.gaussianNB
==================================
Distributed Gaussian Naive-Bayes classifier.

Classes
-------

`GaussianNB(priors=None, var_smoothing=1e-09)`
:   Gaussian Naive Bayes (GaussianNB), based on `scikit-learn.naive_bayes.GaussianNB <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`_.
    Can perform online updates to model parameters via method :func:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see Chan, Golub, and LeVeque 1983 [1].

    Parameters
    ----------
    priors : DNDarray
        Prior probabilities of the classes, with shape ``(n_classes,)``. If specified, the priors are not
        adjusted according to the data.
    var_smoothing : float, optional
        Portion of the largest variance of all features that is added to
        variances for calculation stability.

    Attributes
    ----------
    class_count_ : DNDarray
        Number of training samples observed in each class. Shape = ``(n_classes,)``
    class_prior_ : DNDarray
        Probability of each class. Shape = ``(n_classes,)``
    classes_ : DNDarray
        Class labels known to the classifier. Shape = ``(n_classes,)``
    epsilon_ : float
        Absolute additive value to variances
    sigma_ : DNDarray
        Variance of each feature per class. Shape = ``(n_classes, n_features)``
    theta_ : DNDarray
        Mean of each feature per class. Shape = ``(n_classes, n_features)``

    References
    ----------
    [1] Chan, Tony F., Golub, Gene H., and Leveque, Randall J., "Algorithms for Computing the Sample Variance: Analysis
    and Recommendations", The American Statistician, 37:3, pp. 242-247, 1983

    Examples
    --------
    >>> import heat as ht
    >>> X = ht.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=ht.float32)
    >>> Y = ht.array([1, 1, 1, 2, 2, 2])
    >>> from heat.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    <heat.naive_bayes.gaussianNB.GaussianNB object at 0x1a249f6dd8>
    >>> print(clf.predict(ht.array([[-0.8, -1]])))
    tensor([1])
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, ht.unique(Y, sorted=True))
    <heat.naive_bayes.gaussianNB.GaussianNB object at 0x1a249fbe10>
    >>> print(clf_pf.predict(ht.array([[-0.8, -1]])))
    tensor([1])

    ### Ancestors (in MRO)

    * heat.core.base.ClassificationMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, x: DNDarray, y: DNDarray, sample_weight: Optional[DNDarray] = None)`
    :   Fit Gaussian Naive Bayes according to ``x`` and ``y``

        Parameters
        ----------
        x : DNDarray
            Training set, where n_samples is the number of samples
            and n_features is the number of features.  Shape = (n_classes, n_features)
        y : DNDarray
            Labels for training set. Shape = (n_samples, )
        sample_weight : DNDarray, optional
            Weights applied to individual samples (1. for unweighted). Shape = (n_samples, )

    `logsumexp(self, a: DNDarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, b: Optional[DNDarray] = None, keepdims: bool = False, return_sign: bool = False) ‑> heat.core.dndarray.DNDarray`
    :   Adapted to HeAT from scikit-learn.
        Compute the log of the sum of exponentials of input elements. The result, ``np.log(np.sum(np.exp(a)))``
        calculated in a numerically more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.

        Parameters
        ----------
        a : DNDarray
            Input array.
        axis : None or int or Tuple [int,...], optional
            Axis or axes over which the sum is taken. By default ``axis`` is ``None``,
            and all elements are summed.
        keepdims : bool, optional
            If this is set to ``True``, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the original array.
        b : DNDarray, optional
            Scaling factor for ``exp(a)`` must be of the same shape as ``a`` or
            broadcastable to ``a``. These values may be negative in order to
            implement subtraction.
        return_sign : bool, optional
            If this is set to ``True``, the result will be a pair containing sign
            information; if ``False``, results that are negative will be returned
            as ``NaN``.
            #TODO: returns NotImplementedYet error.
        sgn : DNDarray, NOT IMPLEMENTED YET
            #TODO If return_sign is True, this will be an array of floating-point
            numbers matching res and +1, 0, or -1 depending on the sign
            of the result. If ``False``, only one result is returned.

    `partial_fit(self, x: DNDarray, y: DNDarray, classes: Optional[DNDarray] = None, sample_weight: Optional[DNDarray] = None)`
    :   Adapted to HeAT from scikit-learn.
        Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.
        This is especially useful when the whole dataset is too big to fit in
        memory at once.
        This method has some performance and numerical stability overhead,
        hence it is better to call :func:`partial_fit` on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        x : DNDarray
            Training set, where `n_samples` is the number of samples and
            `n_features` is the number of features. Shape = (n_samples, n_features)
        y : DNDarray
            Labels for training set. Shape = (n_samples,)
        classes : DNDarray, optional
            List of all the classes that can possibly appear in the ``y`` vector.
            Must be provided at the first call to :func:`partial_fit`, can be omitted
            in subsequent calls. Shape = ``(n_classes,)``
        sample_weight : DNDarray, optional
            Weights applied to individual samples (1. for unweighted). Shape = (n_samples,)

    `predict(self, x: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Adapted to HeAT from scikit-learn.
        Perform classification on a tensor of test data ``x``.

        Parameters
        ----------
        x : DNDarray
            Input data with shape (n_samples, n_features)

    `predict_log_proba(self, x: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Adapted to HeAT from scikit-learn.
        Return log-probability estimates of the samples for each class in
        the model. The columns correspond to the classes in sorted
        order, as they appear in the attribute ``classes_``.

        Parameters
        ----------
        x : DNDarray
            Input data. Shape = (n_samples, n_features).

    `predict_proba(self, x: DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Adapted to HeAT from scikit-learn.
        Return probability estimates for the test tensor x of the samples for each class in
        the model. The columns correspond to the classes in sorted
        order, as they appear in the attribute ``classes_``.

        Parameters
        ----------
        x : DNDarray
            Input data. Shape = (n_samples, n_features).
