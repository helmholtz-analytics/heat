import sys
import numpy as np
import torch
import heat as ht


class GaussianNB:
    """
    Gaussian Naive Bayes (GaussianNB)
    Can perform online updates to model parameters via :meth:`partial_fit`.
    For details on algorithm used to update feature means and variance online,
    see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:
        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    Read more in the :ref:`User Guide <gaussian_naive_bayes>`.
    Parameters
    ----------
    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    var_smoothing : float, optional (default=1e-9)
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    Attributes
    ----------
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.
    class_prior_ : array, shape (n_classes,)
        probability of each class.
    classes_ : array, shape (n_classes,)
        class labels known to the classifier
    epsilon_ : float
        absolute additive value to variances
    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class
    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    >>> clf_pf = GaussianNB()
    >>> clf_pf.partial_fit(X, Y, np.unique(Y))
    GaussianNB()
    >>> print(clf_pf.predict([[-0.8, -1]]))
    [1]
    """

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        """
        Fit Gaussian Naive Bayes according to X, y
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        # sanitize input
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))
        if not isinstance(y, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(y)))
        if y.numdims != 1:
            raise ValueError("expected y to be a 1-D tensor, is {}-D".format(y.numdims))
        if sample_weight is not None:
            if not isinstance(sample_weight, ht.DNDarray):
                raise ValueError(
                    "sample_weight needs to be a ht.DNDarray, but was {}".format(
                        type(sample_weight)
                    )
                )
        classes = ht.unique(y, sorted=True)
        if classes.split is not None:
            classes = ht.resplit(classes, axis=None)

        return self.__partial_fit(X, y, classes, _refit=True, sample_weight=sample_weight)

    # @staticmethod
    def __check_partial_fit_first_call(clf, classes=None):
        """
        Private helper function for factorizing common classes param logic
        Estimators that implement the ``partial_fit`` API need to be provided with
        the list of possible classes at the first call to partial_fit.
        Subsequent calls to partial_fit should check that ``classes`` is still
        consistent with a previous value of ``clf.classes_`` when provided.
        This function returns True if it detects that this was the first call to
        ``partial_fit`` on ``clf``. In that case the ``classes_`` attribute is also
        set on ``clf``.
        """
        if getattr(clf, "classes_", None) is None and classes is None:
            raise ValueError("classes must be passed on the first call " "to partial_fit.")

        elif classes is not None:
            unique_labels = classes
            if getattr(clf, "classes_", None) is not None:
                if not ht.equal(clf.classes_, unique_labels):
                    raise ValueError(
                        "`classes=%r` is not the same as on last call "
                        "to partial_fit, was: %r" % (classes, clf.classes_)
                    )

            else:
                # This is the first call to partial_fit
                clf.classes_ = unique_labels
                return True

        # classes is None and clf.classes_ has already previously been set:
        # nothing to do
        return False

    @staticmethod
    def __update_mean_variance(n_past, mu, var, X, sample_weight=None):
        """
        Compute online update of Gaussian mean and variance.
        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).
        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.
        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:
        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.
        mu : array-like, shape (number of Gaussians,)
            Means for Gaussians in original set.
        var : array-like, shape (number of Gaussians,)
            Variances for Gaussians in original set.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        total_mu : array-like, shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.
        total_var : array-like, shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = ht.average(X, axis=0, weights=sample_weight)  # TODO:Issue #351
            new_var = ht.average((X - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = X.shape[0]
            new_var = ht.var(X, axis=0)
            new_mu = ht.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)
        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = old_ssd + new_ssd + (n_new * n_past / n_total) * (mu - new_mu) ** 2
        total_var = total_ssd / n_total

        return total_mu, total_var

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.
        This is especially useful when the whole dataset is too big to fit in
        memory at once.
        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        classes : array-like, shape (n_classes,), optional (default=None)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).
            .. versionadded:: 0.17
        Returns
        -------
        self : object
        """
        return self.__partial_fit(X, y, classes, _refit=False, sample_weight=sample_weight)

    def __partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        """
        Actual implementation of Gaussian NB fitting.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        classes : array-like, shape (n_classes,), optional (default=None)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        _refit : bool, optional (default=False)
            If true, act as though this were the first time __partial_fit is called
            (ie, throw away any past fitting and start over).
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        self : object
        """
        # sanitize X and y shape
        n_samples = X.shape[0]
        if X.numdims != 2:
            raise ValueError("expected X to be a 2-D tensor, is {}-D".format(X.numdims))
        if y.shape[0] != n_samples:
            raise ValueError(
                "y.shape[0] must match number of samples {}, is {}".format(n_samples, y.shape[0])
            )
        # TODO: more complex checks might be needed, see sklearn.utils.validation.check_X_y()
        if sample_weight is not None:
            # sanitize shape of weights
            if sample_weight.numdims != 1:
                raise ValueError("Sample weights must be 1D tensor")
            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    "sample_weight.shape == {}, expected {}!".format(
                        sample_weight.shape, (n_samples,)
                    )
                )

        # TODO possibly deeper checks needed, see sklearn.utils.validation._check_sample_weight
        # sample_weight = _check_sample_weight(sample_weight, X)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * ht.var(X, axis=0).max()

        if _refit:
            self.classes_ = None

        if self.__check_partial_fit_first_call(classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = ht.zeros((n_classes, n_features))
            self.sigma_ = ht.zeros((n_classes, n_features))

            self.class_count_ = ht.zeros((n_classes,), dtype=ht.float64)

            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                priors = ht.asarray(self.priors)
                # Check that the provide prior match the number of classes
                if len(priors) != n_classes:
                    raise ValueError("Number of priors must match number of" " classes.")
                # Check that the sum is 1
                if not ht.isclose(priors.sum(), 1.0):
                    raise ValueError("The sum of the priors should be 1.")
                # Check that the prior are non-negative
                if (priors < 0).any():
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = ht.zeros(len(self.classes_), dtype=ht.float64)
        else:
            if X.shape[1] != self.theta_.shape[1]:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            # Put epsilon back in each time
            self.sigma_[:, :] -= self.epsilon_

        classes = self.classes_

        unique_y = ht.unique(y, sorted=True)
        if unique_y.split is not None:
            unique_y = ht.resplit(unique_y, axis=None)
        unique_y_in_classes = ht.eq(unique_y, classes)

        if not ht.all(unique_y_in_classes):
            raise ValueError(
                "The target label(s) %s in y do not exist in the "
                "initial classes %s" % (unique_y[~unique_y_in_classes], classes)
            )
        for y_i in unique_y:
            # assuming classes.split is None
            if y_i in classes:
                i = ht.where(classes == y_i).item()
            else:
                classes_ext = torch.cat(
                    (classes._DNDarray__array, y_i._DNDarray__array.unsqueeze(0))
                )
                i = torch.argsort(classes_ext)[-1].item()
            X_i = X[ht.where(y == y_i)._DNDarray__array.squeeze().tolist(), :]

            if sample_weight is not None:
                sw_i = sample_weight[y == y_i]
                N_i = sw_i.sum()
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self.__update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :], X_i, sw_i
            )

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_[i] += N_i

        self.sigma_[:, :] += self.epsilon_

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def __joint_log_likelihood(self, X):
        jll_size = self.classes_._DNDarray__array.numel()
        jll_shape = (X.shape[0], jll_size)
        joint_log_likelihood = ht.empty(jll_shape, dtype=X.dtype, split=X.split, device=X.device)
        for i in range(jll_size):
            jointi = ht.log(self.class_prior_[i])
            n_ij = -0.5 * ht.sum(ht.log(2.0 * ht.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * ht.sum(((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1)
            joint_log_likelihood[:, i] = jointi + n_ij

        return joint_log_likelihood

    def logsumexp(self, a, axis=None, b=None, keepdim=False, return_sign=False):
        """
        Compute the log of the sum of exponentials of input elements.
        TODO: update sklearn docs to fit heat
        Parameters
        ----------
        a : array_like
            Input array.
        axis : None or int or tuple of ints, optional
            Axis or axes over which the sum is taken. By default `axis` is None,
            and all elements are summed.
            .. versionadded:: 0.11.0
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the original array.
            .. versionadded:: 0.15.0
        b : array-like, optional
            Scaling factor for exp(`a`) must be of the same shape as `a` or
            broadcastable to `a`. These values may be negative in order to
            implement subtraction.
            .. versionadded:: 0.12.0
        return_sign : bool, optional
            If this is set to True, the result will be a pair containing sign
            information; if False, results that are negative will be returned
            as NaN. Default is False (no sign information).
            .. versionadded:: 0.16.0
        Returns
        -------
        res : ndarray
            The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
            more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
            is returned.
        sgn : ndarray
            If return_sign is True, this will be an array of floating-point
            numbers matching res and +1, 0, or -1 depending on the sign
            of the result. If False, only one result is returned.
        See Also
        --------
        numpy.logaddexp, numpy.logaddexp2
        Notes
        -----
        NumPy has a logaddexp function which is very similar to `logsumexp`, but
        only handles two arguments. `logaddexp.reduce` is similar to this
        function, but may be less stable.
        Examples
        --------
        >>> from scipy.special import logsumexp
        >>> a = np.arange(10)
        >>> np.log(np.sum(np.exp(a)))
        9.4586297444267107
        >>> logsumexp(a)
        9.4586297444267107
        With weights
        >>> a = np.arange(10)
        >>> b = np.arange(10, 0, -1)
        >>> logsumexp(a, b=b)
        9.9170178533034665
        >>> np.log(np.sum(b*np.exp(a)))
        9.9170178533034647
        Returning a sign flag
        >>> logsumexp([1,2],b=[1,-1],return_sign=True)
        (1.5413248546129181, -1.0)
        Notice that `logsumexp` does not directly support masked arrays. To use it
        on a masked array, convert the mask into zero weights:
        >>> a = np.ma.array([np.log(2), 2, np.log(3)],
        ...                  mask=[False, True, False])
        >>> b = (~a.mask).astype(int)
        >>> logsumexp(a.data, b=b), np.log(5)
        1.6094379124341005, 1.6094379124341005
        """
        # a = _asarray_validated(a, check_finite=False)
        if b is not None:
            raise NotImplementedError("Not implemented for weighted logsumexp")
            # a, b = np.broadcast_arrays(a, b)
            # if np.any(b == 0):
            #     a = a + 0.0  # promote to at least float
            #     a[b == 0] = -np.inf

        a_max = ht.max(a, axis=axis, keepdim=True)

        # TODO: CHECK FOR FINITENESS!!
        # if a_max.numdims > 0:  # TODO: implement alias numdims --> ndim
        #     a_max[~np.isfinite(a_max)] = 0  # TODO: np.isfinite
        # elif not np.isfinite(a_max):
        #     a_max = 0

        # TODO: reinstate after allowing b not None
        # if b is not None:
        #     b = np.asarray(b)
        #     tmp = b * np.exp(a - a_max)
        # else:
        tmp = ht.exp(a - a_max)

        # suppress warnings about log of zero
        # with np.errstate(divide="ignore"): #TODO: REINSTATE?
        s = ht.sum(tmp, axis=axis, keepdim=keepdim)
        if return_sign:
            raise NotImplementedError("Not implemented for return_sign")
            # sgn = np.sign(s)  # TODO: np.sign
            # s *= sgn  # /= makes more sense but we need zero -> zero
        out = ht.log(s)

        if not keepdim:
            a_max = ht.squeeze(a_max, axis=axis)
        out += a_max

        # if return_sign: #TODO: np.sign
        #    return out, sgn
        # else:
        return out

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X
        """
        # check_is_fitted(self) #TODO sanitation module
        # X = self._check_X(X)  #TODO sanitation module
        # sanitize input
        if not isinstance(X, ht.DNDarray):
            raise ValueError("input needs to be a ht.DNDarray, but was {}".format(type(X)))
        jll = self.__joint_log_likelihood(X)
        return self.classes_[ht.argmax(jll, axis=1).numpy()]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        # check_is_fitted(self) #TODO sanitation module
        # X = self._check_X(X)  # TODO sanitation module
        jll = self.__joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = self.logsumexp(jll, axis=1)
        return (
            jll - log_prob_x.T  # np.atleast_2d(log_prob_x).T
        )  # TODO sanitation, ensure that log_prob_x is at least a 2D tensor

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in

        the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return ht.exp(self.predict_log_proba(X))
