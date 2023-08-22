"""
Distributed Gaussian Naive-Bayes classifier.
"""
from __future__ import annotations

from typing import Tuple, Union, Optional
import heat as ht
from heat.core.dndarray import DNDarray
import torch


class GaussianNB(ht.ClassificationMixin, ht.BaseEstimator):
    """
    Gaussian Naive Bayes (GaussianNB), based on `scikit-learn.naive_bayes.GaussianNB <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`_.
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
    """

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, x: DNDarray, y: DNDarray, sample_weight: Optional[DNDarray] = None):
        """
        Fit Gaussian Naive Bayes according to ``x`` and ``y``

        Parameters
        ----------
        x : DNDarray
            Training set, where n_samples is the number of samples
            and n_features is the number of features.  Shape = (n_classes, n_features)
        y : DNDarray
            Labels for training set. Shape = (n_samples, )
        sample_weight : DNDarray, optional
            Weights applied to individual samples (1. for unweighted). Shape = (n_samples, )
        """
        # sanitize input - to be moved to sanitation module, cf. #468
        if not isinstance(x, ht.DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")
        if not isinstance(y, ht.DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(y)}")
        if y.ndim != 1:
            raise ValueError(f"expected y to be a 1-D tensor, is {y.ndim}-D")
        if sample_weight is not None and not isinstance(sample_weight, ht.DNDarray):
            raise ValueError(
                f"sample_weight needs to be a ht.DNDarray, but was {type(sample_weight)}"
            )
        classes = ht.unique(y, sorted=True)
        if classes.split is not None:
            classes = ht.resplit(classes, axis=None)

        return self.__partial_fit(x, y, classes, _refit=True, sample_weight=sample_weight)

    def __check_partial_fit_first_call(self, classes: Optional[DNDarray] = None) -> bool:
        """
        Adapted to HeAT from scikit-learn.

        This function returns ``True`` if it detects that this was the first call to
        :meth:`partial_fit` on :class:`GaussianNB`. In that case the :attr:`classes_` attribute is also
        set on :class:`GaussianNB`.
        """
        if getattr(self, "classes_", None) is None and classes is None:
            raise ValueError("classes must be passed on the first call " "to partial_fit.")

        elif classes is not None:
            unique_labels = classes
            if getattr(self, "classes_", None) is None:
                self.classes_ = unique_labels
                # This is the first call to partial_fit
                return True
            if not ht.equal(self.classes_, unique_labels):
                raise ValueError(
                    f"`classes={classes}` is not the same as on last call to partial_fit, was: {self.classes_}"
                )
        # classes is None and self.classes_ has already previously been set:
        # nothing to do
        return False

    @staticmethod
    def __update_mean_variance(
        n_past: int,
        mu: DNDarray,
        var: DNDarray,
        x: DNDarray,
        sample_weight: Optional[DNDarray] = None,
    ) -> Tuple[DNDarray, DNDarray]:
        """
        Adapted to HeAT from scikit-learn.
        Compute online update of Gaussian mean and variance.
        Given starting sample count, mean, and variance, a new set of
        points ``x``, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in ``x`` is treated as independent
        -- you get variance, not covariance).
        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.
        See Chan, Golub, and LeVeque 1983 [1]

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.
        mu : DNDarray
            Means for Gaussians in original set. Shape = (number of Gaussians,)
        var : DNDarray
            Variances for Gaussians in original set. Shape = (number of Gaussians,)
        x : DNDarray
            Input data
        sample_weight : DNDarray, optional
            Weights applied to individual samples (1. for unweighted). Shape = (n_samples,)

        References
        ----------
        [1] Chan, Tony F., Golub, Gene H., and Leveque, Randall J., "Algorithms for Computing the Sample Variance: Analysis
        and Recommendations", The American Statistician, 37:3, pp. 242-247, 1983
        """
        if x.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        # TODO:Issue #351 allow weighted average across multiple axes
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = ht.average(x, axis=0, weights=sample_weight)
            new_var = ht.average((x - new_mu) ** 2, axis=0, weights=sample_weight)
        else:
            n_new = x.shape[0]
            new_var = ht.var(x, axis=0)
            new_mu = ht.mean(x, axis=0)

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

    def partial_fit(
        self,
        x: DNDarray,
        y: DNDarray,
        classes: Optional[DNDarray] = None,
        sample_weight: Optional[DNDarray] = None,
    ):
        """
        Adapted to HeAT from scikit-learn.
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
        """
        return self.__partial_fit(x, y, classes, _refit=False, sample_weight=sample_weight)

    def __partial_fit(
        self,
        x: DNDarray,
        y: DNDarray,
        classes: Optional[DNDarray] = None,
        _refit: bool = False,
        sample_weight: Optional[DNDarray] = None,
    ):
        """
        Actual implementation of Gaussian NB fitting. Adapted to HeAT from scikit-learn.

        Parameters
        ----------
        x : DNDarray
            Training set, where n_samples is the number of samples and
            n_features is the number of features. Shape = (n_samples, n_features)
        y : DNDarray
            Labels for training set. Shape = (n_samples,)
        classes : DNDarray, optional
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to :func:`partial_fit`, can be omitted
            in subsequent calls. Shape = (n_classes,)
        _refit : bool, optional
            If ``True``, act as though this were the first time :func:`__partial_fit` is called
            (ie, throw away any past fitting and start over).
        sample_weight : DNDarray, optional
            Weights applied to individual samples (1. for unweighted). Shape = (n_samples,)
        """
        # TODO: sanitize x and y shape: sanitation/validation module, cf. #468
        n_samples = x.shape[0]
        if x.ndim != 2:
            raise ValueError(f"expected x to be a 2-D tensor, is {x.ndim}-D")
        if y.shape[0] != n_samples:
            raise ValueError(
                f"y.shape[0] must match number of samples {n_samples}, is {y.shape[0]}"
            )

        # TODO: sanitize sample_weight: sanitation/validation module, cf. #468
        if sample_weight is not None:
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D tensor")
            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    f"sample_weight.shape == {sample_weight.shape}, expected {(n_samples, )}!"
                )

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon_ = self.var_smoothing * ht.var(x, axis=0).max()

        if _refit:
            self.classes_ = None

        if self.__check_partial_fit_first_call(classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_features = x.shape[1]
            n_classes = len(self.classes_)
            self.theta_ = ht.zeros((n_classes, n_features), dtype=x.dtype, device=x.device)
            self.sigma_ = ht.zeros((n_classes, n_features), dtype=x.dtype, device=x.device)

            self.class_count_ = ht.zeros(
                (x.comm.size, n_classes), dtype=ht.float64, device=x.device, split=0
            )
            # Initialise the class prior
            # Take into account the priors
            if self.priors is not None:
                if not isinstance(self.priors, ht.DNDarray):
                    priors = ht.array(self.priors, dtype=x.dtype, split=None, device=x.device)
                else:
                    priors = self.priors
                # Check that the provide prior match the number of classes
                if len(priors) != n_classes:
                    raise ValueError("Number of priors must match number of" " classes.")
                # Check that the sum is 1
                if not ht.isclose(priors.sum(), ht.array(1.0, dtype=priors.dtype)):
                    raise ValueError("The sum of the priors should be 1.")
                # Check that the prior are non-negative
                if (priors < 0).any():
                    raise ValueError("Priors must be non-negative.")
                self.class_prior_ = priors
            else:
                # Initialize the priors to zeros for each class
                self.class_prior_ = ht.zeros(
                    len(self.classes_), dtype=ht.float64, split=None, device=x.device
                )
        else:
            if x.shape[1] != self.theta_.shape[1]:
                raise ValueError(
                    f"Number of features {x.shape[1]} does not match previous data {self.theta_.shape[1]}."
                )
            # Put epsilon back in each time
            self.sigma_[:, :] -= self.epsilon_

        classes = self.classes_

        unique_y = ht.unique(y, sorted=True).resplit_(None)
        unique_y_in_classes = ht.eq(unique_y, classes)

        if not ht.all(unique_y_in_classes):
            raise ValueError(
                f"The target label(s) {unique_y[~unique_y_in_classes]} in y do not exist in the initial classes {classes}"
            )
        # from now on: extract torch tensors for local operations
        # DNDarrays for distributed operations only
        for y_i in unique_y.larray:
            # assuming classes.split is None
            if y_i in classes.larray:
                i = torch.where(classes.larray == y_i)[0].item()
            else:
                classes_ext = torch.cat((classes.larray, y_i.larray.unsqueeze(0)))
                i = torch.argsort(classes_ext)[-1].item()
            where_y_i = torch.where(y.larray == y_i)[0]
            X_i = x[where_y_i, :]

            if sample_weight is not None:
                sw_i = sample_weight[where_y_i]
                if 0 not in sw_i.shape:
                    N_i = sw_i.sum().item()
                else:
                    N_i = 0.0
                    sw_i = None
            else:
                sw_i = None
                N_i = X_i.shape[0]

            new_theta, new_sigma = self.__update_mean_variance(
                self.class_count_.larray[:, i].item(),
                self.theta_[i, :],
                self.sigma_[i, :],
                X_i,
                sw_i,
            )
            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.class_count_.larray[:, i] += N_i

        self.sigma_[:, :] += self.epsilon_

        # Update only if no priors are provided
        if self.priors is None:
            # distributed class_count_: sum along distribution axis
            self.class_count_ = self.class_count_.sum(axis=0, keepdims=True)
            # Empirical prior, with sample_weight taken into account
            self.class_prior_ = (self.class_count_ / self.class_count_.sum()).squeeze(0)

        return self

    def __joint_log_likelihood(self, x: DNDarray) -> DNDarray:
        """
        Adapted to HeAT from scikit-learn.
        Calculates joint log-likelihood for `n_samples` to be assigned to each class.
        Returns a ``DNDarray`` `joint_log_likelihood(n_samples, n_classes)`.
        """
        jll_size = self.classes_.larray.numel()
        jll_shape = (x.shape[0], jll_size)
        joint_log_likelihood = ht.empty(jll_shape, dtype=x.dtype, split=x.split, device=x.device)
        for i in range(jll_size):
            jointi = ht.log(self.class_prior_[i])
            n_ij = -0.5 * ht.sum(ht.log(2.0 * ht.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * ht.sum(((x - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1)
            joint_log_likelihood[:, i] = jointi + n_ij
        return joint_log_likelihood

    def logsumexp(
        self,
        a: DNDarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        b: Optional[DNDarray] = None,
        keepdims: bool = False,
        return_sign: bool = False,
    ) -> DNDarray:
        """
        Adapted to HeAT from scikit-learn.
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
        """
        if b is not None:
            raise NotImplementedError("Not implemented for weighted logsumexp")

        a_max = ht.max(a, axis=axis, keepdims=True)

        # TODO: sanitize a_max / implement isfinite(): sanitation module, cf. #468
        # if a_max.ndim > 0:
        #     a_max[~np.isfinite(a_max)] = 0
        # elif not np.isfinite(a_max):
        #     a_max = 0

        # TODO: reinstate after allowing b not None
        # if b is not None:
        #     b = np.asarray(b)
        #     tmp = b * np.exp(a - a_max)
        # else:
        tmp = ht.exp(a - a_max)

        s = ht.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            raise NotImplementedError("Not implemented for return_sign")
            # sgn = np.sign(s)  # TODO: np.sign
            # s *= sgn  # /= makes more sense but we need zero -> zero
        out = ht.log(s)

        if not keepdims:
            a_max = ht.squeeze(a_max, axis=axis)
        out += a_max

        # if return_sign: #TODO: np.sign
        #    return out, sgn
        # else:
        return out

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Adapted to HeAT from scikit-learn.
        Perform classification on a tensor of test data ``x``.

        Parameters
        ----------
        x : DNDarray
            Input data with shape (n_samples, n_features)
        """
        # sanitize input
        # TODO: sanitation/validation module, cf. #468
        if not isinstance(x, ht.DNDarray):
            raise ValueError(f"input needs to be a ht.DNDarray, but was {type(x)}")
        jll = self.__joint_log_likelihood(x)
        return self.classes_[ht.argmax(jll, axis=1)]

    def predict_log_proba(self, x: DNDarray) -> DNDarray:
        """
        Adapted to HeAT from scikit-learn.
        Return log-probability estimates of the samples for each class in
        the model. The columns correspond to the classes in sorted
        order, as they appear in the attribute ``classes_``.

        Parameters
        ----------
        x : DNDarray
            Input data. Shape = (n_samples, n_features).
        """
        # TODO: sanitation/validation module, cf. #468, log_prob_x must be 2D (cf. np.atleast_2D)
        jll = self.__joint_log_likelihood(x)
        log_prob_x_shape = (jll.gshape[0], 1)
        log_prob_x = ht.empty(log_prob_x_shape, dtype=jll.dtype, split=jll.split, device=jll.device)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x.larray = self.logsumexp(jll, axis=1).larray.unsqueeze(1)
        return jll - log_prob_x

    def predict_proba(self, x: DNDarray) -> DNDarray:
        """
        Adapted to HeAT from scikit-learn.
        Return probability estimates for the test tensor x of the samples for each class in
        the model. The columns correspond to the classes in sorted
        order, as they appear in the attribute ``classes_``.

        Parameters
        ----------
        x : DNDarray
            Input data. Shape = (n_samples, n_features).
        """
        return ht.exp(self.predict_log_proba(x))
