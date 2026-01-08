Module heat.regression.lasso
============================
Implementation of the LASSO regression

Classes
-------

`Lasso(lam: float | None = 0.1, max_iter: int | None = 100, tol: float | None = 1e-06)`
:   ``Least absolute shrinkage and selection operator``(LASSO), a linear model with L1 regularization. The optimization
    objective for Lasso is:

    .. math:: E(w) =  \frac{1}{2 m} ||y - Xw||^2_2 + \lambda  ||w\_||_1

    with

    .. math:: w\_=(w_1,w_2,...,w_n),  w=(w_0,w_1,w_2,...,w_n),
    .. math:: y \in M(m \times 1), w \in M(n \times 1), X \in M(m \times n)

    Parameters
    ----------
    lam : float, optional
        Constant that multiplies the L1 term. Default value: 0.1 ``lam = 0.`` is equivalent to an ordinary
        least square (OLS). For numerical reasons, using ``lam = 0.,`` with the ``Lasso`` object is not advised.
    max_iter : int, optional
        The maximum number of iterations. Default value: 100
    tol : float, optional. Default value: 1e-8
        The tolerance for the optimization.

    Attributes
    ----------
    __theta : array, shape (n_features + 1,), first element is the interception parameter vector w.
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)
    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.
    n_iter_ : int or None | array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach the specified tolerance.

    Examples
    --------
    >>> X = ht.random.randn(10, 4, split=0)
    >>> y = ht.random.randn(10, 1, split=0)
    >>> estimator = ht.regression.lasso.Lasso(max_iter=100, tol=None)
    >>> estimator.fit(X, y)

    Initialize lasso parameters

    ### Ancestors (in MRO)

    * heat.core.base.RegressionMixin
    * heat.core.base.BaseEstimator

    ### Instance variables

    `coef_: heat.core.dndarray.DNDarray | None`
    :   Returns coefficients

    `intercept_: heat.core.dndarray.DNDarray | None`
    :   Returns bias term

    `lam: float`
    :   Returns regularization term lambda

    `theta`
    :   Returns regularization term lambda

    ### Methods

    `fit(self, x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray) ‑> None`
    :   Fit lasso model with coordinate descent

        Parameters
        ----------
        x : DNDarray
            Input data, Shape = (n_samples, n_features)
        y : DNDarray
            Labels, Shape = (n_samples,)

    `predict(self, x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Apply lasso model to input data. First row data corresponds to interception

        Parameters
        ----------
        x : DNDarray
            Input data, Shape = (n_samples, n_features)

    `rmse(self, gt: heat.core.dndarray.DNDarray, yest: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Root mean square error (RMSE)

        Parameters
        ----------
        gt : DNDarray
            Input model data, Shape = (1,)
        yest : DNDarray
            Thresholded model data, Shape = (1,)

    `soft_threshold(self, rho: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray | float`
    :   Soft threshold operator

        Parameters
        ----------
        rho : DNDarray
            Input model data, Shape = (1,)
        out : DNDarray or float
            Thresholded model data, Shape = (1,)
