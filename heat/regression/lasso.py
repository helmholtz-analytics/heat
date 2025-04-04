"""
Implementation of the LASSO regression
"""

import heat as ht
from heat.core.dndarray import DNDarray
from typing import Union, Optional


class Lasso(ht.RegressionMixin, ht.BaseEstimator):
    """
    ``Least absolute shrinkage and selection operator``(LASSO), a linear model with L1 regularization. The optimization
    objective for Lasso is:

    .. math:: E(w) =  \\frac{1}{2 m} ||y - Xw||^2_2 + \\lambda  ||w\\_||_1

    with

    .. math:: w\\_=(w_1,w_2,...,w_n),  w=(w_0,w_1,w_2,...,w_n),
    .. math:: y \\in M(m \\times 1), w \\in M(n \\times 1), X \\in M(m \\times n)

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
    >>> y = ht.random.randn(10,1, split=0)
    >>> estimator = ht.regression.lasso.Lasso(max_iter=100, tol=None)
    >>> estimator.fit(X, y)
    """

    def __init__(
        self, lam: Optional[float] = 0.1, max_iter: Optional[int] = 100, tol: Optional[float] = 1e-6
    ) -> None:
        """Initialize lasso parameters"""
        self.__lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.__theta = None
        self.n_iter = None

    @property
    def coef_(self) -> Union[type(None), DNDarray]:
        """Returns coefficients"""
        if self.__theta is None:
            return None
        else:
            return self.__theta[1:]

    @property
    def intercept_(self) -> Union[type(None), DNDarray]:
        """Returns bias term"""
        if self.__theta is None:
            return None
        else:
            return self.__theta[0]

    @property
    def lam(self) -> float:
        """Returns regularization term lambda"""
        return self.__lam

    @lam.setter
    def lam(self, arg: float) -> None:
        self.__lam = arg

    @property
    def theta(self):
        """Returns regularization term lambda"""
        return self.__theta

    def soft_threshold(self, rho: DNDarray) -> Union[DNDarray, float]:
        """
        Soft threshold operator

        Parameters
        ----------
        rho : DNDarray
            Input model data, Shape = (1,)
        out : DNDarray or float
            Thresholded model data, Shape = (1,)
        """
        if rho < -self.__lam:
            return rho + self.__lam
        elif rho > self.__lam:
            return rho - self.__lam
        else:
            return 0.0

    def rmse(self, gt: DNDarray, yest: DNDarray) -> DNDarray:
        """
        Root mean square error (RMSE)

        Parameters
        ----------
        gt : DNDarray
            Input model data, Shape = (1,)
        yest : DNDarray
            Thresholded model data, Shape = (1,)
        """
        return ht.sqrt((ht.mean((gt - yest) ** 2))).larray.item()

    def fit(self, x: DNDarray, y: DNDarray) -> None:
        """
        Fit lasso model with coordinate descent

        Parameters
        ----------
        x : DNDarray
            Input data, Shape = (n_samples, n_features)
        y : DNDarray
            Labels, Shape = (n_samples,)
        """
        # Get number of model parameters
        _, n = x.shape

        if y.ndim > 2:
            raise ValueError(f"y.ndim must <= 2, currently: {y.ndim}")
        if x.ndim != 2:
            raise ValueError(f"X.ndim must == 2, currently: {x.ndim}")

        if len(y.shape) == 1:
            y = ht.expand_dims(y, axis=1)

        # Initialize model parameters
        theta = ht.zeros((n, 1), dtype=float, device=x.device)

        # Looping until max number of iterations or convergence
        for i in range(self.max_iter):
            theta_old = theta.copy()

            # Looping through each coordinate
            for j in range(n):
                X_j = ht.array(x.larray[:, j : j + 1], is_split=0, device=x.device, comm=x.comm)

                y_est = x @ theta
                theta_j = theta.larray[j].item()

                rho = (X_j * (y - y_est + theta_j * X_j)).mean()

                # Intercept parameter theta[0] not be regularized
                if j == 0:
                    theta[j] = rho
                else:
                    theta[j] = self.soft_threshold(rho)

            diff = self.rmse(theta, theta_old)
            if self.tol is not None and diff < self.tol:
                self.n_iter = i + 1
                self.__theta = theta
                break

        self.n_iter = i + 1
        self.__theta = theta

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Apply lasso model to input data. First row data corresponds to interception

        Parameters
        ----------
        x : DNDarray
            Input data, Shape = (n_samples, n_features)
        """
        return x @ self.__theta
