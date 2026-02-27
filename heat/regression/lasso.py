"""
Implementation of the LASSO regression
"""

import math
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

    The implementation uses FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for optimization,
    as described in Beck & Teboulle (2009): "A Fast Iterative Shrinkage-Thresholding Algorithm
    for Linear Inverse Problems".

    Parameters
    ----------
    lam : float, optional
        Constant that multiplies the L1 term. Default value: 0.1 ``lam = 0.`` is equivalent to an ordinary
        least square (OLS). For numerical reasons, using ``lam = 0.,`` with the ``Lasso`` object is not advised.
    max_iter : int, optional
        The maximum number of iterations. Default value: 100
    tol : float, optional. Default value: 1e-6
        The tolerance for the optimization.
    step_size : float, optional
        The step size for the gradient descent step. If None, it will be computed as 1/L where L is the
        Lipschitz constant of the gradient (largest eigenvalue of X^T X / m). Default: None

    Attributes
    ----------
    __theta : array, shape (n_features + 1,), first element is the interception parameter vector w.
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)
    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.
    n_iter_ : int or None | array-like, shape (n_targets,)
        number of iterations run by FISTA to reach the specified tolerance.

    Examples
    --------
    >>> X = ht.random.randn(10, 4, split=0)
    >>> y = ht.random.randn(10, 1, split=0)
    >>> estimator = ht.regression.lasso.Lasso(max_iter=100, tol=None)
    >>> estimator.fit(X, y)
    """

    def __init__(
        self,
        lam: Optional[float] = 0.1,
        max_iter: Optional[int] = 100,
        tol: Optional[float] = 1e-6,
        step_size: Optional[float] = None,
    ) -> None:
        """Initialize lasso parameters"""
        self.__lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
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

    def soft_threshold(self, x: DNDarray, threshold: float) -> DNDarray:
        """
        Vectorized soft threshold operator (proximal operator for L1 norm)

        Parameters
        ----------
        x : DNDarray
            Input data
        threshold : float
            Threshold value (lambda * step_size for FISTA)

        Returns
        -------
        DNDarray
            Thresholded data
        """
        return ht.sign(x) * ht.maximum(ht.abs(x) - threshold, 0.0)

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
        Fit lasso model using FISTA (Fast Iterative Shrinkage-Thresholding
        Algorithm)

        Parameters
        ----------
        x : DNDarray
            Input data, Shape = (n_samples, n_features)
        y : DNDarray
            Labels, Shape = (n_samples,)
        """
        # Get number of samples and features
        m, n = x.shape

        if y.ndim > 2:
            raise ValueError(f"y.ndim must <= 2, currently: {y.ndim}")
        if x.ndim != 2:
            raise ValueError(f"X.ndim must == 2, currently: {x.ndim}")

        if len(y.shape) == 1:
            y = ht.expand_dims(y, axis=1)

        # Compute step size (1/L where L is Lipschitz constant of gradient)
        if self.step_size is None:
            # L = largest eigenvalue of (X^T X) / m
            # For efficiency, we approximate: L ≈ ||X||_F^2 / m
            XtX_norm = ht.linalg.norm(x) ** 2 / m
            L = XtX_norm.item()
            step = 1.0 / L if L > 0 else 1.0
        else:
            step = self.step_size

        # Initialize parameters (not split - these are model parameters)
        theta = ht.zeros((n, 1), dtype=x.dtype, split=None, device=x.device)
        y_k = theta.copy()  # Extrapolation point
        t_k = 1.0  # Momentum parameter

        # FISTA iterations
        for i in range(self.max_iter):
            theta_old = theta.copy()

            # Compute gradient at y_k: (1/m) * X^T (X y_k - y)
            residual = x @ y_k - y
            gradient = (x.T @ residual) / m

            # Gradient descent step
            z = y_k - step * gradient

            # Proximal step: soft thresholding
            # Apply soft thresholding with lambda * step_size
            theta_new = self.soft_threshold(z, self.__lam * step)

            # Don't regularize the intercept (first element)
            theta_new[0] = z[0]  # No thresholding for intercept
            theta = theta_new

            # Update momentum parameter (using math.sqrt for scalar)
            t_k_new = (1.0 + math.sqrt(1.0 + 4.0 * t_k**2)) / 2.0

            # Update extrapolation point
            y_k = theta + ((t_k - 1.0) / t_k_new) * (theta - theta_old)

            t_k = t_k_new

            # Check convergence
            diff = self.rmse(theta, theta_old)
            if self.tol is not None and diff < self.tol:
                self.n_iter = i + 1
                self.__theta = theta
                break

        self.n_iter = i + 1
        self.__theta = theta

    def predict(self, x: DNDarray) -> DNDarray:
        """
        Apply lasso model to input data. First row data corresponds to
        interception

        Parameters
        ----------
        x : DNDarray
            Input data, Shape = (n_samples, n_features)
        """
        return x @ self.__theta
