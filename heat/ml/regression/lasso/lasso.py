import heat as ht
import torch
import numpy as np


class HeatLasso:
    """
    HeAT implementation of a the ``least absolute shrinkage and selection operator``(LASSO), a linear model with
    L1 regularization. The optimization objective for Lasso is:

    .. math:: E(w) = (1 / (2 * m)) * ||y - Xw||^2_2 + lam * ||w\\_||_1

    with

    .. math:: w\\_=(w_1,w_2,...,w_n) \\land w=(w_0,w_1,w_2,...,w_n), y \\in M(m \\times 1), w \\in M(n \\times 1), X \\in M(m \\times n)

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
    # ToDo: example to be added
    """

    def __init__(self, lam=0.1, max_iter=100, tol=1e-6):
        """initialize lasso parameters"""
        self.__lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.__theta = None
        self.n_iter = None

    @property
    def coef_(self):
        if self.__theta is None:
            return None
        else:
            return self.__theta[1:]

    @property
    def intercept_(self):
        if self.__theta is None:
            return None
        else:
            return self.__theta[0]

    @property
    def lam(self):
        return self.__lam

    @lam.setter
    def lam(self, arg):
        self.__lam = arg

    @property
    def theta(self):
        return self.__theta

    def soft_threshold(self, rho):
        """
        Soft threshold operator

        Parameters
        ----------
        rho : HeAT tensor, shape (1,)
            Input model data
        out : HeAT tensor, shape (1,)
            Thresholded model data
        """
        if rho < -self.__lam:
            return rho + self.__lam
        elif rho > self.__lam:
            return rho - self.__lam
        else:
            return 0.0

    def rmse(self, gt, yest):
        """
        Root mean square error (RMSE)

        Parameters
        ----------
        gt : HeAT tensor, shape (1,)
            Input model data
        yest : HeAT tensor, shape (1,)
            Thresholded model data
        """
        return ht.sqrt((ht.mean((gt - yest) ** 2)))._DNDarray__array.item()

    def fit(self, X, y):
        """
        Fit lasso model with coordinate descent

        Parameters
        ----------
        X : HeAT tensor, shape (n_samples, n_features)
            Input data.
        y : HeAT tensor, shape (n_samples,)
            Labels
        """
        # Get number of model parameters
        _, n = X.shape

        # Initialize model parameters
        theta = ht.zeros((n, 1), dtype=float)

        # Looping until max number of iterations or convergence
        for i in range(self.max_iter):

            theta_old = theta.copy()

            # Looping through each coordinate
            for j in range(n):

                y_est = (X @ theta)[:, 0]

                rho = (X[:, j] * (y - y_est + theta[j].copy() * X[:, j])).mean()

                # Intercept parameter theta[0] not be regularized
                if j == 0:
                    theta[j] = rho
                else:
                    theta[j] = self.soft_threshold(rho)

            diff = self.rmse(theta, theta_old)
            if self.tol is not None:
                if diff < self.tol:
                    self.n_iter = i + 1
                    self.__theta = theta
                    break

        self.n_iter = i + 1
        self.__theta = theta

    def predict(self, X):
        """
        Apply lasso model to input data. First row data corresponds to interception

        Parameters
        ----------
        X : HeAT tensor, shape (n_samples, n_features)
            Input data.
        """
        return (X @ self.__theta)[:, 0]


class NumpyLasso:
    """
    Numpy implementation of a the ``least absolute shrinkage and selection operator``(LASSO),
    a linear model with L1 regularizer.
    The optimization objective function for Lasso is:

    .. math:: (1 / (2 * m)) * ||y - Xw||^2_2 + lam * ||w\\_||_1

    with

    .. math:: w\\_=(w_1,w_2,...,w_n) \\land w=(w_0,w_1,w_2,...,w_n), y \\in M(m \\times 1), w \\in M(n \\times 1), X \\in M(m \\times n)

    Parameters
    ----------
    lam : float, optional
        Constant that multiplies the L1 term. Default value: 0.1
        ``lam = 0.`` is equivalent to an ordinary least square (OLS).
        For numerical reasons, using ``lam = 0.,`` with the ``Lasso`` object is not advised.
    max_iter : int, optional
        The maximum number of iterations. Default value: 100
    tol : float, optional. Default value: 1e-8
        The tolerance for the optimization.

    Attributes
    ----------
    __theta : array, shape (n_features + 1,), first element is the interception
        parameter vector (w in the cost function formula)
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)
    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.
    n_iter_ : int or None | array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    Examples
    --------
    # ToDo: example to be added
    """

    def __init__(self, lam=0.1, max_iter=100, tol=1e-6):
        """initialize lasso parameters"""
        self.__lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.__theta = None
        self.n_iter = None

    @property
    def coef_(self):
        if self.__theta is None:
            return None
        else:
            return self.__theta[1:]

    @property
    def intercept_(self):
        if self.__theta is None:
            return None
        else:
            return self.__theta[0]

    @property
    def lam(self):
        return self.__lam

    @lam.setter
    def lam(self, arg):
        self.__lam = arg

    @property
    def theta(self):
        return self.__theta

    def soft_threshold(self, rho):
        """
        Soft threshold operator

        Parameters
        ----------
        rho : HeAT tensor, shape (1,)
            Input model data
        out : HeAT tensor, shape (1,)
            Thresholded model data
        """
        if rho < -self.__lam:
            return rho + self.__lam
        elif rho > self.__lam:
            return rho - self.__lam
        else:
            return 0.0

    def rmse(self, gt, yest):
        """
        Root mean square error (RMSE)

        Parameters
        ----------
        gt : HeAT tensor, shape (1,)
            Input model data
        yest : HeAT tensor, shape (1,)
            Thresholded model data
        """
        return np.sqrt((np.mean((gt - yest) ** 2))).item()

    def fit(self, X, y):
        """
        Fit lasso model with coordinate descent

        Parameters
        ----------
        X : HeAT tensor, shape (n_samples, n_features+1)
            Input data.
        y : HeAT tensor, shape (n_samples,)
            Labels
        """
        # Get number of model parameters
        _, n = X.shape

        # Initialize model parameters
        theta = np.zeros((n, 1), dtype=float)

        # Looping until max number of iterations or converegence
        for i in range(self.max_iter):
            # Looping through each coordinate

            theta_old = theta.copy()

            for j in range(n):
                y_est = (X @ theta)[:, 0]

                rho = (X[:, j] * (y - y_est + theta[j] * X[:, j])).mean()

                # Intercept parameter theta[0] not be regularized
                if j == 0:
                    theta[j] = rho
                else:
                    theta[j] = self.soft_threshold(rho)

            diff = self.rmse(theta, theta_old)
            if self.tol is not None:
                if diff < self.tol:
                    self.n_iter = i + 1
                    self.__theta = theta
                    break

        self.n_iter = i + 1
        self.__theta = theta

    def predict(self, X):
        """
        Apply lasso model to input data. First row data corresponds to interception parameter

        Parameters
        ----------
        X : HeAT tensor, shape (n_samples, n_features+1)
            Input data.
        """
        return (X @ self.__theta)[:, 0]


class PytorchLasso:
    """
    PyTorch implementation of a the ``least absolute shrinkage and selection operator``(LASSO), a linear model with
    L1 regularization. The optimization objective for Lasso is:

    .. math:: E(w) = (1 / (2 * m)) * ||y - Xw||^2_2 + lam * ||w\\_||_1

    with

    .. math:: w\\_=(w_1,w_2,...,w_n) \\land w=(w_0,w_1,w_2,...,w_n), y \\in M(m \\times 1), w \\in M(n \\times 1), X \\in M(m \\times n)

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
    # ToDo: example to be added
    """

    def __init__(self, lam=0.1, max_iter=100, tol=1e-6):
        """initialize lasso parameters"""
        self.__lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.__theta = None
        self.n_iter = None

    @property
    def coef_(self):
        if self.__theta is None:
            return None
        else:
            return self.__theta[1:]

    @property
    def intercept_(self):
        if self.__theta is None:
            return None
        else:
            return self.__theta[0]

    @property
    def lam(self):
        return self.__lam

    @lam.setter
    def lam(self, arg):
        self.__lam = arg

    @property
    def theta(self):
        return self.__theta

    def soft_threshold(self, rho):
        """
        Soft threshold operator

        Parameters
        ----------
        rho : HeAT tensor, shape (1,)
            Input model data
        out : HeAT tensor, shape (1,)
            Thresholded model data
        """
        if rho < -self.__lam:
            return rho + self.__lam
        elif rho > self.__lam:
            return rho - self.__lam
        else:
            return 0.0

    def rmse(self, gt, yest):
        """
        Root mean square error (RMSE)

        Parameters
        ----------
        gt : HeAT tensor, shape (1,)
            Input model data
        yest : HeAT tensor, shape (1,)
            Thresholded model data
        """
        return torch.sqrt((torch.mean((gt - yest) ** 2))).item()

    def fit(self, X, y):
        """
        Fit lasso model with coordinate descent

        Parameters
        ----------
        X : HeAT tensor, shape (n_samples, n_features)
            Input data.
        y : HeAT tensor, shape (n_samples,)
            Labels
        """
        # Get number of model parameters
        _, n = X.shape

        # Initialize model parameters
        theta = torch.zeros(n, 1, device=X.device)
        # Looping until max number of iterations or convergence
        for i in range(self.max_iter):

            theta_old = theta.clone()

            # Looping through each coordinate
            for j in range(n):

                y_est = (X @ theta)[:, 0]

                rho = (X[:, j] * (y - y_est + theta[j] * X[:, j])).mean()

                # Intercept parameter theta[0] not be regularized
                if j == 0:
                    theta[j] = rho
                else:
                    theta[j] = self.soft_threshold(rho)

            diff = self.rmse(theta, theta_old)
            if self.tol is not None:
                if diff < self.tol:
                    self.n_iter = i + 1
                    self.__theta = theta
                    break

        self.n_iter = i + 1
        self.__theta = theta

    def predict(self, X):
        """
        Apply lasso model to input data. First row data corresponds to interception

        Parameters
        ----------
        X : HeAT tensor, shape (n_samples, n_features)
            Input data.
        """
        return (X @ self.__theta)[:, 0]
