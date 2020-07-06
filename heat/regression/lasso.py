import heat as ht
from heat.core.dndarray import DNDarray


class Lasso(ht.RegressionMixin, ht.BaseEstimator):
    """
    `Least absolute shrinkage and selection operator` (LASSO), a linear model with :math:`L1` regularization. The optimization
    objective for Lasso is:

    .. math:: E(w) = (1 / (2 * m)) * ||y - Xw||^2_2 + lam * ||w\\_||_1

    with

    .. math:: w\\_=(w_1,w_2,...,w_n) \\land w=(w_0,w_1,w_2,...,w_n),
    .. math:: y \\in M(m \\times 1), w \\in M(n \\times 1), X \\in M(m \\times n)

    Parameters
    ----------
    lam : float, optional
        Constant that multiplies the :math:`L1` term. ``lam = 0.`` is equivalent to an Ordinary
        Least Square (OLS). For numerical reasons, using ``lam = 0.,`` with the ``Lasso`` object is not advised.
    max_iter : int, optional
        The maximum number of iterations.
    tol : float, optional.
        The tolerance for the optimization.

    Attributes
    ----------
    __theta : DNDarray
        First element is the interception parameter vector :math:`w`. Shape = (n_features + 1,),
    coef_ : DNDarray
        parameter vector (:math:`w` in the cost function formula). Shape  = (n_targets, n_features)
    intercept_ : float or DNDarray
        independent term in decision function. Shape = (n_targets,)
    n_iter_ : int or None or shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach the specified tolerance. Shape = (n_targets,)

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
        rho : DNDarray
            Input model data, Shape = (1,)
        out : DNDarray
            Thresholded model data, Shape = (1,)
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
        gt : DNDarray
            Input model data, Shape = (1,)
        yest : DNDarray
            Thresholded model data, Shape = (1,)
        """
        return ht.sqrt((ht.mean((gt - yest) ** 2)))._DNDarray__array.item()

    def fit(self, x, y):
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
            raise ValueError("y.ndim must <= 2, currently: {}".format(y.ndim))
        if x.ndim != 2:
            raise ValueError("X.ndim must == 2, currently: {}".format(x.ndim))

        if len(y.shape) == 1:
            y = ht.expand_dims(y, axis=1)

        # Initialize model parameters
        theta = ht.zeros((n, 1), dtype=float, device=x.device)

        # Looping until max number of iterations or convergence
        for i in range(self.max_iter):

            theta_old = theta.copy()

            # Looping through each coordinate
            for j in range(n):

                X_j = ht.array(x._DNDarray__array[:, j : j + 1], is_split=0)

                y_est = x @ theta
                theta_j = theta._DNDarray__array[j].item()

                rho = (X_j * (y - y_est + theta_j * X_j)).mean()

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

    def predict(self, x):
        """
        Apply lasso model to input data. First row data corresponds to interception

        Parameters
        ----------
        x : DNDarray
            Input data, Shape = (n_samples, n_features)
        """
        return x @ self.__theta
