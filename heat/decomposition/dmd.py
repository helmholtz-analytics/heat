"""
Module implementing the Dynamic Mode Decomposition (DMD) algorithm.
"""

import heat as ht
from typing import Optional, Tuple, Union, List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class DMD(ht.RegressionMixin, ht.BaseEstimator):
    """
    Dynamic Mode Decomposition (DMD), plain vanilla version with SVD-based implementation.

    The time series of which DMD shall be computed must be given as a 2-D DNDarray of shape (n_features, n_timesteps).

    Parameters
    ----------
    svd_algorithm : str, optional
        Specifies the algorithm to use for the singular value decomposition (SVD). Options are 'full' (default), 'hierarchical', and 'randomized'.
    svd_rank : int, optional
        The rank to which SVD shall be truncated. For `'full'` SVD, `svd_rank = None` together with `svd_tol = None` (default) will result in no truncation.
        For `svd_algorithm='full'`, at most one of `svd_rank` or `svd_tol` may be specified.
        For `svd_algorithm='hierarchical'`, either `svd_rank` (rank to truncate to) or `svd_tol` (tolerance to truncate to) must be specified.
        For `svd_algorithm='randomized'`, `svd_rank` must be specified and determines the the rank to truncate to.
    svd_tol : float, optional
        The tolerance to which SVD shall be truncated. For `'full'` SVD, `svd_tol = None` together with `svd_rank = None` (default) will result in no truncation.
        For `svd_algorithm='hierarchical'`, either `svd_tol` (accuracy to truncate to) or `svd_rank` (rank to truncate to) must be specified.
        For `svd_algorithm='randomized'`, `svd_tol` is meaningless and must be None.

    Attributes
    ----------
    svd_algorithm : str
        The algorithm used for the singular value decomposition (SVD).
    svd_rank : int
        The rank to which SVD shall be truncated.
    svd_tol : float
        The tolerance to which SVD shall be truncated.
    rom_basis_ : DNDarray
        The reduced order model basis.
    rom_transfer_matrix_ : DNDarray
        The reduced order model transfer matrix.
    rom_eigenvalues_ : DNDarray
        The reduced order model eigenvalues.
    rom_eigenmodes_ : DNDarray
        The reduced order model eigenmodes ("DMD modes")

    Notes
    ----------
    We follow the "exact DMD" method as described in [1], Sect. 2.2.

    References
    ----------
    [1] J. L. Proctor, S. L. Brunton, and J. N. Kutz, "Dynamic Mode Decomposition with Control," SIAM Journal on Applied Dynamical Systems, vol. 15, no. 1, pp. 142-161, 2016.
    """

    def __init__(
        self,
        svd_algorithm: Optional[str] = "full",
        svd_rank: Optional[int] = None,
        svd_tol: Optional[float] = None,
    ):
        # Check if the specified SVD algorithm is valid
        if not isinstance(svd_algorithm, str):
            raise TypeError(
                f"Invalid type '{type(svd_algorithm)}' for 'svd_algorithm'. Must be a string."
            )
        # check if the specified SVD algorithm is valid
        if svd_algorithm not in ["full", "hierarchical", "randomized"]:
            raise ValueError(
                f"Invalid SVD algorithm '{svd_algorithm}'. Must be one of 'full', 'hierarchical', 'randomized'."
            )
        # check if the respective algorithm got the right combination of non-None parameters
        if svd_algorithm == "full" and svd_rank is not None and svd_tol is not None:
            raise ValueError(
                "For 'full' SVD, at most one of 'svd_rank' or 'svd_tol' may be specified."
            )
        if svd_algorithm == "hierarchical":
            if svd_rank is None and svd_tol is None:
                raise ValueError(
                    "For 'hierarchical' SVD, exactly one of 'svd_rank' or 'svd_tol' must be specified, but none of them is specified."
                )
            if svd_rank is not None and svd_tol is not None:
                raise ValueError(
                    "For 'hierarchical' SVD, exactly one of 'svd_rank' or 'svd_tol' must be specified, but currently both are specified."
                )
        if svd_algorithm == "randomized":
            if svd_rank is None:
                raise ValueError("For 'randomized' SVD, 'svd_rank' must be specified.")
            if svd_tol is not None:
                raise ValueError("For 'randomized' SVD, 'svd_tol' must be None.")
        # check correct data types of non-None parameters
        if svd_rank is not None:
            if not isinstance(svd_rank, int):
                raise TypeError(
                    f"Invalid type '{type(svd_rank)}' for 'svd_rank'. Must be an integer."
                )
            if svd_rank < 1:
                raise ValueError(
                    f"Invalid value '{svd_rank}' for 'svd_rank'. Must be a positive integer."
                )
        if svd_tol is not None:
            if not isinstance(svd_tol, float):
                raise TypeError(f"Invalid type '{type(svd_tol)}' for 'svd_tol'. Must be a float.")
            if svd_tol <= 0:
                raise ValueError(f"Invalid value '{svd_tol}' for 'svd_tol'. Must be non-negative.")
        # set or initialize the attributes
        self.svd_algorithm = svd_algorithm
        self.svd_rank = svd_rank
        self.svd_tol = svd_tol
        self.rom_basis_ = None
        self.rom_transfer_matrix_ = None
        self.rom_eigenvalues_ = None
        self.rom_eigenmodes_ = None
        return self

    def fit(self, X: ht.DNDarray) -> Self:
        """
        Fits the DMD model to the given data.

        Parameters
        ----------
        X : DNDarray
            The time series data to fit the DMD model to. Must be of shape (n_features, n_timesteps).
        """
        ht.sanitize_in(X)
        # check if the input data is a 2-D DNDarray
        if X.ndim != 2:
            raise ValueError(
                f"Invalid shape '{X.shape}' for input data 'X'. Must be a 2-D DNDarray of shape (n_features, n_timesteps)."
            )
        # check if the input data has at least two time steps
        if X.shape[1] < 2:
            raise ValueError(
                f"Invalid number of time steps '{X.shape[1]}' in input data 'X'. Must have at least two time steps."
            )
        if self.svd_algorithm == "full":
            # full SVD
            pass
        elif self.svd_algorithm == "hierarchical":
            # hierarchical SVD
            pass
        else:
            # randomized SVD
            pass

    def predict(self, X: ht.DNDarray, steps: Union[int, List, ht.DNDarray]) -> ht.DNDarray:
        """
        Predicts and returns the future states at the time steps provided in `steps` with initial condition(s) given by the input `X`.

        Parameters
        ----------
        X : DNDarray
            The initial condition(s) for the prediction. Must have the same number of features as the training data, but can be batched for multiple initial conditions,
            e.g., X can be of shape (n_features,) or (n_initial_conditions, n_features).
        steps : int, list, DNDarray
            The time steps at which to predict the future states. If `steps` is an integer, the future state(s) after `steps` time steps are predicted.
            If `steps` is a list or DNDarray, the future states at the time steps provided in the list or the DNDarray, respectively, are predicted.
        """
        pass
