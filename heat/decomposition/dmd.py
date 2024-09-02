"""
Module implementing the Dynamic Mode Decomposition (DMD) algorithm.
"""

import heat as ht
from typing import Optional, Tuple, Union, List
import torch

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
    svd_solver : str, optional
        Specifies the algorithm to use for the singular value decomposition (SVD). Options are 'full' (default), 'hierarchical', and 'randomized'.
    svd_rank : int, optional
        The rank to which SVD shall be truncated. For `'full'` SVD, `svd_rank = None` together with `svd_tol = None` (default) will result in no truncation.
        For `svd_solver='full'`, at most one of `svd_rank` or `svd_tol` may be specified.
        For `svd_solver='hierarchical'`, either `svd_rank` (rank to truncate to) or `svd_tol` (tolerance to truncate to) must be specified.
        For `svd_solver='randomized'`, `svd_rank` must be specified and determines the the rank to truncate to.
    svd_tol : float, optional
        The tolerance to which SVD shall be truncated. For `'full'` SVD, `svd_tol = None` together with `svd_rank = None` (default) will result in no truncation.
        For `svd_solver='hierarchical'`, either `svd_tol` (accuracy to truncate to) or `svd_rank` (rank to truncate to) must be specified.
        For `svd_solver='randomized'`, `svd_tol` is meaningless and must be None.

    Attributes
    ----------
    svd_solver : str
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
        svd_solver: Optional[str] = "full",
        svd_rank: Optional[int] = None,
        svd_tol: Optional[float] = None,
    ):
        # Check if the specified SVD algorithm is valid
        if not isinstance(svd_solver, str):
            raise TypeError(
                f"Invalid type '{type(svd_solver)}' for 'svd_solver'. Must be a string."
            )
        # check if the specified SVD algorithm is valid
        if svd_solver not in ["full", "hierarchical", "randomized"]:
            raise ValueError(
                f"Invalid SVD algorithm '{svd_solver}'. Must be one of 'full', 'hierarchical', 'randomized'."
            )
        # check if the respective algorithm got the right combination of non-None parameters
        if svd_solver == "full" and svd_rank is not None and svd_tol is not None:
            raise ValueError(
                "For 'full' SVD, at most one of 'svd_rank' or 'svd_tol' may be specified."
            )
        if svd_solver == "hierarchical":
            if svd_rank is None and svd_tol is None:
                raise ValueError(
                    "For 'hierarchical' SVD, exactly one of 'svd_rank' or 'svd_tol' must be specified, but none of them is specified."
                )
            if svd_rank is not None and svd_tol is not None:
                raise ValueError(
                    "For 'hierarchical' SVD, exactly one of 'svd_rank' or 'svd_tol' must be specified, but currently both are specified."
                )
        if svd_solver == "randomized":
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
        self.svd_solver = svd_solver
        self.svd_rank = svd_rank
        self.svd_tol = svd_tol
        self.rom_basis_ = None
        self.rom_transfer_matrix_ = None
        self.rom_eigenvalues_ = None
        self.rom_eigenmodes_ = None
        self.dmdmodes_ = None
        self.n_modes_ = None

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
        # first step of DMD: compute the SVD of the input data from first to second last time step
        if self.svd_solver == "full" or not X.is_distributed():
            U, S, V = ht.linalg.svd(X[:, :-1], full_matrices=False)
            if self.svd_tol is not None:
                # truncation w.r.t. prescribed bound on explained variance
                # determine svd_rank accordingly
                total_variance = (S**2).sum()
                variance_threshold = self.svd_tol * total_variance.larray.item()
                variance_cumsum = (S**2).larray.cumsum(0)
                self.n_modes_ = len(variance_cumsum[variance_cumsum <= variance_threshold]) + 1
            elif self.svd_rank is not None:
                # truncation w.r.t. prescribed rank
                self.n_modes_ = self.svd_rank
            else:
                # no truncation
                self.n_modes_ = S.shape[0]
            self.rom_basis_ = U[:, : self.n_modes_]
            V = V[:, : self.n_modes_]
            S = S[: self.n_modes_]
        # compute SVD via "hierarchical" SVD
        elif self.svd_solver == "hierarchical":
            if self.svd_tol is not None:
                # hierarchical SVD with prescribed upper bound on relative error
                U, S, V, _ = ht.linalg.hsvd_rtol(
                    X[:, :-1], self.svd_tol, compute_sv=True, safetyshift=5
                )
            else:
                # hierarchical SVD with prescribed, fixed rank
                U, S, V, _ = ht.linalg.hsvd_rank(
                    X[:, :-1], self.svd_rank, compute_sv=True, safetyshift=5
                )
            self.rom_basis_ = U
            self.n_modes_ = U.shape[1]
        else:
            # compute SVD via "randomized" SVD
            U, S, V = ht.linalg.rsvd(
                X[:, :-1],
                self.svd_rank,
            )
            self.rom_basis_ = U
            self.n_modes_ = U.shape[1]
        # second step of DMD: compute the reduced order model transfer matrix
        # we need to assume that the the transfer matrix of the ROM is small enough to fit into memory of one process
        self.rom_transfer_matrix_ = self.rom_basis_.T @ X[:, 1:] @ V / S
        self.rom_transfer_matrix_.resplit_(None)
        # third step of DMD: compute the reduced order model eigenvalues and eigenmodes
        eigvals_loc, eigvec_loc = torch.linalg.eig(self.rom_transfer_matrix_.larray)
        self.rom_eigenvalues_ = ht.array(eigvals_loc, split=None)
        self.rom_eigenmodes_ = ht.array(eigvec_loc, split=None)
        self.dmdmodes_ = self.rom_basis_ @ self.rom_eigenmodes_

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
