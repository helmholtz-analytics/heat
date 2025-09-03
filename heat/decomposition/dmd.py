"""
Module implementing the Dynamic Mode Decomposition (DMD) algorithm.
"""

import heat as ht
from typing import Optional, Union, List
import torch

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


def _torch_matrix_diag(diagonal):
    # auxiliary function to create a batch of diagonal matrices from a batch of diagonal vectors
    # source: fmassas comment on Oct 4, 2018 in https://github.com/pytorch/pytorch/issues/12160 [Accessed Oct 09, 2024]
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


class DMD(ht.RegressionMixin, ht.BaseEstimator):
    """
    Dynamic Mode Decomposition (DMD), plain vanilla version with SVD-based implementation.

    The time series of which DMD shall be computed must be provided as a 2-D DNDarray of shape (n_features, n_timesteps).
    Please, note that this deviates from Heat's convention that data sets are handeled as 2-D arrays with the feature axis being the second axis.

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
    -----
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
            U, S, V = ht.linalg.svd(
                X[:, :-1] if X.split == 0 else X[:, :-1].balance(), full_matrices=False
            )
            if self.svd_tol is not None:
                # truncation w.r.t. prescribed bound on explained variance
                # determine svd_rank accordingly
                total_variance = (S**2).sum()
                variance_threshold = (1 - self.svd_tol) * total_variance.larray.item()
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
                    X[:, :-1] if X.split == 0 else X[:, :-1].balance(),
                    self.svd_tol,
                    compute_sv=True,
                    safetyshift=5,
                )
            else:
                # hierarchical SVD with prescribed, fixed rank
                U, S, V, _ = ht.linalg.hsvd_rank(
                    X[:, :-1] if X.split == 0 else X[:, :-1].balance(),
                    self.svd_rank,
                    compute_sv=True,
                    safetyshift=5,
                )
            self.rom_basis_ = U
            self.n_modes_ = U.shape[1]
        else:
            # compute SVD via "randomized" SVD
            U, S, V = ht.linalg.rsvd(
                X[:, :-1] if X.split == 0 else X[:, :-1].balance_(),
                self.svd_rank,
            )
            self.rom_basis_ = U
            self.n_modes_ = U.shape[1]
        # second step of DMD: compute the reduced order model transfer matrix
        # we need to assume that the the transfer matrix of the ROM is small enough to fit into memory of one process
        if X.split == 0 or X.split is None:
            # if split axis of the input data is 0, using X[:,1:] does not result in un-balancedness and corresponding problems in matmul
            self.rom_transfer_matrix_ = self.rom_basis_.T @ X[:, 1:] @ V / S
        else:
            # if input is split along columns, X[:,1:] will be un-balanced and cause problems in matmul
            Xplus = X[:, 1:]
            Xplus.balance_()
            self.rom_transfer_matrix_ = self.rom_basis_.T @ Xplus @ V / S

        self.rom_transfer_matrix_.resplit_(None)
        # third step of DMD: compute the reduced order model eigenvalues and eigenmodes
        eigvals_loc, eigvec_loc = torch.linalg.eig(self.rom_transfer_matrix_.larray)
        self.rom_eigenvalues_ = ht.array(eigvals_loc, split=None, device=X.device)
        self.rom_eigenmodes_ = ht.array(eigvec_loc, split=None, device=X.device)
        self.dmdmodes_ = self.rom_basis_ @ self.rom_eigenmodes_

    def predict_next(self, X: ht.DNDarray, n_steps: int = 1) -> ht.DNDarray:
        """
        Predicts and returns the state(s) after n_steps-many time steps for given a current state(s).

        Parameters
        ----------
        X : DNDarray
            The current state(s) for the prediction. Must have the same number of features as the training data, but can be batched for multiple current states,
            i.e., X can be of shape (n_features,) or (n_features, n_current_states).
            The output will have the same shape as the input.
        n_steps : int, optional
            The number of steps to predict into the future. Default is 1, i.e., the next time step is predicted.
        """
        if not isinstance(n_steps, int):
            raise TypeError(f"Invalid type '{type(n_steps)}' for 'n_steps'. Must be an integer.")
        if self.rom_basis_ is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        # sanitize input data
        ht.sanitize_in(X)
        # if X is a 1-D DNDarray, we add an artificial batch dimension
        if X.ndim == 1:
            X = X.expand_dims(1)
        # check if the input data has the right number of features
        if X.shape[0] != self.rom_basis_.shape[0]:
            raise ValueError(
                f"Invalid number of features '{X.shape[0]}' in input data 'X'. Must have the same number of features as the training data."
            )
        rom_mat = self.rom_transfer_matrix_.copy()
        rom_mat.larray = torch.linalg.matrix_power(rom_mat.larray, n_steps)
        # the following line looks that complicated because we have to make sure that splits of the resulting matrices in
        # each of the products are split along the axis that deserves being splitted
        nextX = (self.rom_basis_.T @ X).T.resplit_(None) @ (self.rom_basis_ @ rom_mat).T
        return (nextX.T).squeeze()

    def predict(self, X: ht.DNDarray, steps: Union[int, List[int]]) -> ht.DNDarray:
        """
        Predics and returns future states given a current state(s) and returns them all as an array of size (n_steps, n_features).

        This function avoids a time-stepping loop (i.e., repeated calls to 'predict_next') and computes the future states in one go.
        To do so, the number of future times to predict must be of moderate size as an array of shape (n_steps, self.n_modes_, self.n_modes_) must fit into memory.
        Moreover, it must be ensured that:

        - the array of initial states is not split or split along the batch axis (axis 1) and the feature axis is small (i.e., self.rom_basis_ is not split)

        Parameters
        ----------
        X : DNDarray
            The current state(s) for the prediction. Must have the same number of features as the training data, but can be batched for multiple current states,
            i.e., X can be of shape (n_features,) or (n_current_states, n_features).
        steps : int or List[int]
            if int: predictions at time step 0, 1, ..., steps-1 are computed
            if List[int]: predictions at time steps given in the list are computed
        """
        if self.rom_basis_ is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        # sanitize input data
        ht.sanitize_in(X)
        # if X is a 1-D DNDarray, we add an artificial batch dimension
        if X.ndim == 1:
            X = X.expand_dims(1)
        # check if the input data has the right number of features
        if X.shape[0] != self.rom_basis_.shape[0]:
            raise ValueError(
                f"Invalid number of features '{X.shape[0]}' in input data 'X'. Must have the same number of features as the training data."
            )
        if isinstance(steps, int):
            steps = torch.arange(steps, dtype=torch.int32, device=X.device.torch_device)
        elif isinstance(steps, list):
            steps = torch.tensor(steps, dtype=torch.int32, device=X.device.torch_device)
        else:
            raise TypeError(
                f"Invalid type '{type(steps)}' for 'steps'. Must be an integer or a list of integers."
            )
        steps = steps.reshape(-1, 1).repeat(1, self.rom_eigenvalues_.shape[0])
        X_rom = self.rom_basis_.T @ X

        transfer_mat = _torch_matrix_diag(torch.pow(self.rom_eigenvalues_.larray, steps))
        transfer_mat = (
            self.rom_eigenmodes_.larray @ transfer_mat @ self.rom_eigenmodes_.larray.inverse()
        )
        transfer_mat = torch.real(
            transfer_mat
        )  # necessary to avoid imaginary parts due to numerical errors

        if self.rom_basis_.split is None and (X.split is None or X.split == 1):
            result = (
                transfer_mat @ X_rom.larray
            )  # here we assume that X_rom is not split or split along the second axis (axis 1)
            del transfer_mat

            result = (
                self.rom_basis_.larray @ result
            )  # here we assume that self.rom_basis_ is not split (i.e., the feature number is small)
            result = ht.array(result, is_split=2 if X.split == 1 else None)
            return result.squeeze().T
        else:
            raise NotImplementedError(
                "Predicting multiple time steps in one go is not supported for the given data layout. Please, use 'predict_next' instead, or open an issue on GitHub if you require this feature."
            )

    def __str__(self):
        if ht.MPI_WORLD.rank == 0:
            if self.rom_basis_ is not None:
                return (
                    f"-------------------- DMD (Dynamic Mode Decomposition) --------------------\n"
                    f"Number of modes: {self.n_modes_}\n"
                    f"State space dimension: {self.rom_basis_.shape[0]}\n"
                    f"DMD eigenvalues: {self.rom_eigenvalues_.larray}\n"
                    f"--------------------------------------------------------------------------\n"
                    f"ROM basis of shape {self.rom_basis_.shape}:\n"
                    f"\t split axis: {self.rom_basis_.split}\n"
                    f"\t device: {self.rom_basis_.device.__str__().split(':')[-2]}\n"
                    f"--------------------------------------------------------------------------\n"
                )
            else:
                return (
                    f"---------------- UNFITTED DMD (Dynamic Mode Decomposition) ---------------\n"
                    f"Parameters for fit are as follows: \n"
                    f"\t SVD-solver: {self.svd_solver}\n"
                    f"\t SVD-rank: {self.svd_rank}\n"
                    f"\t SVD-tolerance: {self.svd_tol}\n"
                    f"--------------------------------------------------------------------------\n"
                )
        else:
            return ""


class DMDc(ht.RegressionMixin, ht.BaseEstimator):
    """
    Dynamic Mode Decomposition with Control (DMDc), plain vanilla version with SVD-based implementation.

    The time series of states and controls must be provided as 2-D DNDarrays of shapes (n_state_features, n_timesteps) and (n_control_features, n_timesteps), respectively.
    Please, note that this deviates from Heat's convention that data sets are handeled as 2-D arrays with the feature axis being the second axis.

    Parameters
    ----------
    svd_solver : str, optional
        Specifies the algorithm to use for the singular value decomposition (SVD). Options are 'full' (default), 'hierarchical', and 'randomized'.
    svd_rank : int, optional
        The rank to which SVD of the states shall be truncated. For `'full'` SVD, `svd_rank = None` together with `svd_tol = None` (default) will result in no truncation.
        For `svd_solver='full'`, at most one of `svd_rank` or `svd_tol` may be specified.
        For `svd_solver='hierarchical'`, either `svd_rank` (rank to truncate to) or `svd_tol` (tolerance to truncate to) must be specified.
        For `svd_solver='randomized'`, `svd_rank` must be specified and determines the the rank to truncate to.
    svd_tol : float, optional
        The tolerance to which SVD of the states shall be truncated. For `'full'` SVD, `svd_tol = None` together with `svd_rank = None` (default) will result in no truncation.
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
    rom_control_matrix_ : DNDarray
        The reduced order model control matrix.
    rom_eigenvalues_ : DNDarray
        The reduced order model eigenvalues.
    rom_eigenmodes_ : DNDarray
        The reduced order model eigenmodes ("DMD modes")

    Notes
    -----
    We follow the approach described in [1], Sects. 3.3 and 3.4.
    In the case that svd_rank is prescribed, the rank of the SVD of the full system matrix is set to svd_rank + n_control_features; cf. https://github.com/dynamicslab/pykoopman
    for the same approach.

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
        self.rom_control_matrix_ = None
        self.rom_eigenvalues_ = None
        self.rom_eigenmodes_ = None
        self.dmdmodes_ = None
        self.n_modes_ = None
        self.n_modes_system_ = None

    def fit(self, X: ht.DNDarray, C: ht.DNDarray) -> Self:
        """
        Fits the DMD model to the given data.

        Parameters
        ----------
        X : DNDarray
            The time series data of states to fit the DMD model to. Must be of shape (n_state_features, n_timesteps).
        C : DNDarray
            The time series of control inputs to fit the DMD model to. Must be of shape (n_control_features, n_timesteps).
        """
        ht.sanitize_in(X)
        ht.sanitize_in(C)
        # check if the input data is a 2-D DNDarray
        if X.ndim != 2:
            raise ValueError(
                f"Invalid shape '{X.shape}' for input data 'X'. Must be a 2-D DNDarray of shape (n_state_features, n_timesteps)."
            )
        if C.ndim != 2:
            raise ValueError(
                f"Invalid shape '{C.shape}' for input data 'C'. Must be a 2-D DNDarray of shape (n_control_features, n_timesteps)."
            )
        # check if the input data has at least two time steps
        if X.shape[1] < 2:
            raise ValueError(
                f"Invalid number of time steps '{X.shape[1]}' in input data 'X'. Must have at least two time steps."
            )
        if C.shape[1] < 2:
            raise ValueError(
                f"Invalid number of time steps '{C.shape[1]}' in input data 'C'. Must have at least two time steps."
            )
        # check if the input data has the same number of time steps
        if X.shape[1] != C.shape[1]:
            raise ValueError(
                f"Invalid number of time steps {X.shape[1]} in input data 'X' and {C.shape[1]} in input data 'C'. Must have the same number of time steps."
            )
        if X.split is not None and C.split is not None and X.split != C.split:
            raise ValueError(
                f"If both input data 'X' and 'C' are distributed, they must be distributed along the same axis, but X.split={X.split}, C.split={C.split}."
            )
        Xplus = X[:, 1:]
        Xplus.balance_()
        Omega = ht.concatenate((X, C), axis=0)[:, :-1]
        # first step of DMDc: compute the SVD of the input data from first to second last time step
        # as well as of the full system matrix
        if self.svd_solver == "full" or not X.is_distributed():
            U, S, V = ht.linalg.svd(Xplus, full_matrices=False)
            Utilde, Stilde, Vtilde = ht.linalg.svd(Omega, full_matrices=False)
            if self.svd_tol is not None:
                # truncation w.r.t. prescribed bound on explained variance
                # determine svd_rank accordingly
                total_variance = (S**2).sum()
                variance_threshold = (1 - self.svd_tol) * total_variance.larray.item()
                variance_cumsum = (S**2).larray.cumsum(0)
                self.n_modes_ = len(variance_cumsum[variance_cumsum <= variance_threshold]) + 1
                total_variance_system = (Stilde**2).sum()
                variance_threshold_system = (1 - self.svd_tol) * total_variance_system.larray.item()
                variance_cumsum_system = (Stilde**2).larray.cumsum(0)
                self.n_modes_system_ = (
                    len(variance_cumsum_system[variance_cumsum_system <= variance_threshold_system])
                    + 1
                )
            elif self.svd_rank is not None:
                # truncation w.r.t. prescribed rank
                self.n_modes_ = self.svd_rank
                self.n_modes_system_ = self.svd_rank + C.shape[0]
            else:
                # no truncation
                self.n_modes_ = S.shape[0]
                self.n_modes_system = Stilde.shape[0]

            self.rom_basis_ = U[:, : self.n_modes_]
            V = V[:, : self.n_modes_]
            S = S[: self.n_modes_]
            Vtilde = Vtilde[:, : self.n_modes_system_]
            Stilde = Stilde[: self.n_modes_system_]
            Utilde1 = Utilde[: X.shape[0], : self.n_modes_system_]
            Utilde2 = Utilde[X.shape[0] :, : self.n_modes_system_]
        # compute SVD via "hierarchical" SVD
        elif self.svd_solver == "hierarchical":
            if self.svd_tol is not None:
                # hierarchical SVD with prescribed upper bound on relative error
                U, S, V, _ = ht.linalg.hsvd_rtol(
                    Xplus,
                    self.svd_tol,
                    compute_sv=True,
                    safetyshift=5,
                )
                Utilde, Stilde, Vtilde, _ = ht.linalg.hsvd_rtol(
                    Omega,
                    self.svd_tol,
                    compute_sv=True,
                    safetyshift=5,
                )
            else:
                # hierarchical SVD with prescribed, fixed rank
                U, S, V, _ = ht.linalg.hsvd_rank(
                    Xplus,
                    self.svd_rank,
                    compute_sv=True,
                    safetyshift=5,
                )
                Utilde, Stilde, Vtilde, _ = ht.linalg.hsvd_rank(
                    Omega,
                    self.svd_rank + C.shape[0],
                    compute_sv=True,
                    safetyshift=5,
                )
            self.rom_basis_ = U
            self.n_modes_ = U.shape[1]
            self.n_modes_system_ = Utilde.shape[1]
            Utilde1 = Utilde[: X.shape[0], :]
            Utilde2 = Utilde[X.shape[0] :, :]
        else:
            # compute SVD via "randomized" SVD
            U, S, V = ht.linalg.rsvd(
                Xplus,
                self.svd_rank,
            )
            Utilde, Stilde, Vtilde = ht.linalg.rsvd(
                Omega,
                self.svd_rank + C.shape[0],
            )
            self.rom_basis_ = U
            self.n_modes_ = U.shape[1]
            self.n_modes_system_ = Utilde.shape[1]
            Utilde1 = Utilde[: X.shape[0], :]
            Utilde2 = Utilde[X.shape[0] :, :]

        # ensure that everything is balanced for the following steps
        Utilde2.balance_()
        Utilde1.balance_()
        Vtilde.balance_()
        if Utilde2.split is not None and Utilde2.shape[Utilde2.split] < Utilde2.comm.size:
            Utilde2.resplit_((Utilde2.split + 1) % 2)
        if Utilde1.split is not None and Utilde1.shape[Utilde1.split] < Utilde1.comm.size:
            Utilde1.resplit_((Utilde1.split + 1) % 2)
        if Vtilde.split is not None and Vtilde.shape[Vtilde.split] < Vtilde.comm.size:
            Vtilde.resplit_((Vtilde.split + 1) % 2)
        # second step of DMD: compute the reduced order model transfer matrix
        # we need to assume that the the transfer matrix of the ROM is small enough to fit into memory of one process
        self.rom_transfer_matrix_ = (
            self.rom_basis_.T
            @ Xplus
            @ (Vtilde / Stilde)
            @ (Utilde1.T @ self.rom_basis_).resplit_(None)
        )
        self.rom_control_matrix_ = (self.rom_basis_.T @ Xplus) @ (
            (Vtilde / Stilde) @ Utilde2.T
        ).resplit_(0)
        self.rom_transfer_matrix_.resplit_(None)
        self.rom_control_matrix_.resplit_(None)

        # third step of DMD: compute the reduced order model eigenvalues and eigenmodes
        eigvals_loc, eigvec_loc = torch.linalg.eig(self.rom_transfer_matrix_.larray)
        self.rom_eigenvalues_ = ht.array(eigvals_loc, split=None, device=X.device)
        self.rom_eigenmodes_ = ht.array(eigvec_loc, split=None, device=X.device)
        self.dmdmodes_ = (
            Xplus @ (Vtilde / Stilde) @ Utilde1.T @ self.rom_basis_ @ self.rom_eigenmodes_
        )

    def predict(self, X: ht.DNDarray, C: ht.DNDarray) -> ht.DNDarray:
        """
        Predicts and returns future states given the current state(s) ``X`` and control trajectory ``C``.

        Parameters
        ----------
        X : DNDarray
            The current state(s) for the prediction. Must have the same number of features as the training data, but can be batched for multiple current states,
            i.e., X can be of shape (n_state_features,) or (n_batch, n_state_features).
        C : DNDarray
            The control trajectory for the prediction. Must have the same number of control features as the training data, i.e., C must be of shape
            (n_control_features,) --for a single time step-- or (n_control_features, n_timesteps).
        """
        if self.rom_basis_ is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        # sanitize input data
        ht.sanitize_in(X)
        ht.sanitize_in(C)
        # if X is a 1-D DNDarray, we add an artificial batch dimension; check correct dimensions for X
        if X.ndim == 1:
            X = X.expand_dims(0)
        if X.ndim > 2:
            raise ValueError(
                f"Invalid shape '{X.shape}' for input data 'X'. Must be a 2-D DNDarray of shape (n_batch, n_state_features) or a 1-D DNDarray of shape (n_state_features,)."
            )
        # if C is a 1-D DNDarray, we add an artificial dimension for the single time step; check correct dimensions for C
        if C.ndim == 1:
            C = C.expand_dims(1)
        if C.ndim > 2:
            raise ValueError(
                f"Invalid shape '{C.shape}' for input data 'C'. Must be a 2-D DNDarray of shape (n_control_features, n_timesteps) or a 1-D DNDarray of shape (n_control_features,) for a single time step."
            )
        # check if the input data has the right number of features for control and state space
        if X.shape[1] != self.rom_basis_.shape[0]:
            raise ValueError(
                f"Invalid number of features '{X.shape[1]}' in input data 'X'. Must have the same number of features as the training data (={self.rom_basis_.shape[0]})."
            )
        if C.shape[0] != self.rom_control_matrix_.shape[1]:
            raise ValueError(
                f"Invalid number of features '{C.shape[1]}' in input data 'C'. Must have the same number of features as the training data (={self.rom_control_matrix_.shape[1]})."
            )
        # different cases
        if C.split is not None:
            raise ValueError("So far only C.split = None is supported .")
        # time evolution in the reduced order model
        X_red = X @ self.rom_basis_
        X_red_full = ht.zeros(
            (X.shape[0], self.rom_basis_.shape[1], C.shape[1]),
            split=X_red.split,
            device=X.device,
            dtype=X.dtype,
        )
        X_red_full[:, :, 0] = X_red
        for i in range(1, C.shape[1]):
            X_red_full[:, :, i] = (self.rom_transfer_matrix_ @ X_red_full[:, :, i - 1].T).T + (
                self.rom_control_matrix_ @ C[:, i - 1]
            ).T
        # reshape in order to be able to multiply with basis again
        X_red_full = X_red_full.reshape(self.rom_basis_.shape[1], -1).resplit_(
            1 if X_red_full.split == 0 else None
        )
        X_pred = self.rom_basis_ @ X_red_full
        # reshape again and return
        return X_pred.reshape(X.shape[0], X.shape[1], C.shape[1])

    def __str__(self):
        if ht.MPI_WORLD.rank == 0:
            if self.rom_basis_ is not None:
                return (
                    f"----------- DMDc (Dynamic Mode Decomposition with control) ---------------\n"
                    f"Number of modes: {self.n_modes_}\n"
                    f"State space dimension: {self.rom_basis_.shape[0]}\n"
                    f"Control space dimension: {self.rom_control_matrix_.shape[1]}\n"
                    f"DMD eigenvalues: {self.rom_eigenvalues_.larray}\n"
                    f"--------------------------------------------------------------------------\n"
                    f"ROM basis of shape {self.rom_basis_.shape}:\n"
                    f"\t split axis: {self.rom_basis_.split}\n"
                    f"\t device: {self.rom_basis_.device.__str__().split(':')[-2]}\n"
                    f"--------------------------------------------------------------------------\n"
                )
            else:
                return (
                    f"-------- UNFITTED DMDc (Dynamic Mode Decomposition with control) ---------\n"
                    f"Parameters for fit are as follows: \n"
                    f"\t SVD-solver: {self.svd_solver}\n"
                    f"\t SVD-rank: {self.svd_rank}\n"
                    f"\t SVD-tolerance: {self.svd_tol}\n"
                    f"--------------------------------------------------------------------------\n"
                )
        else:
            return ""
