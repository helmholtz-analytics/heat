Module heat.decomposition.dmd
=============================
Module implementing the Dynamic Mode Decomposition (DMD) algorithm.

Classes
-------

`DMD(svd_solver: str | None = 'full', svd_rank: int | None = None, svd_tol: float | None = None)`
:   Dynamic Mode Decomposition (DMD), plain vanilla version with SVD-based implementation.

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

    ### Ancestors (in MRO)

    * heat.core.base.RegressionMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray) ‑> Self`
    :   Fits the DMD model to the given data.

        Parameters
        ----------
        X : DNDarray
            The time series data to fit the DMD model to. Must be of shape (n_features, n_timesteps).

    `predict(self, X: heat.core.dndarray.DNDarray, steps: int | List[int]) ‑> heat.core.dndarray.DNDarray`
    :   Predics and returns future states given a current state(s) and returns them all as an array of size (n_steps, n_features).

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

    `predict_next(self, X: heat.core.dndarray.DNDarray, n_steps: int = 1) ‑> heat.core.dndarray.DNDarray`
    :   Predicts and returns the state(s) after n_steps-many time steps for given a current state(s).

        Parameters
        ----------
        X : DNDarray
            The current state(s) for the prediction. Must have the same number of features as the training data, but can be batched for multiple current states,
            i.e., X can be of shape (n_features,) or (n_features, n_current_states).
            The output will have the same shape as the input.
        n_steps : int, optional
            The number of steps to predict into the future. Default is 1, i.e., the next time step is predicted.

`DMDc(svd_solver: str | None = 'full', svd_rank: int | None = None, svd_tol: float | None = None)`
:   Dynamic Mode Decomposition with Control (DMDc), plain vanilla version with SVD-based implementation.

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

    ### Ancestors (in MRO)

    * heat.core.base.RegressionMixin
    * heat.core.base.BaseEstimator

    ### Methods

    `fit(self, X: heat.core.dndarray.DNDarray, C: heat.core.dndarray.DNDarray) ‑> Self`
    :   Fits the DMD model to the given data.

        Parameters
        ----------
        X : DNDarray
            The time series data of states to fit the DMD model to. Must be of shape (n_state_features, n_timesteps).
        C : DNDarray
            The time series of control inputs to fit the DMD model to. Must be of shape (n_control_features, n_timesteps).

    `predict(self, X: heat.core.dndarray.DNDarray, C: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
    :   Predicts and returns future states given the current state(s) ``X`` and control trajectory ``C``.

        Parameters
        ----------
        X : DNDarray
            The current state(s) for the prediction. Must have the same number of features as the training data, but can be batched for multiple current states,
            i.e., X can be of shape (n_state_features,) or (n_batch, n_state_features).
        C : DNDarray
            The control trajectory for the prediction. Must have the same number of control features as the training data, i.e., C must be of shape
            (n_control_features,) --for a single time step-- or (n_control_features, n_timesteps).
