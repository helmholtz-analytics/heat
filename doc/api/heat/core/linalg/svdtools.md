Module heat.core.linalg.svdtools
================================
distributed hierarchical SVD

Functions
---------

`hsvd(A: heat.core.dndarray.DNDarray, maxrank: int | None = None, maxmergedim: int | None = None, rtol: float | None = None, safetyshift: int = 0, no_of_merges: int | None = 2, compute_sv: bool = False, silent: bool = True, warnings_off: bool = False) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, float] | Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray] | heat.core.dndarray.DNDarray`
:   Computes an approximate truncated SVD of A utilizing a distributed hiearchical algorithm; see the references.
    The present function `hsvd` is a low-level routine, provides many options/parameters, but no default values, and is not recommended for usage by non-experts since conflicts
    arising from inappropriate parameter choice will not be catched. We strongly recommend to use the corresponding high-level functions `hsvd_rank` and `hsvd_rtol` instead.

    Input
    -------
    A: DNDarray
        2D-array (float32/64) of which hSVD has to be computed
    maxrank: int, optional
        truncation rank of the SVD
    maxmergedim: int, optional
        maximal size of the concatenation matrices when "merging" the local SVDs
    rtol: float, optional
        upper bound on the relative reconstruction error ||A-U Sigma V^T ||_F / ||A||_F (may deteriorate due to other parameters)
    safetyshift: int, optional
        shift that increases the actual truncation rank of the local SVDs during the computations in order to increase accuracy
    no_of_merges: int, optional
        maximum number of local SVDs to be "merged" at one step
    compute_sv: bool, optional
        determines whether to compute U, Sigma, V (compute_sv=True) or not (then U only)
    silent: bool, optional
        determines whether to print infos on the computations performed (silent=False)
    warnings_off: bool, optional
        switch on and off warnings that are not intended for the high-level routines based on this function

    Returns
    -------
    (Union[    Tuple[DNDarray, DNDarray, DNDarray, float], Tuple[DNDarray, DNDarray, DNDarray], DNDarray])
        if compute_sv=True: U, Sigma, V, a-posteriori error estimate for the reconstruction error ||A-U Sigma V^T ||_F / ||A||_F (computed according to [2] along the "true" merging tree used in the computations).
        if compute_sv=False: U, a-posteriori error estimate

    References
    ----------
    [1] Iwen, Ong. A distributed and incremental SVD algorithm for agglomerative data analysis on large networks. SIAM J. Matrix Anal. Appl., 37(4), 2016.
    [2] Himpe, Leibner, Rave. Hierarchical approximate proper orthogonal decomposition. SIAM J. Sci. Comput., 40 (5), 2018.

    See Also
    --------
    :func:`hsvd_rank`
    :func:`hsvd_rtol`

`hsvd_rank(A: heat.core.dndarray.DNDarray, maxrank: int, compute_sv: bool = False, maxmergedim: int | None = None, safetyshift: int = 5, silent: bool = True) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, float] | Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray] | heat.core.dndarray.DNDarray`
:   Hierarchical SVD (hSVD) with prescribed truncation rank `maxrank`.
    If A = U diag(sigma) V^T is the true SVD of A, this routine computes an approximation for U[:,:maxrank] (and sigma[:maxrank], V[:,:maxrank]).

    The accuracy of this approximation depends on the structure of A ("low-rank" is best) and appropriate choice of parameters.

    One can expect a similar outcome from this routine as for sci-kit learn's TruncatedSVD (with `algorithm='randomized'`) although a different, determinstic algorithm is applied here. Hereby, the parameters `n_components`
    and `n_oversamples` (sci-kit learn) roughly correspond to `maxrank` and `safetyshift` (see below).

    Parameters
    ----------
    A : DNDarray
        2D-array (float32/64) of which the hSVD has to be computed.
    maxrank : int
        truncation rank. (This parameter corresponds to `n_components` in sci-kit learn's TruncatedSVD.)
    compute_sv : bool, optional
        compute_sv=True implies that also Sigma and V are computed and returned. The default is False.
    maxmergedim : int, optional
        maximal size of the concatenation matrices during the merging procedure. The default is None and results in an appropriate choice depending on the size of the local slices of A and maxrank.
        Too small choices for this parameter will result in failure if the maximal size of the concatenation matrices does not allow to merge at least two matrices. Too large choices for this parameter can cause memory errors if the resulting merging problem becomes too large.
    safetyshift : int, optional
        Increases the actual truncation rank within the computations by a safety shift. The default is 5. (There is some similarity to `n_oversamples` in sci-kit learn's TruncatedSVD.)
    silent : bool, optional
        silent=False implies that some information on the computations are printed. The default is True.

    Returns
    -------
    (Union[    Tuple[DNDarray, DNDarray, DNDarray, float], Tuple[DNDarray, DNDarray, DNDarray], DNDarray])
        if compute_sv=True: U, Sigma, V, a-posteriori error estimate for the reconstruction error ||A-U Sigma V^T ||_F / ||A||_F (computed according to [2] along the "true" merging tree).
        if compute_sv=False: U, a-posteriori error estimate

    Notes
    -----
    The size of the process local SVDs to be computed during merging is proportional to the non-split size of the input A and (maxrank + safetyshift). Therefore, conservative choice of maxrank and safetyshift is advised to avoid memory issues.
    Note that, as sci-kit learn's randomized SVD, this routine is different from `numpy.linalg.svd` because not all singular values and vectors are computed
    and even those computed may be inaccurate if the input matrix exhibts a unfavorable structure.

    See Also
    --------
    :func:`hsvd`
    :func:`hsvd_rtol`

    References
    ----------
    [1] Iwen, Ong. A distributed and incremental SVD algorithm for agglomerative data analysis on large networks. SIAM J. Matrix Anal. Appl., 37(4), 2016.
    [2] Himpe, Leibner, Rave. Hierarchical approximate proper orthogonal decomposition. SIAM J. Sci. Comput., 40 (5), 2018.

`hsvd_rtol(A: heat.core.dndarray.DNDarray, rtol: float, compute_sv: bool = False, maxrank: int | None = None, maxmergedim: int | None = None, safetyshift: int = 5, no_of_merges: int | None = None, silent: bool = True) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, float] | Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray] | heat.core.dndarray.DNDarray`
:   Hierchical SVD (hSVD) with prescribed upper bound on the relative reconstruction error.
    If A = U diag(sigma) V^T is the true SVD of A, this routine computes an approximation for U[:,:r] (and sigma[:r], V[:,:r])
    such that the rel. reconstruction error ||A-U[:,:r] diag(sigma[:r]) V[:,:r]^T ||_F / ||A||_F does not exceed rtol.

    The accuracy of this approximation depends on the structure of A ("low-rank" is best) and appropriate choice of parameters. This routine is similar to `hsvd_rank` with the difference that
    truncation is not performed after a fixed number (namly `maxrank` many) singular values but after such a number of singular values that suffice to capture a prescribed fraction of the amount of information
    contained in the input data (`rtol`).

    Parameters
    ----------
    A : DNDarray
        2D-array (float32/64) of which the hSVD has to be computed.
    rtol : float
        desired upper bound on the relative reconstruction error ||A-U Sigma V^T ||_F / ||A||_F. This upper bound is processed into 'local'
        tolerances during the actual computations assuming the worst case scenario of a binary "merging tree"; therefore, the a-posteriori
        error for the relative error using the true "merging tree" (see output) may be significantly smaller than rtol.
        Prescription of maxrank or maxmergedim (disabled in default) can result in loss of desired precision, but can help to avoid memory issues.
    compute_sv : bool, optional
        compute_sv=True implies that also Sigma and V are computed and returned. The default is False.
    no_of_merges : int, optional
        Maximum number of processes to be merged at each step. If no further arguments are provided (see below),
        this completely determines the "merging tree" and may cause memory issues. The default is None and results in a binary merging tree.
        Note that no_of_merges dominates maxrank and maxmergedim in the sense that at most no_of_merges processes are merged
        even if maxrank and maxmergedim would allow merging more processes.
    maxrank : int, optional
        maximal truncation rank. The default is None.
        Setting at least one of maxrank and maxmergedim is recommended to avoid memory issues, but can result in loss of desired precision.
        Setting only maxrank (and not maxmergedim) results in an appropriate default choice for maxmergedim depending on the size of the local slices of A and the value of maxrank.
    maxmergedim : int, optional
        maximal size of the concatenation matrices during the merging procedure. The default is None and results in an appropriate choice depending on the size of the local slices of A and maxrank. The default is None.
        Too small choices for this parameter will result in failure if the maximal size of the concatenation matrices does not allow to merge at least two matrices. Too large choices for this parameter can cause memory errors if the resulting merging problem becomes too large.
        Setting at least one of maxrank and maxmergedim is recommended to avoid memory issues, but can result in loss of desired precision.
        Setting only maxmergedim (and not maxrank) results in an appropriate default choice for maxrank.
    safetyshift : int, optional
        Increases the actual truncation rank within the computations by a safety shift. The default is 5.
    silent : bool, optional
        silent=False implies that some information on the computations are printed. The default is True.

    Returns
    -------
    (Union[    Tuple[DNDarray, DNDarray, DNDarray, float], Tuple[DNDarray, DNDarray, DNDarray], DNDarray])
        if compute_sv=True: U, Sigma, V, a-posteriori error estimate for the reconstruction error ||A-U Sigma V^T ||_F / ||A||_F (computed according to [2] along the "true" merging tree used in the computations).
        if compute_sv=False: U, a-posteriori error estimate

    Notes
    -----
        The maximum size of the process local SVDs to be computed during merging is proportional to the non-split size of the input A and (maxrank + safetyshift). Therefore, conservative choice of maxrank and safetyshift is advised to avoid memory issues.
        For similar reasons, prescribing only rtol and the number of processes to be merged in each step (without specifying maxrank or maxmergedim) may result in memory issues.
        Although prescribing maxrank is therefore strongly recommended to avoid memory issues, but may result in loss of desired precision (rtol). If this occures, a separate warning will be raised.

        Note that this routine is different from `numpy.linalg.svd` because not all singular values and vectors are computed and even those computed may be inaccurate if the input matrix exhibts a unfavorable structure.

        To avoid confusion, note that `rtol` in this routine does not have any similarity to `tol` in scikit learn's TruncatedSVD.

    See Also
    --------
    :func:`hsvd`
    :func:`hsvd_rank`

    References
    ----------
        [1] Iwen, Ong. A distributed and incremental SVD algorithm for agglomerative data analysis on large networks. SIAM J. Matrix Anal. Appl., 37(4), 2016.
        [2] Himpe, Leibner, Rave. Hierarchical approximate proper orthogonal decomposition. SIAM J. Sci. Comput., 40 (5), 2018.

`isvd(new_data: heat.core.dndarray.DNDarray, U_old: heat.core.dndarray.DNDarray, S_old: heat.core.dndarray.DNDarray, V_old: heat.core.dndarray.DNDarray, maxrank: int | None = None) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Incremental SVD (iSVD) for the addition of new data to an existing SVD.
    Given the the SVD of an "old" matrix, :math:`X_\textnormal{old} = `U_\textnormal{old} \cdot S_\textnormal{old} \cdot V_\textnormal{old}^T`, and additional columns :math:`N` (\"`new_data`\"), this routine computes
    (a possibly approximate) SVD of the extended matrix :math:`X_\textnormal{new} = [ X_\textnormal{old} | N]`.

    Parameters
    ----------
    new_data : DNDarray
        2D-array (float32/64) of columns that are added to the "old" SVD. It must hold `new_data.split != 1` if `U_old.split = 0`.
    U_old : DNDarray
        U-factor of the SVD of the "old" matrix, 2D-array (float32/64). It must hold `U_old.split != 0` if `new_data.split = 1`.
    S_old : DNDarray
        Sigma-factor of the SVD of the "old" matrix, 1D-array (float32/64)
    V_old : DNDarray
        V-factor of the SVD of the "old" matrix, 2D-array (float32/64)
    maxrank : int, optional
        truncation rank of the SVD of the extended matrix. The default is None, i.e., no bound on the maximal rank is imposed.

    Notes
    -----
    Inexactness may arise due to truncation to maximal rank `maxrank` if rank of the data to be processed exceeds this rank.
    If you set `maxrank` to a high number (or None) in order to avoid inexactness, you may encounter memory issues.
    The implementation follows the approach described in Ref. [1], Sect. 2.

    References
    ----------
    [1] Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear algebra and its applications, 415(1), 20-30.

`rsvd(A: heat.core.dndarray.DNDarray, rank: int, n_oversamples: int = 10, power_iter: int = 0, qr_procs_to_merge: int = 2) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray] | Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Randomized SVD (rSVD) with prescribed truncation rank `rank`.
    If :math:`A = U \operatorname{diag}(S) V^T` is the true SVD of A, this routine computes an approximation for U[:,:rank] (and S[:rank], V[:,:rank]).

    The accuracy of this approximation depends on the structure of A ("low-rank" is best) and appropriate choice of parameters.

    Parameters
    ----------
    A : DNDarray
        2D-array (float32/64) of which the rSVD has to be computed.
    rank : int
        truncation rank. (This parameter corresponds to `n_components` in scikit-learn's TruncatedSVD.)
    n_oversamples : int, optional
        number of oversamples. The default is 10.
    power_iter : int, optional
        number of power iterations. The default is 0.
        Choosing `power_iter > 0` can improve the accuracy of the SVD approximation in the case of slowly decaying singular values, but increases the computational cost.
    qr_procs_to_merge : int, optional
        number of processes to merge at each step of QR decomposition in the power iteration (if power_iter > 0). The default is 2. See the corresponding remarks for :func:`heat.linalg.qr() <heat.core.linalg.qr.qr()>` for more details.


    Notes
    -----
    Memory requirements: the SVD computation of a matrix of size (rank + n_oversamples) x (rank + n_oversamples) must fit into the memory of a single process.
    The implementation follows Algorithm 4.4 (randomized range finder) and Algorithm 5.1 (direct SVD) in [1].

    References
    ----------
    [1] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.
