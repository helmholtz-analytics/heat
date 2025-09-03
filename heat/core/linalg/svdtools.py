"""
distributed hierarchical SVD
"""

import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple, Optional

from ..communication import MPICommunication
from ..dndarray import DNDarray
from .. import factories
from .. import types
from ..linalg import matmul, vector_norm, qr, svd
from ..indexing import where
from ..random import randn
from ..sanitation import sanitize_in_nd_realfloating
from ..manipulations import vstack, hstack, diag, balance

from .. import statistics
from math import log, ceil, floor, sqrt


__all__ = ["hsvd_rank", "hsvd_rtol", "hsvd", "rsvd", "isvd"]


#######################################################################################
# hierachical SVD "hSVD"
#######################################################################################


def hsvd_rank(
    A: DNDarray,
    maxrank: int,
    compute_sv: bool = False,
    maxmergedim: Optional[int] = None,
    safetyshift: int = 5,
    silent: bool = True,
) -> Union[
    Tuple[DNDarray, DNDarray, DNDarray, float], Tuple[DNDarray, DNDarray, DNDarray], DNDarray
]:
    """
    Hierarchical SVD (hSVD) with prescribed truncation rank `maxrank`.
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
    """
    sanitize_in_nd_realfloating(A, "A", [2])
    A_local_size = max(A.lshape_map[:, 1])

    if maxmergedim is not None and maxmergedim < 2 * (maxrank + safetyshift) + 1:
        raise RuntimeError(
            "maxmergedim=%d is too small. Please ensure `maxmergedim > 2*(maxrank + safetyshift)`, or set `maxmergedim=None` in order to work with the default value."
            % maxmergedim
        )

    if maxmergedim is None:
        if A_local_size >= 2 * (maxrank + safetyshift):
            maxmergedim = A_local_size
        else:
            maxmergedim = 2 * (maxrank + safetyshift) + 1

    return hsvd(
        A,
        maxrank=maxrank,
        maxmergedim=maxmergedim,
        rtol=None,
        safetyshift=safetyshift,
        no_of_merges=None,
        compute_sv=compute_sv,
        silent=silent,
        warnings_off=True,
    )


def hsvd_rtol(
    A: DNDarray,
    rtol: float,
    compute_sv: bool = False,
    maxrank: Optional[int] = None,
    maxmergedim: Optional[int] = None,
    safetyshift: int = 5,
    no_of_merges: Optional[int] = None,
    silent: bool = True,
) -> Union[
    Tuple[DNDarray, DNDarray, DNDarray, float], Tuple[DNDarray, DNDarray, DNDarray], DNDarray
]:
    """
    Hierchical SVD (hSVD) with prescribed upper bound on the relative reconstruction error.
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
    """
    sanitize_in_nd_realfloating(A, "A", [2])
    A_local_size = max(A.lshape_map[:, 1])

    if maxmergedim is not None and maxrank is None:
        maxrank = floor(A_local_size / 2) - safetyshift
        if maxrank <= 0:
            raise ValueError("safetyshift is too large.")

    if maxmergedim is None and maxrank is not None:
        if A_local_size >= 2 * (maxrank + safetyshift):
            maxmergedim = A_local_size
        else:
            maxmergedim = 2 * (maxrank + safetyshift) + 1

    if (
        maxmergedim is not None
        and maxrank is not None
        and maxmergedim < 2 * (maxrank + safetyshift) + 1
    ):
        raise ValueError(
            "maxmergedim=%d is too small. Please ensure `maxmergedim > 2*(maxrank + safetyshift)`, or set `maxmergedim=None` in order to work with the default value."
            % maxmergedim
        )

    if maxmergedim is None and maxrank is None:
        if no_of_merges is None:
            no_of_merges = 2
        maxmergedim = 2 * (A.shape[1] + safetyshift) + 1
        maxrank = A.shape[1]

    if no_of_merges is not None and no_of_merges < 2:
        raise ValueError("`no_of_merges` must be >= 2.")

    return hsvd(
        A,
        maxrank=maxrank,
        maxmergedim=maxmergedim,
        rtol=rtol,
        safetyshift=safetyshift,
        no_of_merges=no_of_merges,
        compute_sv=compute_sv,
        silent=silent,
        warnings_off=True,
    )


def hsvd(
    A: DNDarray,
    maxrank: Optional[int] = None,
    maxmergedim: Optional[int] = None,
    rtol: Optional[float] = None,
    safetyshift: int = 0,
    no_of_merges: Optional[int] = 2,
    compute_sv: bool = False,
    silent: bool = True,
    warnings_off: bool = False,
) -> Union[
    Tuple[DNDarray, DNDarray, DNDarray, float], Tuple[DNDarray, DNDarray, DNDarray], DNDarray
]:
    """
    Computes an approximate truncated SVD of A utilizing a distributed hiearchical algorithm; see the references.
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
    """
    # if split dimension is 0, transpose matrix and remember this
    transposeflag = False
    if A.split == 0:
        transposeflag = True
        A = A.T

    no_procs = A.comm.Get_size()

    Anorm = vector_norm(A)

    if rtol is not None:
        loc_atol = Anorm.larray * rtol / sqrt(2 * no_procs - 1)
    else:
        loc_atol = None

    # compute the SVDs on the 0th level
    # Important notice: in the following 'node' refers to the nodes of the tree-like merging structure of hSVD, not to the compute nodes of an HPC-cluster
    level = 0
    active_nodes = [i for i in range(no_procs)]
    if A.comm.rank == 0 and not silent:
        print(
            "hSVD level %d...\t" % level,
            "processes ",
            "\t\t".join(["%d" % an for an in active_nodes]),
        )

    U_loc, sigma_loc, err_squared_loc = _compute_local_truncated_svd(
        level, A.comm.rank, A.larray, maxrank, loc_atol, safetyshift
    )
    U_loc = torch.matmul(U_loc, torch.diag(sigma_loc))

    finished = False
    while not finished:
        # communicate dimension of each nodes to all other nodes
        dims_global = A.comm.allgather(U_loc.shape[1])

        if A.comm.rank == 0 and not silent:
            print(
                "              current ranks:",
                "\t\t".join(["%d" % dims_global[an] for an in active_nodes]),
            )

        # determine future nodes and prepare sending
        future_nodes = [0]
        send_to = [[]] * no_procs
        current_idx = 0
        current_future_node = 0
        used_budget = 0
        k = 0
        counter = 0
        #        print("active_nodes", active_nodes)
        while k < len(active_nodes):
            current_idx = active_nodes[k]
            if used_budget + dims_global[current_idx] > maxmergedim or counter == no_of_merges:
                current_future_node = current_idx
                future_nodes.append(current_future_node)
                used_budget = dims_global[current_idx]
                counter = 1
            else:
                if not used_budget == 0:
                    send_to[current_idx] = current_future_node
                used_budget += dims_global[current_idx]
                counter += 1
            k += 1

        recv_from = [[]] * no_procs
        for i in future_nodes:
            recv_from[i] = [k for k in range(no_procs) if send_to[k] == i]

        if A.comm.rank in future_nodes:
            # FUTURE NODES
            # in the future nodes receive local arrays from previous level
            err_squared_loc = [err_squared_loc] + [
                torch.zeros_like(err_squared_loc) for i in recv_from[A.comm.rank]
            ]
            U_loc = [U_loc] + [
                torch.zeros(
                    (A.shape[0] + 1, dims_global[i]),
                    dtype=A.larray.dtype,
                    device=A.device.torch_device,
                )
                for i in recv_from[A.comm.rank]
            ]
            for k in range(len(recv_from[A.comm.rank])):
                # receive concatenated U_loc and err_squared_loc
                A.comm.Recv(U_loc[k + 1], recv_from[A.comm.rank][k], tag=recv_from[A.comm.rank][k])
                # separate U_loc and err_squared_loc
                err_squared_loc[k + 1] = U_loc[k + 1][-1, 0]
                U_loc[k + 1] = U_loc[k + 1][:-1]

            # concatenate the received arrays
            U_loc = torch.hstack(U_loc)
            err_squared_loc = sum(err_squared_loc)
            level += 1
            if A.comm.rank == 0 and not silent:
                print(
                    "hSVD level %d...\t" % level,
                    "processes ",
                    "\t\t".join(["%d" % fn for fn in future_nodes]),
                )
            # compute "local" SVDs on the current level

            if len(future_nodes) == 1:
                safetyshift = 0
            U_loc, sigma_loc, err_squared_loc_new = _compute_local_truncated_svd(
                level, A.comm.rank, U_loc, maxrank, loc_atol, safetyshift
            )

            if len(future_nodes) > 1:
                # prepare next level or...
                U_loc = torch.matmul(U_loc, torch.diag(sigma_loc))
            err_squared_loc += err_squared_loc_new
        elif A.comm.rank in active_nodes and A.comm.rank not in future_nodes:
            # concatenate U_loc and err_squared_loc to avoid sending multiple messages
            err_squared_loc = torch.full((1, U_loc.shape[1]), err_squared_loc, device=U_loc.device)
            U_loc = torch.vstack([U_loc, err_squared_loc])
            A.comm.Send(U_loc, send_to[A.comm.rank], tag=A.comm.rank)
            # separate U_loc and err_squared_loc again
            err_squared_loc = U_loc[-1, 0]
            U_loc = U_loc[:-1]
        if len(future_nodes) == 1:
            finished = True
        else:
            active_nodes = future_nodes
    # After completion of the SVD, distribute the result from process 0 to all processes again
    # stack U_loc and err_squared_loc to avoid sending multiple messages
    err_squared_loc = torch.full((1, U_loc.shape[1]), err_squared_loc, device=U_loc.device)
    U_loc = torch.vstack([U_loc, err_squared_loc])
    U_loc_shape = A.comm.bcast(U_loc.shape, root=0)
    if A.comm.rank != 0:
        U_loc = torch.zeros(U_loc_shape, dtype=A.larray.dtype, device=A.device.torch_device)
    A.comm.Bcast(U_loc, root=0)
    # separate U_loc and err_squared_loc again
    err_squared_loc = U_loc[-1, 0]
    U = factories.array(U_loc[:-1], device=A.device, split=None, comm=A.comm)
    rel_error_estimate = (
        factories.array(err_squared_loc**0.5, device=A.device, split=None, comm=A.comm) / Anorm
    )

    # Postprocessing:
    # compute V if required or if split=0 for the input
    # in case of split=0 undo the transposition...
    if transposeflag or compute_sv:
        V = matmul(A.T, U)
        sigma = vector_norm(V, axis=0)
        if vector_norm(sigma) > 0:
            V = matmul(V, diag(1 / sigma))

        if transposeflag:
            if compute_sv:
                return V, sigma, U, rel_error_estimate
            return V, rel_error_estimate

        return U, sigma, V, rel_error_estimate

    return U, rel_error_estimate


def _compute_local_truncated_svd(
    level: int,
    proc_id: int,
    U_loc: torch.Tensor,
    maxrank: int,
    loc_atol: Optional[float],
    safetyshift: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Auxiliary routine for hsvd: computes the truncated SVD ("U-factor" and "sigma-factor" of the SVD, i.e. first and second output) of the respective local array `U_loc` together with an estimate for the truncation error (third output).
    Truncation of the SVD either to absolute (!) tolerance `loc_atol` or to maximal rank `maxrank` is performed; moreover, singular values close to or below the level of "numerical noise" (1e-14 for float64, 1e-7 for float32) are cut.
    A safetyshift is added, i.e. the final truncation rank determined from `loc_atol` and `maxrank` is increased by `safetyshift`.
    """
    U_loc, sigma_loc, _ = torch.linalg.svd(U_loc, full_matrices=False)

    if U_loc.dtype == torch.float64:
        noiselevel = 1e-14
    elif U_loc.dtype == torch.float32:
        noiselevel = 1e-7

    # the "intuitive" choice torch.argwhere is only available in torch>=1.11.0, so we need to use torch.nonzero that works similar
    no_noise_idx = torch.nonzero(sigma_loc >= noiselevel)

    if len(no_noise_idx) != 0:
        cut_noise_rank = max(no_noise_idx) + 1
        if loc_atol is None:
            loc_trunc_rank = min(maxrank, cut_noise_rank)
        else:
            # the "intuitive" choice torch.argwhere is only available in torch>=1.11.0, so we need to use torch.nonzero that works similar
            ideal_trunc_rank = min(
                torch.nonzero(
                    torch.tensor(
                        [torch.norm(sigma_loc[k:]) ** 2 for k in range(sigma_loc.shape[0] + 1)],
                        device=U_loc.device,
                    )
                    < loc_atol**2
                )
            )

            loc_trunc_rank = min(maxrank, ideal_trunc_rank, cut_noise_rank)
            if loc_trunc_rank != ideal_trunc_rank:
                print(
                    "in hSVD (level %d, process %d): abs tol = %2.2e requires truncation to rank %d, but maxrank=%d. Loss of desired precision (rtol) very likely!"
                    % (level, proc_id, loc_atol, ideal_trunc_rank, maxrank)
                )

        loc_trunc_rank = min(sigma_loc.shape[0], loc_trunc_rank + safetyshift)
        err_squared_loc = torch.linalg.norm(sigma_loc[loc_trunc_rank - safetyshift :]) ** 2
        return U_loc[:, :loc_trunc_rank], sigma_loc[:loc_trunc_rank], err_squared_loc
    else:
        err_squared_loc = torch.linalg.norm(sigma_loc) ** 2
        sigma_loc = torch.zeros(1, dtype=U_loc.dtype, device=U_loc.device)
        U_loc = torch.zeros(U_loc.shape[0], 1, dtype=U_loc.dtype, device=U_loc.device)
        return U_loc, sigma_loc, err_squared_loc


##############################################################################################
# Randomized SVD "rSVD"
##############################################################################################


def rsvd(
    A: DNDarray,
    rank: int,
    n_oversamples: int = 10,
    power_iter: int = 0,
    qr_procs_to_merge: int = 2,
) -> Union[Tuple[DNDarray, DNDarray, DNDarray], Tuple[DNDarray, DNDarray]]:
    r"""
    Randomized SVD (rSVD) with prescribed truncation rank `rank`.
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
    """
    sanitize_in_nd_realfloating(A, "A", [2])
    if not isinstance(rank, int):
        raise TypeError(f"rank must be an integer, but is {type(rank)}.")
    if rank < 1:
        raise ValueError(f"rank must be positive, but is {rank}.")
    if not isinstance(n_oversamples, int):
        raise TypeError(
            f"if provided, n_oversamples must be an integer, but is {type(n_oversamples)}."
        )
    if n_oversamples < 0:
        raise ValueError(f"n_oversamples must be non-negative, but is {n_oversamples}.")
    if not isinstance(power_iter, int):
        raise TypeError(f"if provided, power_iter must be an integer, but is {type(power_iter)}.")
    if power_iter < 0:
        raise ValueError(f"power_iter must be non-negative, but is {power_iter}.")

    ell = rank + n_oversamples
    q = power_iter

    # random matrix
    splitOmega = 1 if A.split == 0 else 0
    Omega = randn(A.shape[1], ell, dtype=A.dtype, device=A.device, split=splitOmega)

    # compute the range of A
    Y = matmul(A, Omega)
    Q, _ = qr(Y, procs_to_merge=qr_procs_to_merge)

    # power iterations
    for _ in range(q):
        if Q.split is not None and Q.shape[Q.split] < Q.comm.size:
            Q.resplit_(None)
        Y = matmul(A.T, Q)
        Q, _ = qr(Y, procs_to_merge=qr_procs_to_merge)
        if Q.split is not None and Q.shape[Q.split] < Q.comm.size:
            Q.resplit_(None)
        Y = matmul(A, Q)
        Q, _ = qr(Y, procs_to_merge=qr_procs_to_merge)

    # compute the SVD of the projected matrix
    if Q.split is not None and Q.shape[Q.split] < Q.comm.size:
        Q.resplit_(None)
    B = matmul(Q.T, A)
    B.resplit_(
        None
    )  # B will be of size ell x ell and thus small enough to fit into memory of a single process
    U, sigma, V = svd.svd(B)  # actually just torch svd as input is not split anymore
    U = matmul(Q, U)[:, :rank]
    U.balance_()
    S = sigma[:rank]
    V = V[:, :rank]
    V.balance_()
    return U, S, V


##############################################################################################
# Incremental SVD "iSVD"
##############################################################################################


def _isvd(
    new_data: DNDarray,
    U_old: DNDarray,
    S_old: DNDarray,
    V_old: Optional[DNDarray] = None,
    maxrank: Optional[int] = None,
    old_matrix_size: Optional[int] = None,
    old_rowwise_mean: Optional[DNDarray] = None,
) -> Union[Tuple[DNDarray, DNDarray, DNDarray], Tuple[DNDarray, DNDarray, DNDarray, DNDarray]]:
    """
    Helper function for iSVD and iPCA; follows roughly the "incremental PCA with mean update", Fig.1 in:
    David A. Ross, Jongwoo Lim, Ruei-Sung Lin, Ming-Hsuan Yang. Incremental Learning for Robust Visual Tracking. IJCV, 2008.

    Either incremental SVD / PCA or incremental SVD / PCA  with mean subtraction is performed.

    Parameters
    ----------
    new_data: DNDarray
        new data as DNDarray
    U_old, S_old, V_old: DNDarrays
        "old" SVD-factors
        if no V_old is provided, only U and S are computed (PCA)
    maxrank: int, optional
        rank to which new SVD should be truncated
    old_matrix_size: int, optional
        size of the old matrix; this does not need to be identical to V_old.shape[0] as "old" SVD might have been truncated
    old_rowwise_mean: int, optional
        row-wise mean of the old matrix; if not provided, no mean subtraction is performed
    """
    # old SVD is SVD of a matrix of dimension m x n as has rank r
    # new data have shape m x d
    d = new_data.shape[1]
    n = V_old.shape[0] if V_old is not None else old_matrix_size
    r = S_old.shape[0]
    if maxrank is None:
        maxrank = min(n + d, U_old.shape[0])
    else:
        maxrank = min(maxrank, min(n + d, U_old.shape[0]))

    if old_rowwise_mean is not None:
        new_data_rowwise_mean = statistics.mean(new_data, axis=1)
        new_rowwise_mean = (old_matrix_size * old_rowwise_mean + d * new_data_rowwise_mean) / (
            old_matrix_size + d
        )
        new_data -= new_data_rowwise_mean.reshape(-1, 1)
        new_data = hstack(
            [
                new_data,
                (new_data_rowwise_mean - old_rowwise_mean)
                * (d * old_matrix_size / (d + old_matrix_size)) ** 0.5,
            ]
        )
        d += 1

    # orthogonalize and decompose new_data
    UtC = U_old.T @ new_data
    if U_old.split is not None:
        new_data = new_data.resplit_(U_old.split) - U_old @ UtC
    else:
        new_data = new_data - (U_old @ UtC).resplit_(new_data.split)
    P, Rc = qr(new_data)

    # prepare one component of "new" V-factor
    if V_old is not None:
        V_new = vstack(
            [
                V_old,
                factories.zeros(
                    (d, r),
                    device=V_old.device,
                    dtype=V_old.dtype,
                    split=V_old.split,
                    comm=V_old.comm,
                ),
            ]
        )
        helper = vstack(
            [
                factories.zeros(
                    (n, d),
                    device=V_old.device,
                    dtype=V_old.dtype,
                    split=V_old.split,
                    comm=V_old.comm,
                ),
                factories.eye(
                    d, device=V_old.device, dtype=V_old.dtype, split=V_old.split, comm=V_old.comm
                ),
            ]
        )
        V_new = hstack([V_new, helper])
        del helper

    # prepare one component of "new" U-factor
    U_new = hstack([U_old, P])

    # prepare "inner" matrix that needs to be decomposed, decompose it
    helper1 = vstack(
        [
            diag(S_old),
            factories.zeros(
                (Rc.shape[0] + UtC.shape[0] - r, r),
                device=S_old.device,
                dtype=S_old.dtype,
                split=S_old.split,
                comm=S_old.comm,
            ),
        ]
    )
    if r > d:
        Rc = Rc.resplit_(UtC.split)
    else:
        UtC = UtC.resplit_(Rc.split)
    helper2 = vstack([UtC, Rc])
    innermat = hstack([helper1, helper2])
    del (helper1, helper2)
    # as innermat is small enough to fit into memory of a single process, we can use torch svd
    u, s, v = svd.svd(innermat.resplit_(None))
    del innermat

    # truncate if desired
    if maxrank < s.shape[0]:
        u = u[:, :maxrank]
        s = s[:maxrank]
        v = v[:, :maxrank]

    U_new = U_new @ u
    if V_old is not None:
        V_new = V_new @ v

    if V_old is not None:  # use-case: SVD
        return U_new, s, V_new
    if old_rowwise_mean is not None:  # use-case PCA
        return U_new, s, new_rowwise_mean


def isvd(
    new_data: DNDarray,
    U_old: DNDarray,
    S_old: DNDarray,
    V_old: DNDarray,
    maxrank: Optional[int] = None,
) -> Tuple[DNDarray, DNDarray, DNDarray]:
    r"""Incremental SVD (iSVD) for the addition of new data to an existing SVD.
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
    """
    # check if new_data, U_old, V_old are 2D DNDarrays and float32/64
    sanitize_in_nd_realfloating(new_data, "new_data", [2])
    sanitize_in_nd_realfloating(U_old, "U_old", [2])
    sanitize_in_nd_realfloating(S_old, "S_old", [1])
    sanitize_in_nd_realfloating(V_old, "V_old", [2])
    # check if number of columns of U_old and V_old match the number of elements in S_old
    if U_old.shape[1] != S_old.shape[0]:
        raise ValueError(
            "The number of columns of U_old must match the number of elements in S_old."
        )
    if V_old.shape[1] != S_old.shape[0]:
        raise ValueError(
            "The number of columns of V_old must match the number of elements in S_old."
        )
    # check if the number of columns of new_data matches the number of rows of U_old and V_old
    if new_data.shape[0] != U_old.shape[0]:
        raise ValueError("The number of rows of new_data must match the number of rows of U_old.")

    return _isvd(new_data, U_old, S_old, V_old, maxrank)
