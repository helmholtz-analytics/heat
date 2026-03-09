"""
Implements polar decomposition (PD)
"""

import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple

from ..communication import MPICommunication, MPI
from ..dndarray import DNDarray
from .. import factories
from .. import types
from . import matrix_norm, vector_norm, matmul, qr, solve_triangular
from .basics import _estimate_largest_singularvalue, condest
from ..indexing import where
from ..random import randn
from ..devices import Device
from ..manipulations import vstack, hstack, concatenate, diag, balance
from ..exponential import sqrt
from .. import statistics

from scipy.special import ellipj
from scipy.special import ellipkm1

__all__ = ["polar"]


def _zolopd_n_iterations(r: int, kappa: float) -> int:
    """
    Returns the number of iterations required in the Zolotarev-PD algorithm.
    See the Table 3.1 in: Nakatsukasa, Y., & Freund, R. W. (2016). Computing Fundamental Matrix Decompositions Accurately via the Matrix Sign Function in Two Iterations: The Power of Zolotarev's Functions. SIAM Review, 58(3), DOI: https://doi.org/10.1137/140990334

    Inputs are `r` and `kappa` (named as in the paper), and the output is the number of iterations.
    """
    if kappa <= 1e2:
        its = [4, 3, 2, 2, 2, 2, 2, 2]
    elif kappa <= 1e3:
        its = [3, 3, 2, 2, 2, 2, 2, 2]
    elif kappa <= 1e5:
        its = [5, 3, 3, 3, 2, 2, 2, 2]
    elif kappa <= 1e7:
        its = [5, 4, 3, 3, 3, 2, 2, 2]
    else:
        its = [6, 4, 3, 3, 3, 3, 3, 2]
    return its[r - 1]


def _compute_zolotarev_coefficients(
    r: int, ell: float, device: str, dtype: types.datatype = types.float64
) -> Tuple[DNDarray, DNDarray, DNDarray]:
    """
    Computes c=(c_i)_i defined in equation (3.4), as well as a=(a_j)_j and Mhat defined in formulas (4.2)/(4.3) of the paper Nakatsukasa, Y., & Freund, R. W. (2016). Computing the polar decomposition with applications. SIAM Review, 58(3), DOI: https://doi.org/10.1137/140990334.
    Evaluations of the respective complete elliptic integral of the first kind and the Jacobi elliptic functions are imported from SciPy.

    Inputs are `r` and `ell` (named as in the paper), as well as the Heat data type `dtype` of the output (required for reasons of consistency).
    Output is a tupe containing the vectors `a` and `c` as DNDarrays and `Mhat`.
    """
    uu = np.arange(1, 2 * r + 1) * ellipkm1(ell**2) / (2 * r + 1)
    ellipfcts = np.asarray(ellipj(uu, 1 - ell**2)[:2])
    cc = ell**2 * ellipfcts[0, :] ** 2 / ellipfcts[1, :] ** 2
    aa = np.zeros(r)
    Mhat = 1
    for j in range(1, r + 1):
        p1 = 1
        p2 = 1
        for k in range(1, r + 1):
            p1 *= cc[2 * j - 2] - cc[2 * k - 1]
            if k != j:
                p2 *= cc[2 * j - 2] - cc[2 * k - 2]
        aa[j - 1] = -p1 / p2
        Mhat *= (1 + cc[2 * j - 2]) / (1 + cc[2 * j - 1])
    return (
        factories.array(cc, dtype=dtype, split=None, device=device),
        factories.array(aa, dtype=dtype, split=None, device=device),
        factories.array(Mhat, dtype=dtype, split=None, device=device),
    )


def _in_place_qr_with_q_only(A: DNDarray, procs_to_merge: int = 2) -> None:
    r"""
    Input A and procs_to_merge are as in heat.linalg.qr; difference it that this routine modified A in place and replaces it with Q.
    """
    if not A.is_distributed() or A.split < A.ndim - 2:
        # handle the case of a single process or split=None: just PyTorch QR
        # difference to heat.linalg.qr: we only return Q and put it directly in place of A
        A.larray, R = torch.linalg.qr(A.larray, mode="reduced")
        del R

    elif A.split == A.ndim - 1:
        # handle the case that A is split along the columns
        # unlike in heat.linalg.qr, we know by assumption of Zolo-PD that A has at least as many rows as columns

        nprocs = A.comm.size
        with torch.no_grad():
            for i in range(nprocs):
                # this loop goes through all the column-blocks (i.e. local arrays) of the matrix
                # this corresponds to the loop over all columns in classical Gram-Schmidt
                A_lshapes = A.lshape_map
                if i < nprocs - 1:
                    if A.comm.rank > i:
                        Q_buf = torch.zeros(
                            tuple(A_lshapes[i, :]),
                            dtype=A.larray.dtype,
                            device=A.device.torch_device,
                        )
                    color = 0 if A.comm.rank < i else 1
                    sub_comm = A.comm.Split(color, A.comm.rank)

                if A.comm.rank == i:
                    # orthogonalize the current block of columns by utilizing PyTorch QR
                    Q, R = torch.linalg.qr(A.larray, mode="reduced")
                    del R
                    A.larray[...] = Q
                    del Q
                    if i < nprocs - 1:
                        Q_buf = A.larray

                if i < nprocs - 1 and A.comm.rank >= i:
                    sub_comm.Bcast(Q_buf, root=0)

                if A.comm.rank > i:
                    # subtract the contribution of the current block of columns from the remaining columns
                    R_loc = torch.transpose(Q_buf, -2, -1) @ A.larray
                    A.larray -= Q_buf @ R_loc
                    del R_loc, Q_buf

    else:
        A, r = qr(A)
        del r


def polar(
    A: DNDarray,
    r: int = None,
    calcH: bool = True,
    condition_estimate: float = 1.0e16,
    silent: bool = True,
    r_max: int = 8,
) -> Tuple[DNDarray, DNDarray]:
    """
    Computes the so-called polar decomposition of the input 2D DNDarray ``A``, i.e., it returns the orthogonal matrix ``U`` and the symmetric, positive definite
    matrix ``H`` such that ``A = U @ H``.

    Input
    -----
    A : ht.DNDarray,
        The input matrix for which the polar decomposition is computed;
        must be two-dimensional, of data type float32 or float64, and must have at least as many rows as columns.
    r : int, optional, default: None
        The parameter r used in the Zolotarev-PD algorithm; if provided, must be an integer between 1 and 8 that divides the number of MPI processes.
        Higher values of r lead to faster convergence, but memory consumption is proportional to r.
        If not provided, the largest 1 <= r <= r_max that divides the number of MPI processes is chosen.
    calcH : bool, optional, default: True
        If True, the function returns the symmetric, positive definite matrix H. If False, only the orthogonal matrix U is returned.
    condition_estimate : float, optional, default: 1.e16.
        This argument allows to provide an estimate for the condition number of the input matrix ``A``, if such estimate is already known.
        If a positive number greater than 1., this value is used as an estimate for the condition number of A.
        If smaller or equal than 1., the condition number is estimated internally.
        The default value of 1.e16 is the worst case scenario considered in [1].
    silent : bool, optional, default: True
        If True, the function does not print any output. If False, some information is printed during the computation.
    r_max : int, optional, default: 8
        See the description of r for the meaning; r_max is only taken into account if r is not provided.


    Notes
    -----
    The implementation follows Algorithm 5.1 in Reference [1]; however, instead of switching from QR to Cholesky decomposition depending on the condition number,
    we stick to QR decomposition in all iterations.

    References
    ----------
    [1] Nakatsukasa, Y., & Freund, R. W. (2016). Computing Fundamental Matrix Decompositions Accurately via the Matrix Sign Function in Two Iterations: The Power of Zolotarev's Functions. SIAM Review, 58(3), DOI: https://doi.org/10.1137/140990334.
    """
    # check whether input is DNDarray of correct shape
    if not isinstance(A, DNDarray):
        raise TypeError(f"Input ``A`` needs to be a DNDarray but is {type(A)}.")
    if not A.ndim == 2:
        raise ValueError(f"Input ``A`` needs to be a 2D DNDarray, but its dimension is {A.ndim}.")
    if A.shape[0] < A.shape[1]:
        raise ValueError(
            f"Input ``A`` must have at least as many rows as columns, but has shape {A.shape}."
        )

    # check if A is a real floating point matrix and choose tolerances tol accordingly
    if A.dtype == types.float32:
        tol = 1.19e-7
    elif A.dtype == types.float64:
        tol = 2.22e-16
    else:
        raise TypeError(
            f"Input ``A`` must be of data type float32 or float64 but has data type {A.dtype}"
        )

    # check if input for r is reasonable
    if r is not None:
        if not isinstance(r, int) or r < 1 or r > 8:
            raise ValueError(
                f"If specified, input ``r`` must be an integer between 1 and 8, but is {r} of data type {type(r)}."
            )
        if A.is_distributed() and (A.comm.size % r != 0 or A.comm.size == r):
            raise ValueError(
                f"If specified, input ``r`` must be a non-trivial divisor of the number MPI processes , but r={r} and A.comm.size={A.comm.size}."
            )
    else:
        if not isinstance(r_max, int) or r_max < 1 or r_max > 8:
            raise ValueError(
                f"If specified, input ``r_max`` must be an integer between 1 and 8, but is {r_max} of data type {type(r_max)}."
            )
        for i in range(r_max, 0, -1):
            if A.comm.size % i == 0 and A.comm.size // i > 1:
                r = i
                break
        if not silent:
            if A.comm.rank == 0:
                print(f"Automatically chosen r={r} (r_max = {r_max}, {A.comm.size} processes).")

    # check if input for condition_estimate is reasonable
    if not isinstance(condition_estimate, float):
        raise TypeError(
            f"If specified, input ``condition_estimate`` must be a float but is {type(condition_estimate)}."
        )

    # early out for the non-distributed case
    if not A.is_distributed():
        U, s, vh = torch.linalg.svd(A.larray, full_matrices=False)
        U @= vh
        H = vh.T @ torch.diag(s) @ vh
        if calcH:
            return factories.array(U, is_split=None, comm=A.comm), factories.array(
                H, is_split=None, comm=A.comm
            )
        else:
            return factories.array(U, is_split=None, comm=A.comm)

    alpha = _estimate_largest_singularvalue(A).item()

    if condition_estimate <= 1.0:
        kappa = condest(A).item()
    else:
        kappa = condition_estimate

    if A.comm.rank == 0 and not silent:
        print(
            f"Condition number estimate: {kappa:2.2e} / Estimate for largest singular value: {alpha:2.2e}."
        )

    # each of these communicators has size r, along these communicators we parallelize the r many QR decompositions that are performed in parallel
    horizontal_comm = A.comm.Split(A.comm.rank // r, A.comm.rank)

    # each of these communicators has size MPI_WORLD.size / r and will carray a full copy of X for QR decomposition
    vertical_comm = A.comm.Split(A.comm.rank % r, A.comm.rank)

    # in each horizontal communicator, collect the local array of X from all processes
    local_shapes = horizontal_comm.allgather(A.lshape[A.split])
    new_local_shape = (
        (sum(local_shapes), A.shape[1]) if A.split == 0 else (A.shape[0], sum(local_shapes))
    )
    counts = tuple(local_shapes)
    displacements = tuple(np.cumsum([0] + list(local_shapes))[:-1])
    X_collected_local = torch.zeros(
        new_local_shape, dtype=A.dtype.torch_type(), device=A.device.torch_device
    )
    horizontal_comm.Allgatherv(
        A.larray, (X_collected_local, counts, displacements), recv_axis=A.split
    )

    X = factories.array(X_collected_local, is_split=A.split, comm=vertical_comm)
    X.balance_()
    X /= alpha

    # iteration counter and maximum number of iterations
    it = 0
    itmax = _zolopd_n_iterations(r, kappa)

    # parameters and coefficients, see Ref. [1] for their meaning
    ell = 1.0 / kappa
    c, a, Mhat = _compute_zolotarev_coefficients(r, ell, A.device, dtype=A.dtype)

    itmax = _zolopd_n_iterations(r, kappa)
    while it < itmax:
        it += 1
        if not silent:
            if A.comm.rank == 0:
                print(f"Starting Zolotarev-PD iteration no. {it}...")
        # remember current X for later convergence check
        X_old = X.copy()
        cId = factories.eye(X.shape[1], dtype=X.dtype, comm=X.comm, split=X.split, device=X.device)
        cId *= c[2 * horizontal_comm.rank].item() ** 0.5
        X = concatenate([X, cId], axis=0)
        del cId
        if X.split == 0:
            Q, R = qr(X)
            del R
            Q1 = Q[: A.shape[0], :].balance()
            Q2 = Q[A.shape[0] :, :].transpose().balance()
            Q1Q2 = matmul(Q1, Q2)
            del Q1, Q2
            X = X[: A.shape[0], :].balance()
            X /= r
        else:
            _in_place_qr_with_q_only(X)
            Q1 = X[: A.shape[0], :].balance()
            Q2 = X[A.shape[0] :, :].transpose().balance()
            del X
            Q1Q2 = matmul(Q1, Q2)
            del Q1, Q2
            X = X_old / r
        X += a[horizontal_comm.rank].item() / c[2 * horizontal_comm.rank].item() ** 0.5 * Q1Q2
        del Q1Q2
        X *= Mhat.item()
        # finally, sum over the horizontal communicators
        horizontal_comm.Allreduce(MPI.IN_PLACE, X.larray, op=MPI.SUM)

        # check for convergence and break if tolerance is reached
        if it > 1 and matrix_norm(X - X_old, ord="fro") / matrix_norm(X, ord="fro") <= tol ** (
            1 / (2 * r + 1)
        ):
            if not silent:
                if A.comm.rank == 0:
                    print(f"Zolotarev-PD iteration converged after {it} iterations.")
            break
        elif it < itmax:
            # if another iteration is necessary, update coefficients and parameters for next iteration
            ellold = ell
            ell = 1
            for j in range(r):
                ell *= (ellold**2 + c[2 * j + 1].item()) / (ellold**2 + c[2 * j].item())
            ell *= Mhat.item() * ellold
            if ell >= 1.0:
                ell = 1.0 - tol
            c, a, Mhat = _compute_zolotarev_coefficients(r, ell, A.device, dtype=A.dtype)
        else:
            if not silent:
                if A.comm.rank == 0:
                    print(
                        f"Zolotarev-PD iteration did not reach the convergence criterion after {itmax} iterations, which is most likely due to limited numerical accuracy and/or poor estimation of the condition number. The result may still be useful, but should be handled with care!"
                    )

    # as every process has much more data than required, we need to split the result into the parts that are actually
    counts = [
        X.lshape[X.split] // horizontal_comm.size + (r < X.lshape[X.split] % horizontal_comm.size)
        for r in range(horizontal_comm.size)
    ]
    displacements = [sum(counts[:r]) for r in range(horizontal_comm.size)]

    if A.split == 1:
        U_local = X.larray[
            :,
            displacements[horizontal_comm.rank] : displacements[horizontal_comm.rank]
            + counts[horizontal_comm.rank],
        ]
    else:
        U_local = X.larray[
            displacements[horizontal_comm.rank] : displacements[horizontal_comm.rank]
            + counts[horizontal_comm.rank],
            :,
        ]
    U = factories.array(U_local, is_split=A.split, comm=A.comm, device=A.device)
    del X
    U.balance_()

    # postprocessing: compute H if requested
    if calcH:
        H = matmul(U.T, A)
        H = 0.5 * (H + H.T.resplit(H.split))
        return U, H.resplit(A.split)
    else:
        return U
