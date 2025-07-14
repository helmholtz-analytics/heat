"""
Implements Symmetric Eigenvalue Decomposition
"""

import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple

from ..dndarray import DNDarray
from .. import factories
from .. import types
from ..linalg import matrix_norm, vector_norm, matmul, qr, polar
from ..indexing import where
from ..random import randn
from ..devices import Device
from ..manipulations import vstack, hstack, concatenate, diag, balance
from .. import statistics
from mpi4py import MPI
from ..sanitation import sanitize_in_nd_realfloating


__all__ = ["eigh"]


def _subspaceiteration(
    A: DNDarray,
    C: DNDarray,
    silent: bool = True,
    safetyparam: int = 3,
    maxit: int = None,
    tol: float = None,
    depth: int = 0,
) -> DNDarray:
    """
    Auxiliary function that implements the subspace iteration as required for symmetric eigenvalue decomposition
    via polar decomposition; cf. Ref. 2 below. The algorithm for subspace iteration itself is taken from Ref. 1,
    Algorithm 3 in Sect. 5.1.

    Given a symmetric matrix ``A`` and and a matrix ``C`` that is the orthogonal projection onto an invariant
    subspace of A, this function computes and returns an orthogonal matrix ``Q`` such that Q = [V_1 V_2] with
    C = V_1 V_1.T. Moreover, the dimension of the invariant subspace, i.e., the number of column of V_1 is
    returned as well.

    References
    ----------
    1.  Nakatsukasa, Y., & Higham, N. J. (2013). Stable and efficient spectral divide and conquer algorithms for
        Hermitian eigenproblems. SIAM Journal on Scientific Computing, 35(3).
    2.  Nakatsukasa, Y., & Freund, R. W. (2016). Computing fundamental matrix decompositions accurately via the
        matrix sign function in two iterations: The power of Zolotarev's functions. SIAM Review, 58(3).
    """
    # set parameters for convergence
    if A.dtype == types.float64:
        maxit = 3 if maxit is None else maxit
        tol = 1e-8 if tol is None else tol
    elif A.dtype == types.float32:
        maxit = 6 if maxit is None else maxit
        tol = 1e-4 if tol is None else tol
    else:
        raise TypeError(
            f"Input DNDarray must be of data type float32 or float64, but is of type {A.dtype}."
        )

    Anorm = matrix_norm(A, ord="fro")

    # this initialization is proposed in Ref. 1, Sect. 5.1
    k = int(round(matrix_norm(C, ord="fro").item() ** 2))
    columnnorms = vector_norm(C, axis=0)
    idx = where(
        columnnorms
        >= factories.ones(
            columnnorms.shape,
            comm=columnnorms.comm,
            split=columnnorms.split,
            device=columnnorms.device,
        )
        * statistics.percentile(columnnorms, 100.0 * (1 - (k + safetyparam) / columnnorms.shape[0]))
    )
    X = C[:, idx].balance()

    # actual subspace iteration
    it = 1
    while it < maxit + 1:
        # enrich X by additional random columns to get a full orthonormal basis by QR
        X = hstack(
            [
                X,
                randn(
                    X.shape[0],
                    X.shape[0] - X.shape[1],
                    dtype=X.dtype,
                    device=X.device,
                    comm=X.comm,
                    split=X.split,
                ),
            ]
        )
        Q, _ = qr(X)
        Q_k = Q[:, :k].balance()
        Q_k_orth = Q[:, k:].balance()
        E = (Q_k_orth.T @ A) @ Q_k
        Enorm = matrix_norm(E, ord="fro")
        if Enorm / Anorm < tol:
            # exit if success
            if A.comm.rank == 0 and not silent:
                print("\t" * depth + f"            Number of subspace iterations: {it}")
            return Q, k
        # else go on with iteration
        X = C @ Q_k
        it += 1
    # warning if the iteration did not converge within the maximum number of iterations
    if A.comm.rank == 0 and not silent:
        print(
            "\t" * depth
            + f"            Subspace iteration did not converge in {maxit} iterations. \n"
            + "\t" * depth
            + f"            It holds ||E||_F/||A||_F = {Enorm / Anorm}, which might impair the accuracy of the result."  # noqa E226
        )
    return Q, k


def _eigh(
    A: DNDarray,
    r: int = None,
    silent: bool = True,
    r_max: int = 8,
    depth: int = 0,
    orig_lsize: int = 0,
) -> Tuple[DNDarray, DNDarray]:
    """
    Auxiliary function for eigh containing the main algorithmic content.
    Inputs are as for the public `eigh`-function, except for:
        `depth`:  an internal variable that is used to track the recursion depth,
        `orig_lsize` an internal variable that is used to propagate the local shapes of the original input matrix
            through the recursions in order to determine when the direct solution of the reduced problems is possible),
        `r`: a hyperparameter for the computation of the polar decomposition via :func:`heat.linalg.polar` which is
            applied multiple times in this function. See the documentation of :func:`heat.linalg.polar` for more details.
            In the actual implementation, this parameter is set to `None` for simplicity.
    """
    n = A.shape[0]
    global_comm = A.comm
    nprocs = global_comm.Get_size()
    rank = global_comm.rank

    # direct solution in torch if the problem is small enough
    if n <= orig_lsize or not A.is_distributed():
        orig_split = A.split
        A.resplit_(None)
        Lambda_loc, Q_loc = torch.linalg.eigh(A.larray)
        Lambda = factories.array(torch.flip(Lambda_loc, (0,)), split=0, comm=A.comm)
        V = factories.array(torch.flip(Q_loc, (1,)), split=orig_split, comm=A.comm)
        A.resplit_(orig_split)
        return Lambda, V

    if orig_lsize == 0:
        orig_lsize = min(A.lshape_map[:, A.split])

    # now we handle the main case: Zolo-PD is used to reduce the problem to two independent problems
    sigma = statistics.median(diag(A))

    U = polar.polar(
        A
        - sigma * factories.eye((n, n), dtype=A.dtype, device=A.device, comm=A.comm, split=A.split),
        r,
        False,
    )

    V, k = _subspaceiteration(
        A,
        0.5
        * (U + factories.eye((n, n), dtype=A.dtype, device=A.device, comm=A.comm, split=A.split)),
        silent,
        depth,
    )
    A = V.T @ A @ V

    if A.comm.rank == 0 and not silent:
        print(
            "\t" * depth
            + f"At depth {depth}: Zolo-PD(r={'auto' if r is None else r}) on {nprocs} processes reduced symmetric eigenvalue problem of size {n} to"
        )
        print(
            "\t" * depth
            + f"            two independent problems of size {k} and {n - k} respectively."
        )

    # from the "global" A, two independent "local" A's are created
    # the number of processes per local array is roughly proportional to their size with the constraint that
    # each "local" A needs to get at least one process
    nprocs1 = max(1, min(nprocs - 1, round(k / n * nprocs)))
    nprocs2 = nprocs - nprocs1
    new_lshapes = torch.tensor(
        [k // nprocs1 + (i < k % nprocs1) for i in range(nprocs1)]
        + [(n - k) // nprocs2 + (i < (n - k) % nprocs2) for i in range(nprocs2)]
    )
    new_lshape_map = A.lshape_map
    new_lshape_map[:, A.split] = new_lshapes
    A.redistribute_(target_map=new_lshape_map)
    local_comm = A.comm.Split(color=rank < nprocs1, key=rank)
    if A.split == 1:
        A_local = factories.array(
            A.larray[:k, :] if rank < nprocs1 else A.larray[k:, :],
            comm=local_comm,
            is_split=A.split,
        )
    else:
        A_local = factories.array(
            A.larray[:, :k] if rank < nprocs1 else A.larray[:, k:],
            comm=local_comm,
            is_split=A.split,
        )

    Lambda_local, V_local = _eigh(A_local, r, silent, r_max, depth + 1, orig_lsize)

    Lambda = factories.array(Lambda_local.larray, is_split=0, comm=A.comm)
    V_local_larray = V_local.larray
    if A.split == 0:
        if rank < nprocs1:
            V_local_larray = torch.hstack(
                [
                    V_local_larray,
                    torch.zeros(V_local_larray.shape[0], n - k, device=V_local.device.torch_device),
                ]
            )
        else:
            V_local_larray = torch.hstack(
                [
                    torch.zeros(V_local_larray.shape[0], k, device=V_local.device.torch_device),
                    V_local_larray,
                ]
            )
    else:
        if rank < nprocs1:
            V_local_larray = torch.vstack(
                [
                    V_local_larray,
                    torch.zeros(n - k, V_local_larray.shape[1], device=V_local.device.torch_device),
                ]
            )
        else:
            V_local_larray = torch.vstack(
                [
                    torch.zeros(k, V_local_larray.shape[1], device=V_local.device.torch_device),
                    V_local_larray,
                ]
            )
    V_new = factories.array(V_local_larray, is_split=A.split, comm=A.comm, device=A.device)
    V.balance_()
    V_new.balance_()
    V = V @ V_new

    if A.comm.rank == 0 and not silent:
        print(
            "\t" * depth
            + f"At depth {depth}: solutions of two independent problems of size {k} and {n - k} have been merged successfully."
        )

    return Lambda, V


def eigh(
    A: DNDarray,
    r_max_zolopd: int = 8,
    silent: bool = True,
) -> Tuple[DNDarray, DNDarray]:
    """
    Computes the symmetric eigenvalue decomposition of a symmetric n x n - matrix A, provided as a DNDarray.

    The function returns DNDarrays Lambda (shape (n,) with split = 0) and V (shape (n,n)) such that
    A = V @ diag(Lambda) @ V^T, where Lambda contains the eigenvalues of A and V is an orthonormal matrix
    containing the corresponding eigenvectors as columns.

    Parameters
    ----------
    A : DNDarray
        The input matrix. Must be symmetric.
    r_max_zolopd : int, optional
        This is a hyperparameter for the computation of the polar decomposition via :func:`heat.linalg.polar` which is
        applied multiple times in this function. See the documentation of :func:`heat.linalg.polar` for more details on its
        meaning and the respective default value.
    silent : bool, optional
        If True (default), suppresses output messages; otherwise, some information on the recursion is printed to the console.

    Notes
    -----
    Unlike the :func:`torch.linalg.eigh` function, the eigenvalues are returned in descending order.
    Note that no check of symmetry is performed on the input matrix A; thus, applying this function to a non-symmetric matrix may
    result in unpredictable behaviour without a specific error message pointing to this issue.

    The algorithm used for the computation of the symmetric eigenvalue decomposition is based on the Zolotarev polar decomposition;
    see Algorithm 5.2 in:

        Nakatsukasa, Y., & Freund, R. W. (2016). Computing fundamental matrix decompositions accurately via the
        matrix sign function in two iterations: The power of Zolotarev's functions. SIAM Review, 58(3).

    See Also
    --------
    :func:`heat.linalg.polar`
    """
    sanitize_in_nd_realfloating(A, "A", [2])
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Input matrix must be symmetric and, consequently, square, but input shape was {A.shape[0]} x {A.shape[1]}."
        )
    if not isinstance(r_max_zolopd, int) or r_max_zolopd < 1 or r_max_zolopd > 8:
        raise ValueError(
            f"If provided, parameter r_max_zolopd must be a positive integer, but was {r_max_zolopd} of type {type(r_max_zolopd)}."
        )
    return _eigh(A, None, silent, r_max_zolopd, 0, 0)
