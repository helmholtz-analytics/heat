"""
Implements Symmetric Eigenvalue Decomposition
"""

import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple

from ..communication import MPICommunication
from ..dndarray import DNDarray
from .. import factories
from .. import types
from ..linalg import matrix_norm, vector_norm, matmul, qr, solve_triangular
from .basics import _estimate_largest_singularvalue, condest
from ..indexing import where
from ..random import randn
from ..devices import Device
from ..manipulations import vstack, hstack, concatenate, diag, balance
from ..exponential import sqrt
from .. import statistics
from mpi4py import MPI

from scipy.special import ellipj
from scipy.special import ellipkm1

__all__ = []


def _subspaceiteration(
    A: DNDarray,
    C: DNDarray,
    silent: bool = True,
    safetyparam: int = 3,
    maxit: int = None,
    tol: float = None,
) -> DNDarray:
    """
    This auxiliary function implements the subspace iteration as required for symmetric eigenvalue decomposition
    via polar decomposition; cf. Ref. 2 below. The algorithm for subspace iteration itself is taken from Ref. 1,
    Algorithm 3.

    --- TODO explain what the function does TODO ---

    References
    ------------
    1. Nakatsukasa, Y., & Higham, N. J. (2013). Stable and efficient spectral divide and conquer algorithms for Hermitian eigenproblems. SIAM Journal on Scientific Computing, 35(3).
    2. Nakatsukasa, Y., & Freund, R. W. (2016). Computing fundamental matrix decompositions accurately via the matrix sign function in two iterations: The power of Zolotarev's functions. SIAM Review, 58(3).
    """
    # set parameters for convergence
    if A.dtype == types.float64:
        maxit = 3 if maxit is None else maxit
        tol = 1e-12 if tol is None else tol
    elif A.dtype == types.float32:
        maxit = 6 if maxit is None else maxit
        tol = 1e-6 if tol is None else tol
    else:
        raise TypeError(
            f"Input DNDarray must be of data type float32 or float64, but is of type {A.dtype}."
        )

    Anorm = matrix_norm(A, ord="fro")

    # this initialization is proposed in Ref. 1, Sect. 5.1
    k = int(np.round(matrix_norm(C, ord="fro").numpy() ** 2))
    columnnorms = vector_norm(C, axis=0)
    idx = where(
        columnnorms
        >= factories.ones(columnnorms.shape, comm=columnnorms.comm, split=columnnorms.split)
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
                    X.shape[0], X.shape[0] - X.shape[1], dtype=X.dtype, device=X.device, comm=X.comm
                ),
            ]
        )
        Q, _ = qr(X)
        Q_k = Q[:, :k].balance()
        Q_k_orth = Q[:, k:].balance()
        E = Q_k_orth.T @ A @ Q_k
        Enorm = matrix_norm(E, ord="fro")
        if Enorm / Anorm < tol:
            # exit if success
            if A.comm.rank == 0 and not silent:
                print(f"Number of subspace iterations: {it}")
            return Q, k
        # else go on with iteration
        X = C @ Q_k
        it += 1
    # warning if the iteration did not converge within the maximum number of iterations
    if A.comm.rank == 0 and not silent:
        print(
            f"Subspace iteration did not converge in {maxit} iterations. \n It holds ||E||_F/||A||_F = {Enorm/Anorm}, which might impair the accuracy of the result."
        )
    return Q, k


# def eigh(
#     A: DNDarray,
#     r: int = 8,
#     depth: int = 0,
#     group: int = 0,
#     orig_lsize: int = 0,
#     silent: bool = True,
# ) -> Tuple[DNDarray, DNDarray]:
#     """
#     Eigenvalue decomposition of symmetric matrices
#     following the approach based on Zolotarev-PD
#     """
#     # consistency checks
#     if not isinstance(A, DNDarray):
#         raise RuntimeError("Argument needs to be a DNDarray but is {}.".format(type(A)))
#     if A.shape[0] != A.shape[1]:
#         raise RuntimeError(
#             "Computation of Eigenvalues requires square matrix, but matrix is %d x %d"
#             % (A.shape[0], A.shape[1])
#         )
#     if not A.ndim == 2:
#         raise RuntimeError("A needs to be a 2D matrix")
#     if r > 8:
#         raise RuntimeError("It is required that r <= 8, but input was r=%d." % r)

#     n = A.shape[0]
#     nprocs = A.comm.Get_size()
#     global_comm = A.comm.handle
#     if orig_lsize == 0:
#         orig_lsize = min(A.lshape_map[:, A.split])

#     # handle the case where there is no split
#     if A.split is None:
#         lamda, v = torch.linalg.eigh(A.larray)
#         Lambda = factories.array(lamda, device=A.device, split=A.split, comm=A.comm)
#         V = factories.array(v, device=A.device, split=A.split, comm=A.comm)
#         if A.comm.rank == 0 and not silent:
#             print(
#                 "\t" * depth
#                 + +"At depth %d: solution of symmetric eigenvalue problem of size %dx%d in pytorch."
#                 % (depth, n, n)
#             )
#             print("\t" * depth + "             (no split dimension)")
#         return Lambda, V

#     # handle the case where there is a split dimension, but the number of processes is too small to apply ZoloPD efficiently
#     if nprocs < r or (r == 1 and nprocs == 1) or A.shape[0] <= orig_lsize:
#         lamda, v = torch.linalg.eigh(A.copy().resplit_(None).larray)
#         Lambda = factories.array(lamda, device=A.device, split=0, comm=A.comm)
#         V = factories.array(v, device=A.device, split=A.split, comm=A.comm)
#         if A.comm.rank == 0 and not silent:
#             print(
#                 "\t" * depth
#                 + "At depth %d: solution of symmetric eigenvalue problem of size %dx%d in pytorch."
#                 % (depth, n, n)
#             )
#             if nprocs < r:
#                 print(
#                     "\t" * depth
#                     + "             (only %d processes left, and ZoloPD requires %d.)" % (nprocs, r)
#                 )
#             else:
#                 print("\t" * depth + "             (problem small enough for direct solution)")
#         return Lambda, V

#     # now we have to handle the main case: there is a split dimension and the number of processes is high enough to apply ZoloPD

#     sigma = statistics.median(diag(A))

#     U = pd(
#         A
#         - sigma * factories.eye((n, n), dtype=A.dtype, device=A.device, comm=A.comm, split=A.split),
#         r,
#         False,
#     )

#     V, k = subspaceiteration(
#         A,
#         0.5
#         * (U + factories.eye((n, n), dtype=A.dtype, device=A.device, comm=A.comm, split=A.split)),
#         silent,
#     )

#     A = (V.T @ A @ V).resplit(A.split)

#     if A.comm.rank == 0 and not silent:
#         print(
#             "\t" * depth
#             + "At depth %d: ZoloPD(r=%d) on %d processes reduced sym.eig. problem of size %dx%d to"
#             % (depth, r, nprocs, n, n)
#         )
#         print(
#             "\t" * depth
#             + "            two independent problems of size %dx%d and %dx%d respectively."
#             % (k, k, n - k, n - k)
#         )

#     # Get A1 and A2 from the paper as matrices on the respective process groups...
#     # strategy (very first rough idea):
#     #   1. we have already formed A = V.T A V (see above)
#     #   2. This matrix is now sent to process groups 1 and 2
#     #   3. On the groups the respective diagonal block of A is extracted

#     nprocs1 = round(k / n * nprocs)
#     idx_all = [i for i in range(nprocs)]
#     idx1 = [i for i in range(nprocs1)]
#     idx2 = [i for i in range(nprocs1, nprocs)]
#     group1 = global_comm.group.Incl(idx1)
#     group2 = global_comm.group.Incl(idx2)
#     comm1 = global_comm.Create_group(group1)
#     comm2 = global_comm.Create_group(group2)
#     comm1_ht = MPICommunication(handle=comm1)
#     comm2_ht = MPICommunication(handle=comm2)

#     if global_comm.rank in idx1:
#         comm_ht = comm1_ht
#         group = 1
#     else:
#         comm_ht = comm2_ht
#         group = 2

#     A_local_shapes = A.lshape_map[:, A.split].numpy().tolist()

#     Anew = factories.empty(A.shape, dtype=A.dtype, split=A.split, comm=comm_ht)

#     Anew_local_shapes1 = (
#         global_comm.bcast(Anew.lshape_map[:, A.split], root=idx1[0]).numpy().tolist()
#     )
#     Anew_local_shapes2 = (
#         global_comm.bcast(Anew.lshape_map[:, A.split], root=idx2[0]).numpy().tolist()
#     )

#     # print(depth, global_comm.rank, A_local_shapes, Anew_local_shapes1, Anew_local_shapes2)
#     to_send_to_1, to_recv_at_1 = what_to_send_and_to_recv(
#         A_local_shapes, Anew_local_shapes1, idx_all, idx1
#     )
#     to_send_to_2, to_recv_at_2 = what_to_send_and_to_recv(
#         A_local_shapes, Anew_local_shapes2, idx_all, idx2
#     )
#     to_send = to_send_to_1[global_comm.rank] + to_send_to_2[global_comm.rank]
#     if global_comm.rank in idx1:
#         to_recv = to_recv_at_1[global_comm.rank]
#     else:
#         to_recv = to_recv_at_2[global_comm.rank - nprocs1]

#     if A.split == 0:
#         recv_bufs = [
#             torch.zeros(
#                 (entry[1], A.shape[1]), dtype=A.dtype.torch_type(), device=A.device.torch_device
#             )
#             for entry in to_recv
#         ]
#     else:
#         recv_bufs = [
#             torch.zeros(
#                 (A.shape[0], entry[1]), dtype=A.dtype.torch_type(), device=A.device.torch_device
#             )
#             for entry in to_recv
#         ]
#     reqs = [global_comm.Irecv(recv_bufs[k], to_recv[k][0], tag=1) for k in range(len(to_recv))]

#     if A.split == 0:
#         [
#             global_comm.Send(A.larray[entry[1][0] : entry[1][1], :].clone(), entry[0], tag=1)
#             for entry in to_send
#         ]
#     else:
#         [
#             global_comm.Send(A.larray[:, entry[1][0] : entry[1][1]].clone(), entry[0], tag=1)
#             for entry in to_send
#         ]

#     [req.wait() for req in reqs]

#     if A.split == 0:
#         Anew.larray = torch.vstack(recv_bufs)
#     elif A.split == 1:
#         Anew.larray = torch.hstack(recv_bufs)
#     del recv_bufs

#     if global_comm.rank in idx1:
#         Anew = balance(Anew[:k, :k])
#     else:
#         Anew = balance(Anew[k:, k:])

#     Lambdanew, Vnew = eigh(Anew, r, depth + 1, group, orig_lsize, silent)

#     if A.comm.rank == 0 and not silent:
#         print(
#             "\t" * depth
#             + "At depth %d: subproblems of sizes %d and %d have been solved on level %d."
#             % (depth, k, n - k, depth + 1)
#         )

#     # now we have to send back to all processes and to "merge"

#     Lambda1 = factories.empty(k, dtype=A.dtype, split=0, comm=A.comm)
#     Lambda2 = factories.empty(n - k, dtype=A.dtype, split=0, comm=A.comm)
#     V1 = factories.empty((k, k), dtype=A.dtype, split=A.split, comm=A.comm)
#     V2 = factories.empty((n - k, n - k), dtype=A.dtype, split=A.split, comm=A.comm)

#     if A.comm.rank in idx1:
#         target_loc_shapes_L = Lambda1.lshape_map[:, 0].numpy().tolist()
#         target_loc_shapes_V = V1.lshape_map[:, V1.split].numpy().tolist()
#         idx_curr = idx1
#     else:
#         target_loc_shapes_L = Lambda2.lshape_map[:, 0].numpy().tolist()
#         target_loc_shapes_V = V2.lshape_map[:, V2.split].numpy().tolist()
#         idx_curr = idx2

#     Lambdanew_local_shapes = Lambdanew.lshape_map[:, 0].numpy().tolist()
#     Vnew_local_shapes = Vnew.lshape_map[:, Vnew.split].numpy().tolist()

#     # print(depth, global_comm.rank, Vnew_local_shapes, idx_curr)
#     to_send_V, to_recv_V_prelim = what_to_send_and_to_recv(
#         Vnew_local_shapes, target_loc_shapes_V, idx_curr, idx_all
#     )
#     to_send_L, to_recv_L_prelim = what_to_send_and_to_recv(
#         Lambdanew_local_shapes, target_loc_shapes_L, idx_curr, idx_all
#     )
#     if A.comm.rank in idx2:
#         to_send_V = to_send_V[global_comm.rank - len(idx1)]
#         to_send_L = to_send_L[global_comm.rank - len(idx1)]
#     else:
#         to_send_V = to_send_V[global_comm.rank]
#         to_send_L = to_send_L[global_comm.rank]

#     to_recv_V1 = A.comm.bcast(to_recv_V_prelim, root=idx1[0])[global_comm.rank]
#     to_recv_V2 = A.comm.bcast(to_recv_V_prelim, root=idx2[0])[global_comm.rank]
#     to_recv_L1 = A.comm.bcast(to_recv_L_prelim, root=idx1[0])[global_comm.rank]
#     to_recv_L2 = A.comm.bcast(to_recv_L_prelim, root=idx2[0])[global_comm.rank]

#     send_reqs_L = [
#         global_comm.isend(Lambdanew.larray[entry[1][0] : entry[1][1]], entry[0], tag=1)
#         for entry in to_send_L
#     ]
#     if Vnew.split == 0:
#         send_reqs_V = [
#             global_comm.isend(Vnew.larray[entry[1][0] : entry[1][1], :], entry[0], tag=1)
#             for entry in to_send_V
#         ]
#     elif Vnew.split == 1:
#         send_reqs_V = [
#             global_comm.isend(Vnew.larray[:, entry[1][0] : entry[1][1]], entry[0], tag=1)
#             for entry in to_send_V
#         ]
#     else:
#         raise NotImplementedError("Not yet implemented!")

#     recv_arrays_L1 = [global_comm.recv(None, entry[0], tag=1) for entry in to_recv_L1]
#     recv_arrays_L2 = [global_comm.recv(None, entry[0], tag=1) for entry in to_recv_L2]
#     recv_arrays_V1 = [global_comm.recv(None, entry[0], tag=1) for entry in to_recv_V1]
#     recv_arrays_V2 = [global_comm.recv(None, entry[0], tag=1) for entry in to_recv_V2]

#     [req.wait() for req in send_reqs_L]
#     [req.wait() for req in send_reqs_V]

#     Lambda1.larray = torch.hstack(recv_arrays_L1)
#     Lambda2.larray = torch.hstack(recv_arrays_L2)

#     if A.split == 0:
#         V1.larray = torch.vstack(recv_arrays_V1)
#         V2.larray = torch.vstack(recv_arrays_V2)
#     elif A.split == 1:
#         V1.larray = torch.hstack(recv_arrays_V1)
#         V2.larray = torch.hstack(recv_arrays_V2)

#     del (recv_arrays_L1, recv_arrays_L2, recv_arrays_V1, recv_arrays_V2)

#     Lambda = hstack([Lambda2, Lambda1])
#     V = hstack([matmul(V[:, k:].balance(), V2), matmul(V[:, :k].balance(), V1)])

#     del (V1, V2, Lambda1, Lambda2)

#     if A.comm.rank == 0 and not silent:
#         print("\t" * depth + "At depth %d: merged solution of sizes %d and %d." % (depth, k, n - k))

#     return Lambda, V
