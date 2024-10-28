"""
Implements polar decomposition (PD)
"""

import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple

from ..communication import MPICommunication
from ..dndarray import DNDarray
from .. import factories
from .. import types
from ..linalg import matrix_norm, vector_norm, matmul
from ..linalg.myqr import myqr, triu_solver
from ..indexing import where
from ..random import randn
from ..devices import Device
from ..manipulations import vstack, hstack, diag, balance
from ..exponential import sqrt
from .. import statistics
from mpi4py import MPI

from scipy.special import ellipj
from scipy.special import ellipkm1

__all__ = []


# def svd(A: DNDarray, r: int = 8, silent: bool = True) -> Tuple[DNDarray, DNDarray, DNDarray]:
#     """
#     Singular Value Decomposition via Zolotarev-PD/Eigh
#     """
#     if A.comm.Get_size() == 1 or A.split is None:
#         Uloc, Sloc, Vloc = torch.linalg.svd(A.larray, full_matrices=False)
#         U = factories.array(Uloc, dtype=A.dtype, split=A.split, device=A.device, comm=A.comm)
#         V = factories.array(Vloc, dtype=A.dtype, split=A.split, device=A.device, comm=A.comm)
#         if A.split is not None:
#             Ssplit = 0
#         else:
#             Ssplit = None
#         S = factories.array(Sloc, dtype=A.dtype, split=Ssplit, device=A.device, comm=A.comm)
#         return U, S, V, [0, 0]

#     t0 = MPI.Wtime()
#     U, H = pd(A, r=r, silent=silent)
#     t1 = MPI.Wtime()
#     Sigma, V = eigh(H, r=r, silent=silent)
#     t2 = MPI.Wtime()

#     return U @ V, Sigma, V, [t1 - t0, t2 - t1]


# def compute_zolotarev_coefficients(
#     r: int, ell: float, dtype: types.datatype = types.float64
# ) -> Tuple[DNDarray, DNDarray, types.datatype]:

#     # coefficients from (3.4) in the paper [Nakatsukasa, Freund '16 (SIAM Review, 2016)]
#     # evaluation of the elliptic integral and the elliptic functions is done with scipy (wrapper for C functions)

#     uu = np.arange(1, 2 * r + 1) * ellipkm1(ell**2) / (2 * r + 1)
#     ellipfcts = np.asarray(ellipj(uu, 1 - ell**2)[:2])
#     cc = ell**2 * ellipfcts[0, :] ** 2 / ellipfcts[1, :] ** 2
#     aa = np.zeros(r)
#     Mhat = 1
#     for j in range(1, r + 1):
#         p1 = 1
#         p2 = 1
#         for k in range(1, r + 1):
#             p1 *= cc[2 * j - 2] - cc[2 * k - 1]
#             if k != j:
#                 p2 *= cc[2 * j - 2] - cc[2 * k - 2]
#         aa[j - 1] = -p1 / p2
#         Mhat *= (1 + cc[2 * j - 2]) / (1 + cc[2 * j - 1])
#     # print(cc,aa,Mhat)
#     return (
#         factories.array(cc, dtype=dtype, split=None),
#         factories.array(aa, dtype=dtype, split=None),
#         dtype(Mhat),
#     )


def estimate_largest_singularvalue(A: DNDarray, algorithm: str = "fro") -> types.datatype:
    """
    Estimates the largest singular value of a matrix
    """
    # Estimates the largest singular value of the matrix A
    # 'fro': uses Frobenius norm of A as upper estimate
    if algorithm == "fro":
        return matrix_norm(A, ord="fro").flatten()
    else:
        raise NotImplementedError("So far only algorithm='fro' implemented.")


def estimate_l2_condition(A: DNDarray, algorithm: str, params: list) -> types.datatype:
    """
    Estimates the l2-condition number of a matrix (randomized!)
    """
    if algorithm == "GKL1995":
        # probabilistic upper estimate for the condition number according to the paper
        # of [Gudmundsson, Kenney, Laub (SIAM J. Matrix Analysis, 1995)]
        # actually, they estimate the condition number w.r.t. the Frobenius norm
        # this yields an upper bound for kappa_2 as well...

        nsamples = params[0]
        m = A.shape[0]
        n = A.shape[1]

        if n > m:
            raise RuntimeError(
                "Condition number estimator 'GKL1995' requires numbers of columns <= number of rows... Try other method?"
            )

        _, R = myqr(A, calc_R=True)
        del _

        # in [GKL95]: random samples from unit sphere
        Q = myqr(randn(n, nsamples, dtype=A.dtype, split=1, device=A.device, comm=A.comm))

        # --------------------------------------------------------------------------
        # !!! TODO: implementation that only uses heat instead of going back to torch
        #          -> requires: triangular solver in heat
        # --------------------------------------------------------------------------

        est = (
            matrix_norm(R @ Q)
            * A.dtype((m / nsamples) ** 0.5, comm=A.comm)
            * matrix_norm(triu_solver(R, Q.resplit(0)))
        )

        return est.squeeze()
    else:
        print(
            "Algorithm '",
            algorithm,
            "' not yet implemented. Returns 1e16 as a very pessimistic estimate for the condition number.",
        )
        return A.dtype(1e16, comm=A.comm)


# def minimal_iterations(r, kappa):
#     # see the respective table in the paper of Nakatsukasa and Freund
#     if kappa <= 1e2:
#         its = [0, 4, 3, 2, 2, 2, 2, 2, 2]
#     elif kappa <= 1e3:
#         its = [0, 3, 3, 2, 2, 2, 2, 2, 2]
#     elif kappa <= 1e5:
#         its = [0, 5, 3, 3, 3, 2, 2, 2, 2]
#     elif kappa <= 1e7:
#         its = [0, 5, 4, 3, 3, 3, 2, 2, 2]
#     else:
#         its = [6, 4, 3, 3, 3, 3, 3, 3, 2]
#     return its[r]


# def what_to_send_and_to_recv(source_loc_shapes, target_loc_shapes, source_idx, target_idx):
#     i = 0
#     j = 0
#     rem_i = source_loc_shapes[i]
#     rem_j = target_loc_shapes[j]
#     idx = 0
#     to_send = [[]]
#     to_recv = [[]]
#     while i < len(source_loc_shapes) and j < len(target_loc_shapes):
#         now_to_send = min(rem_i, rem_j)
#         to_send[i].append([target_idx[j], [idx, idx + now_to_send]])
#         to_recv[j].append([source_idx[i], now_to_send])
#         idx += now_to_send
#         rem_j -= now_to_send
#         rem_i -= now_to_send
#         if rem_j == 0:
#             j += 1
#             if j < len(target_loc_shapes):
#                 rem_j = target_loc_shapes[j]
#                 to_recv.append([])
#         if rem_i == 0:
#             i += 1
#             idx = 0
#             if i < len(source_loc_shapes):
#                 rem_i = source_loc_shapes[i]
#                 to_send.append([])
#     return to_send, to_recv


# def subspaceiteration(A: DNDarray, C: DNDarray, silent: bool = True) -> DNDarray:
#     # subspace iteration as required for ZoloSymEig [NF'16, Alg. 5.2]
#     # algorithm from [Nakatsukasa, Higham (SIAM J Sci Comp '13), Algorithm 3]

#     safetyparam = 3

#     if A.dtype == types.float64:
#         maxit = 3
#     elif A.dtype == types.float32:
#         maxit = 6
#     else:
#         raise RuntimeError("Wrong data type for A. Only float32 or float64 allowed.")

#     # TODO: determine suitable stopping criteria!
#     if A.dtype == types.float32:
#         tol = 1e-6
#     elif A.dtype == types.float64:
#         tol = 1e-12
#     Anorm = matrix_norm(A, ord="fro")

#     k = int(np.round(matrix_norm(C, ord="fro").numpy() ** 2))

#     # this initialization is proposed in [NH'13, Sect. 5.1]
#     # Brauchen hier Anpassung in percentile (da sonst globaler Kommunikator für output verwendet wird...)
#     columnnorms = vector_norm(C, axis=0)
#     idx = where(
#         columnnorms
#         >= factories.ones(columnnorms.shape, comm=columnnorms.comm, split=columnnorms.split)
#         * statistics.percentile(columnnorms, 100.0 * (1 - (k + safetyparam) / columnnorms.shape[0]))
#     )

#     X = C[:, idx].balance()

#     it = 1
#     while it < maxit + 1:
#         Q = myqr(X, full_Q=True)
#         Q_k = Q[:, :k].balance()
#         Q_k_orth = Q[:, k:].balance()
#         E = matmul(Q_k_orth.T, matmul(A, Q_k))
#         Enorm = matrix_norm(E, ord="fro")
#         if Enorm / Anorm < tol:
#             if A.comm.rank == 0 and not silent:
#                 print("\t\t Number of subspace iterations: ", it)
#             return Q, k
#         X = C @ Q_k
#         it += 1
#     if A.comm.rank == 0:
#         print(
#             "\t\t -> Watch out: Subspace iteration did not converge in %d iterations. \n \t\t    We have ||E||_F/||A||_F = %2.2e. \n"
#             % (maxit, Enorm / Anorm)
#         )

#     return Q, k


# def pd(
#     A: DNDarray,
#     r: int = 8,
#     calcH: bool = True,
#     silent: bool = True,
#     cond_est: float = 0,
# ) -> Tuple[DNDarray, DNDarray]:
#     """
#     Polar decomposition of a matrix
#     Following the 'Zolotarev-approch'
#     """
#     # Polar decomposition in two "iterations" utilizing Zolotarev functions
#     # based on [NF'16, Alg. 5.1]
#     # returns orthogonal matrix U and symmetric, positive definite matrix H
#     # such that A = UH (polar decomposition of A)

#     # note: variables are named as in the paper [NF'16]

#     # Some simplifactions:
#     #   2. only QR decompositions, no Cholesky (not implemented in heat so far..)
#     # TODO:
#     #   2. Replace QR by Cholesky in the second 'iteration': we need Cholesky and solve_triangular in heat
#     #      (possibly) Use QR solver that is adapted to the special matrix structure...
#     #   3. (possibly) improve condition number estimator at the beginning

#     # consistency checks
#     if not isinstance(A, DNDarray):
#         raise RuntimeError("Argument needs to be a DNDarray but is {}.".format(type(A)))
#     if A.shape[0] < A.shape[1]:
#         raise RuntimeError(
#             "Zolotarev-based polar decomposition requires numbers of columns <= number of rows... Try other method or transpose matrix?"
#         )
#     if not A.ndim == 2:
#         raise RuntimeError("A needs to be a 2D matrix")

#     if r > 8:
#         raise RuntimeError("It is required that r <= 8, but input was r=%d." % r)

#     # for simplicity: avoids different local shapes of the X (otherwise: communication of the 'vertical groups' (see below) becomes quite tricky... )
#     if A.comm.Get_size() < r:
#         raise RuntimeError(
#             "For Zolotarev-based PD, number of processes needs to be at least as large as r, but nprocs=%d while r=%d."
#             % (A.comm.Get_size(), r)
#         )

#     # determine required accuracy depending on data type
#     if A.dtype == types.float32:
#         tol = 1.19e-7
#     elif A.dtype == types.float64:
#         tol = 2.22e-16
#     else:
#         raise TypeError("Data type {} not supported".format(A.dtype))

#     # Estimate condition number and largest singular value  // TODO: das geht schon noch besser!
#     alpha = estimate_largest_singularvalue(A)

#     if cond_est == 0:
#         condesttype = "GKL1995"
#         condestparams = [max(2 * A.comm.Get_size(), int(A.shape[1] / 10))]
#         kappa = estimate_l2_condition(A, algorithm=condesttype, params=condestparams)
#         if kappa != kappa:  # check for NaN
#             kappa = kappa.dtype(1e16, comm=A.comm)
#             if A.comm.rank == 0 and not silent:
#                 print("Condition number estimator falied. Assume worst case condition...")
#     else:
#         kappa = cond_est

#     if A.comm.rank == 0 and not silent:
#         print("Condition number estimate: %2.2e" % kappa.numpy())

#     # unschöner workaraound:
#     if kappa < alpha:
#         alpha = kappa * 0.9

#     A /= alpha

#     # Introduce communicators and process groups used in the following
#     # note: we need mpi4py and mpi4py-wrapped-by-heat versions for some communicators...
#     global_comm = A.comm.handle
#     nprocs = global_comm.Get_size()

#     idx_all = [k for k in range(nprocs)]
#     idx_set_horizontal = [
#         [k for k in range(nprocs) if k % r == j and k < nprocs - nprocs % r] for j in range(r)
#     ]
#     idx_used = [k for k in range(nprocs) if k < nprocs - nprocs % r]
#     groups_horizontal = [global_comm.group.Incl(idx) for idx in idx_set_horizontal]
#     horizontal_comms = [global_comm.Create_group(red_group) for red_group in groups_horizontal]
#     ht_horizontal_comms = [MPICommunication(handle=red_comm) for red_comm in horizontal_comms]

#     idx_set_vertical = [[j * r + k for k in range(r)] for j in range(len(idx_set_horizontal[0]))]
#     groups_vertical = [global_comm.group.Incl(idx) for idx in idx_set_vertical]
#     vertical_comms = [global_comm.Create_group(vert_group) for vert_group in groups_vertical]

#     # jeweilige Kommunikatoren für horizontale und vertikale Gruppen festlegen
#     for k in range(r):
#         if global_comm.rank in idx_set_horizontal[k]:
#             ht_horizontal_comm = ht_horizontal_comms[k]

#     """
#     ----------------------------- Vorstellung dahinter ---------------------------------
#         -> r-viele (hier r=8) 'horizontale Gruppen'
#         -> so viele 'vertikale Gruppen', dass es aufgeht...
#         -> ein paar Prozesse (max. r-1) werden ggf. teilweise nicht benutzt...

#                                 idx_set_vert[0]     idx_set_vert[1]     idx_set_vert[2]     idx_set_vert[3] . . .
#     idx_set_horizontal[0]:      0                   8                   16                  24              ...
#     idx_set_horizontal[1]:      1                   9                   17                  25              ...
#     idx_set_horizontal[2]:      2                   10                  18                  26              ...
#     idx_set_horizontal[3]:      3                   11                  19                  27              ...
#     idx_set_horizontal[4]:      4                   12                  20                  28              ...
#     idx_set_horizontal[5]:      5                   13                  21                  29              ...
#     idx_set_horizontal[6]:      6                   14                  22                  30              ...
#     idx_set_horizontal[7]:      7                   15                  23                  31              ...

#     Zunächst lebt A auf Prozessen 0,1,2,...,nprocs
#     Dann wird auf jeder der 8 horizontalen Gruppen (no. 0-7) eine Kopie von A erzeugt (heißt X)
#     Mit diesen 8 Kopien von A wird dann gearbeitet (u.a. QR-Zerlegungen etc.)
#     Summation über die 8 weiterverarbeiteten Kopien erfordert dann die vertikalen Gruppen...

#     ganz am Ende: X auf horizontaler Gruppe 0 wird wieder auf alle Prozesse zurückkopiert...

#     """

#     # Hier wird A r mal kopiert und auf insg. r Prozessgruppen (Liste idx_set) verteilt

#     if A.split == 0:
#         A_local_shapes = A.lshape_map[:, 0].numpy().tolist()
#     else:
#         A_local_shapes = A.lshape_map[:, 1].numpy().tolist()

#     X_local_shapes = 0

#     # TODO: get local shapes directly instead of creating X as zero array!
#     if global_comm.rank in idx_used:
#         X = factories.zeros(A.shape, dtype=A.dtype, split=A.split, comm=ht_horizontal_comm)
#         if X.split == 0:
#             X_local_shapes = X.lshape_map[:, 0]
#         else:
#             X_local_shapes = X.lshape_map[:, 1]

#     X_local_shapes = global_comm.bcast(X_local_shapes, root=0).numpy().tolist()

#     to_send = []
#     for k in range(r):
#         to_send_prelim, to_recv_prelim = what_to_send_and_to_recv(
#             A_local_shapes, X_local_shapes, idx_all, idx_set_horizontal[k]
#         )
#         to_send += to_send_prelim[global_comm.rank]
#         if global_comm.rank in idx_set_horizontal[k]:
#             to_recv = to_recv_prelim[int((global_comm.rank - k) / r)]

#     if global_comm.rank in idx_used:
#         if A.split == 0:
#             recv_bufs = [
#                 torch.zeros(
#                     (entry[1], A.shape[1]), dtype=A.dtype.torch_type(), device=A.device.torch_device
#                 )
#                 for entry in to_recv
#             ]
#         else:
#             recv_bufs = [
#                 torch.zeros(
#                     (A.shape[0], entry[1]), dtype=A.dtype.torch_type(), device=A.device.torch_device
#                 )
#                 for entry in to_recv
#             ]
#         reqs = [global_comm.Irecv(recv_bufs[k], to_recv[k][0], tag=1) for k in range(len(to_recv))]

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

#     if global_comm.rank in idx_used:
#         [req.Wait() for req in reqs]

#     if global_comm.rank in idx_used:
#         if A.split == 0:
#             X.larray = torch.vstack(recv_bufs)
#         elif A.split == 1:
#             X.larray = torch.hstack(recv_bufs)
#         del recv_bufs

#     # for k in range(len(idx_set)):
#     #     if global_comm.rank in idx_set[k]:
#     #         print('Process %d: \t A local shape %d x %d \t X local shape %d x %d (group %d consisting of'%(global_comm.rank, A.larray.shape[0], A.larray.shape[1], X.larray.shape[0], X.larray.shape[1], k), idx_set[k], ')')

#     # Jetzt kommt die eigentliche "Iteration"
#     it = 0
#     itmax = minimal_iterations(r, kappa)
#     ell = 1.0 / kappa
#     c, a, Mhat = compute_zolotarev_coefficients(r, ell.numpy())
#     breakflag = False

#     while it < itmax + 2:
#         it += 1
#         if global_comm.rank == 0 and not silent:
#             print("Now comes Zolo-PD iteration %d..." % it)
#         for k in range(r):
#             if global_comm.rank in idx_set_horizontal[k]:
#                 # berechne die r-vielen QR-Zerlegungen in parallel
#                 cdummy = A.dtype(c[2 * k], comm=ht_horizontal_comms[k])
#                 adummy = A.dtype(a[k], comm=ht_horizontal_comms[k])
#                 Mhatdummy = A.dtype(Mhat, comm=ht_horizontal_comms[k])
#                 rdummy = A.dtype(r, comm=ht_horizontal_comms[k])
#                 X = vstack(
#                     [
#                         X,
#                         sqrt(cdummy)
#                         * factories.eye(
#                             X.shape[1], dtype=A.dtype, comm=ht_horizontal_comms[k], split=X.split
#                         ),
#                     ]
#                 )
#                 Q = myqr(X)
#                 Q1 = Q[: A.shape[0], : A.shape[1]].balance()
#                 Q2 = Q[A.shape[0] :, : A.shape[1]].T.balance()
#                 del Q
#                 X = Mhatdummy * (
#                     X[: A.shape[0], :].balance() / rdummy
#                     + adummy / sqrt(cdummy) * matmul(Q1, Q2).resplit_(A.split)
#                 )
#                 del Q1
#                 del Q2
#         for k in range(len(idx_set_vertical)):
#             if global_comm.rank in idx_set_vertical[k]:
#                 # summiere die QR-Zerlegungen auf: Hier braucht man insbesondere, dass die X alle die selbe Form haben
#                 # so spart man sich nochmal recht viel Kommunikation, lässt aber auch schlimmstenfalls 7 Prozessoren unbenutzt...
#                 Xold = X
#                 vertical_comms[k].Allreduce(MPI.IN_PLACE, X.larray)
#         if it >= itmax:
#             if global_comm.rank in idx_set_horizontal[0]:
#                 # if A.split == 1:
#                 #     X.resplit_(0)
#                 #     Xold.resplit_(0)
#                 if matrix_norm(X - Xold, ord="fro") / matrix_norm(X, ord="fro") <= tol ** (
#                     1 / (2 * r + 1)
#                 ):
#                     breakflag = True
#                     del Xold
#                     if global_comm.rank == 0 and not silent:
#                         print("Desired tolerance reached after iteration %d." % it)
#                 # if A.split == 1:
#                 #     X.resplit_(1)
#             breakflag = global_comm.bcast(breakflag, root=0)
#             if breakflag:
#                 break
#         # falls nicht Abbruch, Koeffizienten updaten für nächste Iteration...
#         ellold = ell
#         ell = 1
#         for j in range(r):
#             ell *= (ellold**2 + c[2 * j + 1]) / (ellold**2 + c[2 * j])
#         ell *= Mhat * ellold
#         c, a, Mhat = compute_zolotarev_coefficients(r, ell.numpy())

#     # Jetzt muss X von Prozessgruppe idx_set_horizontal[0] auf alle Prozesse zurückkopiert werden...

#     U = factories.zeros(A.shape, dtype=A.dtype, split=A.split, comm=A.comm)

#     to_send, to_recv = what_to_send_and_to_recv(
#         X_local_shapes, A_local_shapes, idx_set_horizontal[0], idx_all
#     )

#     if A.split == 0:
#         recv_bufs = [
#             torch.zeros(
#                 (entry[1], A.shape[1]), dtype=A.dtype.torch_type(), device=A.device.torch_device
#             )
#             for entry in to_recv[global_comm.rank]
#         ]
#     else:
#         recv_bufs = [
#             torch.zeros(
#                 (A.shape[0], entry[1]), dtype=A.dtype.torch_type(), device=A.device.torch_device
#             )
#             for entry in to_recv[global_comm.rank]
#         ]
#     reqs = [
#         global_comm.Irecv(recv_bufs[k], to_recv[global_comm.rank][k][0], tag=2)
#         for k in range(len(to_recv[global_comm.rank]))
#     ]

#     if global_comm.rank in idx_set_horizontal[0]:
#         if X.split == 0:
#             [
#                 global_comm.Send(X.larray[entry[1][0] : entry[1][1], :].clone(), entry[0], tag=2)
#                 for entry in to_send[int(global_comm.rank / r)]
#             ]
#         else:
#             [
#                 global_comm.Send(X.larray[:, entry[1][0] : entry[1][1]].clone(), entry[0], tag=2)
#                 for entry in to_send[int(global_comm.rank / r)]
#             ]

#     [req.Wait() for req in reqs]

#     if global_comm.rank in idx_used:
#         del X

#     if A.split == 0:
#         U.larray = torch.vstack(recv_bufs)
#     elif A.split == 1:
#         U.larray = torch.hstack(recv_bufs)

#     del recv_bufs

#     # Postprocessing: berechne H
#     if calcH:
#         H = matmul(U.T, A) * alpha
#         H = 0.5 * (H + H.T.resplit(H.split))
#         return U, H.resplit(A.split)
#     else:
#         return U


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
