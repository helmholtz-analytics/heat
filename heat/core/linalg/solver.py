"""
Collection of solvers for systems of linear equations.
"""
import heat as ht
from ..dndarray import DNDarray
from ..sanitation import sanitize_out
from typing import List, Dict, Any, TypeVar, Union, Tuple, Optional

import torch

__all__ = ["cg", "lanczos"]


def cg(A: DNDarray, b: DNDarray, x0: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Conjugate gradients method for solving a system of linear equations :math: `Ax = b`

    Parameters
    ----------
    A : DNDarray
        2D symmetric, positive definite Matrix
    b : DNDarray
        1D vector
    x0 : DNDarray
        Arbitrary 1D starting vector
    out : DNDarray, optional
        Output Vector
    """
    if not isinstance(A, DNDarray) or not isinstance(b, DNDarray) or not isinstance(x0, DNDarray):
        raise TypeError(
            f"A, b and x0 need to be of type ht.DNDarray, but were {type(A)}, {type(b)}, {type(x0)}"
        )

    if A.ndim != 2:
        raise RuntimeError("A needs to be a 2D matrix")
    if b.ndim != 1:
        raise RuntimeError("b needs to be a 1D vector")
    if x0.ndim != 1:
        raise RuntimeError("c needs to be a 1D vector")

    r = b - ht.matmul(A, x0)
    p = r
    rsold = ht.matmul(r, r)
    x = x0

    for i in range(len(b)):
        Ap = ht.matmul(A, p)
        alpha = rsold / ht.matmul(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = ht.matmul(r, r)
        if ht.sqrt(rsnew).item() < 1e-10:
            print(f"Residual reaches tolerance in it = {i}")
            if out is not None:
                out = x
                return out
            return x
        p = r + ((rsnew / rsold) * p)
        rsold = rsnew

    if out is not None:
        out = x
        return out
    return x


def lanczos(
    A: DNDarray,
    m: int,
    v0: Optional[DNDarray] = None,
    V_out: Optional[DNDarray] = None,
    T_out: Optional[DNDarray] = None,
) -> Tuple[DNDarray, DNDarray]:
    r"""
    The Lanczos algorithm is an iterative approximation of the solution to the eigenvalue problem, as an adaptation of
    power methods to find the m "most useful" (tending towards extreme highest/lowest) eigenvalues and eigenvectors of
    an :math:`n \times n` Hermitian matrix, where often :math:`m<<n`.
    It returns two matrices :math:`V` and :math:`T`, where:

        - :math:`V` is a Matrix of size :math:`n\times m`, with orthonormal columns, that span the Krylow subspace \n
        - :math:`T` is a Tridiagonal matrix of size :math:`m\times m`, with coefficients :math:`\alpha_1,..., \alpha_n`
          on the diagonal and coefficients :math:`\beta_1,...,\beta_{n-1}` on the side-diagonals\n

    Parameters
    ----------
    A : DNDarray
        2D Hermitian (if complex) or symmetric positive-definite matrix.
        Only distribution along axis 0 is supported, i.e. `A.split` must be `0` or `None`.
    m : int
        Number of Lanczos iterations
    v0 : DNDarray, optional
        1D starting vector of Euclidean norm 1. If not provided, a random vector will be used to start the algorithm
    V_out : DNDarray, optional
        Output Matrix for the Krylow vectors, Shape = (n, m), dtype=A.dtype, must be initialized to zero
    T_out : DNDarray, optional
        Output Matrix for the Tridiagonal matrix, Shape = (m, m), must be initialized to zero
    """
    if not isinstance(A, DNDarray):
        raise TypeError(f"A needs to be of type ht.dndarray, but was {type(A)}")
    if A.ndim != 2:
        raise RuntimeError("A needs to be a 2D matrix")
    if A.dtype is ht.int32 or A.dtype is ht.int64:
        raise TypeError(f"A can be float or complex, got {A.dtype}")
    if not isinstance(m, (int, float)):
        raise TypeError(f"m must be int, got {type(m)}")

    n, column = A.shape
    if n != column:
        raise TypeError("Input Matrix A needs to be symmetric positive-definite.")

    # output data types: T is always Real
    A_is_complex = A.dtype is ht.complex128 or A.dtype is ht.complex64
    T_dtype = A.real.dtype

    # initialize or sanitize output buffers
    if T_out is not None:
        sanitize_out(
            T_out,
            output_shape=(m, m),
            output_split=None,
            output_device=A.device,
            output_comm=A.comm,
        )
        T = T_out
    else:
        T = ht.zeros((m, m), dtype=T_dtype, device=A.device, comm=A.comm)
    if A.split == 0:
        if V_out is not None:
            sanitize_out(
                V_out,
                output_shape=(n, m),
                output_split=0,
                output_device=A.device,
                output_comm=A.comm,
            )
            V = V_out
        else:
            # This is done for better memory access in the reorthogonalization Gram-Schmidt algorithm
            V = ht.zeros((n, m), split=0, dtype=A.dtype, device=A.device, comm=A.comm)
    else:
        if A.split == 1:
            raise NotImplementedError("Distribution along axis 1 not implemented yet.")
        if V_out is not None:
            sanitize_out(
                V_out,
                output_shape=(n, m),
                output_split=None,
                output_device=A.device,
                output_comm=A.comm,
            )
            V = V_out
        else:
            V = ht.zeros((n, m), split=None, dtype=A.dtype, device=A.device, comm=A.comm)

    if A_is_complex:
        if v0 is None:
            vr = (
                ht.random.rand(n, split=V.split, dtype=T_dtype, device=V.device, comm=V.comm)
                + ht.random.rand(n, split=V.split, dtype=T_dtype, device=V.device, comm=V.comm) * 1j
            )
            v0 = vr / ht.norm(vr)
        elif v0.split != V.split:
            v0.resplit_(axis=V.split)
        # # 0th iteration
        # # vector v0 has Euclidean norm = 1
        w = ht.matmul(A, v0)
        alpha = ht.dot(ht.conj(w).T, v0)
        w = w - alpha * v0
        T[0, 0] = alpha.real
        V[:, 0] = v0
        for i in range(1, int(m)):
            beta = ht.norm(w)
            if ht.abs(beta) < 1e-10:
                # print("Lanczos breakdown in iteration {}".format(i))
                # Lanczos Breakdown, pick a random vector to continue
                vr = (
                    ht.random.rand(n, split=V.split, dtype=T_dtype, device=V.device, comm=V.comm)
                    + ht.random.rand(n, split=V.split, dtype=T_dtype, device=V.device, comm=V.comm)
                    * 1j
                )
                # orthogonalize v_r with respect to all vectors v[i]
                for j in range(i):
                    vi_loc = V._DNDarray__array[:, j]
                    a = torch.dot(vr.larray, torch.conj(vi_loc))
                    b = torch.dot(vi_loc, torch.conj(vi_loc))
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, a, ht.communication.MPI.SUM)
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, b, ht.communication.MPI.SUM)
                    vr._DNDarray__array = vr._DNDarray__array - a / b * vi_loc
                # normalize v_r to Euclidean norm 1 and set as ith vector v
                vi = vr / ht.norm(vr)
            else:
                vr = w

                # Reorthogonalization
                for j in range(i):
                    vi_loc = V.larray[:, j]
                    a = torch.dot(vr._DNDarray__array, torch.conj(vi_loc))
                    b = torch.dot(vi_loc, torch.conj(vi_loc))
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, a, ht.communication.MPI.SUM)
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, b, ht.communication.MPI.SUM)
                    vr._DNDarray__array = vr._DNDarray__array - a / b * vi_loc

                vi = vr / ht.norm(vr)

            w = ht.matmul(A, vi)
            alpha = ht.dot(ht.conj(w).T, vi)

            w = w - alpha * vi - beta * V[:, i - 1]

            T[i - 1, i] = beta.real
            T[i, i - 1] = beta.real
            T[i, i] = alpha.real
            V[:, i] = vi
    else:
        if v0 is None:
            vr = ht.random.rand(n, split=V.split, dtype=T_dtype, device=V.device, comm=V.comm)
            v0 = vr / ht.norm(vr)
        elif v0.split != V.split:
            v0.resplit_(axis=V.split)
        # # 0th iteration
        # # vector v0 has Euclidean norm = 1
        w = ht.matmul(A, v0)
        alpha = ht.dot(w, v0)
        w = w - alpha * v0
        T[0, 0] = alpha
        V[:, 0] = v0
        for i in range(1, int(m)):
            beta = ht.norm(w)
            if ht.abs(beta) < 1e-10:
                # print("Lanczos breakdown in iteration {}".format(i))
                # Lanczos Breakdown, pick a random vector to continue
                vr = ht.random.rand(n, split=V.split, dtype=T_dtype, device=V.device, comm=V.comm)
                # orthogonalize v_r with respect to all vectors v[i]
                for j in range(i):
                    vi_loc = V._DNDarray__array[:, j]
                    a = torch.dot(vr.larray, vi_loc)
                    b = torch.dot(vi_loc, vi_loc)
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, a, ht.communication.MPI.SUM)
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, b, ht.communication.MPI.SUM)
                    vr._DNDarray__array = vr._DNDarray__array - a / b * vi_loc
                # normalize v_r to Euclidean norm 1 and set as ith vector v
                vi = vr / ht.norm(vr)
            else:
                vr = w

                # Reorthogonalization
                for j in range(i):
                    vi_loc = V.larray[:, j]
                    a = torch.dot(vr._DNDarray__array, vi_loc)
                    b = torch.dot(vi_loc, vi_loc)
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, a, ht.communication.MPI.SUM)
                    A.comm.Allreduce(ht.communication.MPI.IN_PLACE, b, ht.communication.MPI.SUM)
                    vr._DNDarray__array = vr._DNDarray__array - a / b * vi_loc

                vi = vr / ht.norm(vr)

            w = ht.matmul(A, vi)
            alpha = ht.dot(w, vi)

            w = w - alpha * vi - beta * V[:, i - 1]

            T[i - 1, i] = beta
            T[i, i - 1] = beta
            T[i, i] = alpha
            V[:, i] = vi

    if V.split is not None:
        V.resplit_(axis=None)

    return V, T
