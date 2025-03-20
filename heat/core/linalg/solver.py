"""
Collection of solvers for systems of linear equations.
"""

import heat as ht
from ..dndarray import DNDarray
from ..sanitation import sanitize_out
from typing import List, Dict, Any, TypeVar, Union, Tuple, Optional
from .. import factories

import torch

__all__ = ["cg", "lanczos", "solve_triangular"]


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


def solve_triangular(A: DNDarray, b: DNDarray) -> DNDarray:
    """
    Solver for (possibly batched) upper triangular systems of linear equations: it returns `x` in `Ax = b`, where `A` is a (possibly batched) upper triangular matrix and
    `b` a (possibly batched) vector or matrix of suitable shape, both provided as input to the function.
    The implementation builts on the corresponding solver in PyTorch and implements an memory-distributed, MPI-parallel block-wise version thereof.

    Parameters
    ----------
    A : DNDarray
        An upper triangular invertible square (n x n) matrix or a batch thereof,  i.e. a ``DNDarray`` of shape `(..., n, n)`.
    b : DNDarray
        a (possibly batched) n x k matrix, i.e. an DNDarray of shape (..., n, k), where the batch-dimensions denoted by ... need to coincide with those of A.
        (Batched) Vectors have to be provided as ... x n x 1 matrices and the split dimension of b must the second last dimension if not None.
    Note
    ---------
    Since such a check might be computationally expensive, we do not check whether A is indeed upper triangular.
    If you require such a check, please open an issue on our GitHub page and request this feature.
    """
    if not isinstance(A, DNDarray) or not isinstance(b, DNDarray):
        raise TypeError(f"Arguments need to be of type DNDarray, got {type(A)}, {type(b)}.")
    if not A.ndim >= 2:
        raise ValueError("A needs to be a (batched) matrix.")
    if not b.ndim == A.ndim:
        raise ValueError("b needs to have the same number of (batch) dimensions as A.")
    if not A.shape[-2] == A.shape[-1]:
        raise ValueError("A needs to be a (batched) square matrix.")

    batch_dim = A.ndim - 2
    batch_shape = A.shape[:batch_dim]

    if not A.shape[:batch_dim] == b.shape[:batch_dim]:
        raise ValueError("Batch dimensions of A and b must be of the same shape.")
    if b.split == batch_dim + 1:
        raise ValueError("split=1 is not allowed for the right hand side.")
    if not b.shape[batch_dim] == A.shape[-1]:
        raise ValueError("Dimension mismatch of A and b.")

    if (
        A.split is not None and A.split < batch_dim or b.split is not None and b.split < batch_dim
    ):  # batch split
        if A.split != b.split:
            raise ValueError(
                "If a split dimension is a batch dimension, A and b must have the same split dimension. A possible solution would be a resplit of A or b to the same split dimension."
            )
    else:
        if (
            A.split is not None and b.split is not None
        ):  # both la dimensions split --> b.split = batch_dim
            # TODO remove?
            if not all(A.lshape_map[:, A.split] == b.lshape_map[:, batch_dim]):
                raise RuntimeError(
                    "The process-local arrays of A and b have different sizes along the splitted axis. This is most likely due to one of the DNDarrays being in unbalanced state. \n Consider using `A.is_balanced(force_check=True)` and `b.is_balanced(force_check=True)` to check if A and b are balanced; \n then call `A.balance_()` and/or `b.balance_()` in order to achieve equal local shapes along the split axis before applying `solve_triangular`."
                )

    comm = A.comm
    dev = A.device
    tdev = dev.torch_device

    nprocs = comm.Get_size()

    if A.split is None:  # A not split
        if b.split is None:
            x = torch.linalg.solve_triangular(A.larray, b.larray, upper=True)

            return factories.array(x, dtype=b.dtype, device=dev, comm=comm)
        else:  # A not split, b.split == -2
            b_lshapes_cum = torch.hstack(
                [
                    torch.zeros(1, dtype=torch.int64, device=tdev),
                    torch.cumsum(b.lshape_map[:, -2], 0),
                ]
            )

            btilde_loc = b.larray.clone()
            A_loc = A.larray[..., b_lshapes_cum[comm.rank] : b_lshapes_cum[comm.rank + 1]]

            x = factories.zeros_like(b, device=dev, comm=comm)

            for i in range(nprocs - 1, 0, -1):
                count = x.lshape_map[:, batch_dim].to(torch.device("cpu")).clone().numpy()
                displ = b_lshapes_cum[:-1].to(torch.device("cpu")).clone().numpy()
                count[i:] = 0  # nothing to send, as there are only zero rows
                displ[i:] = 0

                res_send = torch.empty(0)
                res_recv = torch.zeros((*batch_shape, count[comm.rank], b.shape[-1]), device=tdev)

                if comm.rank == i:
                    x.larray = torch.linalg.solve_triangular(
                        A_loc[..., b_lshapes_cum[i] : b_lshapes_cum[i + 1], :],
                        btilde_loc,
                        upper=True,
                    )
                    res_send = A_loc @ x.larray

                comm.Scatterv((res_send, count, displ), res_recv, root=i, axis=batch_dim)

                if comm.rank < i:
                    btilde_loc -= res_recv

            if comm.rank == 0:
                x.larray = torch.linalg.solve_triangular(
                    A_loc[..., : b_lshapes_cum[1], :], btilde_loc, upper=True
                )

            return x

    if A.split < batch_dim:  # batch split
        x = factories.zeros_like(b, device=dev, comm=comm, split=A.split)
        x.larray = torch.linalg.solve_triangular(A.larray, b.larray, upper=True)

        return x

    if A.split >= batch_dim:  # both splits in la dims
        A_lshapes_cum = torch.hstack(
            [
                torch.zeros(1, dtype=torch.int64, device=tdev),
                torch.cumsum(A.lshape_map[:, A.split], 0),
            ]
        )

        if b.split is None:
            btilde_loc = b.larray[
                ..., A_lshapes_cum[comm.rank] : A_lshapes_cum[comm.rank + 1], :
            ].clone()
        else:  # b is split at la dim 0
            btilde_loc = b.larray.clone()

        x = factories.zeros_like(
            b, device=dev, comm=comm, split=batch_dim
        )  # split at la dim 0 in case b is not split

        if A.split == batch_dim + 1:
            for i in range(nprocs - 1, 0, -1):
                count = x.lshape_map[:, batch_dim].to(torch.device("cpu")).clone().numpy()
                displ = A_lshapes_cum[:-1].to(torch.device("cpu")).clone().numpy()
                count[i:] = 0  # nothing to send, as there are only zero rows
                displ[i:] = 0

                res_send = torch.empty(0)
                res_recv = torch.zeros(
                    (*batch_shape, count[comm.rank], b.shape[-1]),
                    device=tdev,
                    dtype=b.dtype.torch_type(),
                )

                if comm.rank == i:
                    x.larray = torch.linalg.solve_triangular(
                        A.larray[..., A_lshapes_cum[i] : A_lshapes_cum[i + 1], :],
                        btilde_loc,
                        upper=True,
                    )
                    res_send = A.larray @ x.larray

                comm.Scatterv((res_send, count, displ), res_recv, root=i, axis=batch_dim)

                if comm.rank < i:
                    btilde_loc -= res_recv

            if comm.rank == 0:
                x.larray = torch.linalg.solve_triangular(
                    A.larray[..., : A_lshapes_cum[1], :], btilde_loc, upper=True
                )

        else:  # split dim is la dim 0
            for i in range(nprocs - 1, 0, -1):
                idims = tuple(x.lshape_map[i])
                if comm.rank == i:
                    x.larray = torch.linalg.solve_triangular(
                        A.larray[..., :, A_lshapes_cum[i] : A_lshapes_cum[i + 1]],
                        btilde_loc,
                        upper=True,
                    )
                    x_from_i = x.larray
                else:
                    x_from_i = torch.zeros(
                        idims,
                        dtype=b.dtype.torch_type(),
                        device=tdev,
                    )

                comm.Bcast(x_from_i, root=i)

                if comm.rank < i:
                    btilde_loc -= (
                        A.larray[..., :, A_lshapes_cum[i] : A_lshapes_cum[i + 1]] @ x_from_i
                    )

            if comm.rank == 0:
                x.larray = torch.linalg.solve_triangular(
                    A.larray[..., :, : A_lshapes_cum[1]], btilde_loc, upper=True
                )

        return x
