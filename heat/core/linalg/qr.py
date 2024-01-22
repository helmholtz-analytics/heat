"""
QR decomposition of (distributed) 2-D ``DNDarray``s.
"""
import collections
import torch
from typing import Tuple

from ..dndarray import DNDarray
from ..manipulations import hstack
from ..random import randn
from .. import factories

__all__ = ["qr"]


def qr(
    A: DNDarray,
    calc_r: bool = True,
    calc_q: bool = True,
    overwrite_a: bool = False,
    full_q: bool = False,
    crop_r_at: int = None,
) -> Tuple[DNDarray, DNDarray]:
    r"""
    Calculates the QR decomposition of a 2D ``DNDarray``.
    Factor the matrix ``A`` as *QR*, where ``Q`` is orthonormal and ``R`` is upper-triangular.
    If ``calc_q==True``, function returns ``QR(Q=Q, R=R)``, else function returns ``QR(Q=None, R=R)``

    Parameters
    ----------
    A : DNDarray
        Array which will be decomposed
    calc_r : bool, optional
        Whether or not to calculate R.
        If ``True``, function returns ``(Q, R)``.
        If ``False``, function returns ``(Q, None)``.
    calc_q : bool, optional
        Whether or not to calculate Q.
        If ``True``, function returns ``(Q, R)``.
        If ``False``, function returns ``(None, R)``.
    overwrite_a : bool, optional
        If ``True``, function overwrites ``a`` with R
        If ``False``, a new array will be created for R
    full_q : bool, optional
        If ``True``, function returns the full (i.e. square) Q matrix; note that this option may result in heaviy memory overhead.
        If ``False`` (default), function returns the reduced Q matrix.
    crop_r_at : int, optional
        for internal use only, do not set this parameter

    Notes
    -----
    This function is built on top of PyTorch's QR function. ``torch.linalg.qr()`` using LAPACK (CPU) and MAGMA (CUDA) on
    the backend; due to limited availability of QR in ROCm, this function is currently not available on AMD GPUs.
    Basic information about QR factorization/decomposition can be found at
    https://en.wikipedia.org/wiki/QR_factorization.
    """
    if not isinstance(A, DNDarray):
        raise TypeError(f"'A' must be a DNDarray, but is {type(A)}")
    if not isinstance(calc_q, bool):
        raise TypeError(f"calc_q must be a bool, currently {type(calc_q)}")
    if not isinstance(calc_r, bool):
        raise TypeError(f"calc_r must be a bool, currently {type(calc_r)}")
    if not isinstance(overwrite_a, bool):
        raise TypeError(f"overwrite_a must be a bool, currently {type(overwrite_a)}")
    if not isinstance(full_q, bool):
        raise TypeError(f"full_q must be a bool, currently {type(full_q)}")
    if crop_r_at is not None and not isinstance(crop_r_at, int):
        raise TypeError(f"crop_r_at must be a bool, currently {type(crop_r_at)}")
    if A.ndim != 2:
        raise ValueError(f"Array 'A' must be 2 dimensional, buts has {A.ndim} dimensions")

    if a.split == 0 and tiles_per_proc == 1:
        raise Warning(
            "Using tiles_per_proc=1 with split=0 can result in an error. We highly recommend to use tiles_per_proc > 1."
        )

    QR = collections.namedtuple("QR", "Q, R")

    if A.split == 0 and A.is_distributed():
        raise NotImplementedError(
            "QR decomposition is currently not implemented for split dimension 0. An implementation of TS-QR is going to close this gap soon."
        )

    if not A.is_distributed():
        if not full_q:
            Q, R = torch.linalg.qr(A.larray, mode="reduced")
        else:
            Q, R = torch.linalg.qr(A.larray)
        Q = factories.array(Q, dtype=A.dtype, split=A.split, device=A.device, comm=A.comm)
        if calc_r:
            R = factories.array(R, dtype=A.dtype, split=A.split, device=A.device, comm=A.comm)
            return QR(Q, R)
        else:
            return QR(Q, None)

    if A.split == 1:
        if full_q:
            if A.shape[1] < A.shape[0]:
                fill_up_array = randn(
                    A.shape[0],
                    A.shape[0] - A.shape[1],
                    dtype=A.dtype,
                    split=A.split,
                    device=A.device,
                    comm=A.comm,
                )
                A_tilde = hstack([A, fill_up_array]).balance()
                # A_tilde = hstack([A, factories.ones((A.shape[0], A.shape[0]-A.shape[1]), dtype=A.dtype, split=A.split,device=A.device, comm=A.comm)]).balance()
                return qr(A_tilde, calc_r=calc_r, full_q=full_q, crop_r_at=A.shape[1])

        lshapes = A.lshape_map[:, 1]
        lshapes_cum = torch.cumsum(lshapes, 0)
        nprocs = A.comm.size

        if A.shape[0] >= A.shape[1]:
            last_row_reached = nprocs
            k = A.shape[1]
        else:
            last_row_reached = min(torch.argwhere(lshapes_cum >= A.shape[0]))[0]
            k = A.shape[0]

        Q = factories.zeros(A.shape, dtype=A.dtype, split=1, device=A.device, comm=A.comm)
        if calc_r:
            R = factories.zeros(
                (k, A.shape[1]), dtype=A.dtype, split=1, device=A.device, comm=A.comm
            )
            R_shapes = torch.hstack(
                [torch.zeros(1, dtype=torch.int32), torch.cumsum(R.lshape_map[:, 1], 0)]
            )
        A_columns = A.larray.clone()

        for i in range(last_row_reached + 1):
            if i < nprocs - 1:
                k_loc_i = min(A.shape[0], A.lshape_map[i, 1])
                Q_buf = torch.zeros(
                    (A.shape[0], k_loc_i), dtype=A.larray.dtype, device=A.device.torch_device
                )

            if A.comm.rank == i:
                Q.larray, R_loc = torch.linalg.qr(A_columns, mode="reduced")
                if i < nprocs - 1:
                    Q_buf = Q.larray
                if calc_r:
                    r_size = R.larray[R_shapes[i] : R_shapes[i + 1], :].shape[0]
                    R.larray[R_shapes[i] : R_shapes[i + 1], :] = R_loc[:r_size, :]

            if i < nprocs - 1:
                req = A.comm.Ibcast(Q_buf, root=i)
                req.Wait()

            if A.comm.rank > i:
                R_loc = Q_buf.T @ A_columns
                A_columns -= Q_buf @ R_loc
                if calc_r:
                    r_size = R.larray[R_shapes[i] : R_shapes[i + 1], :].shape[0]
                    R.larray[R_shapes[i] : R_shapes[i + 1], :] = R_loc[:r_size, :]

        if calc_r:
            if crop_r_at != 0:
                return Q[:, :k].balance(), R[:, :crop_r_at].balance()
            else:
                return Q[:, :k].balance(), R
        else:
            return Q[:, :k].balance()
