"""
QR decomposition of ``DNDarray``s.
"""

import collections
import torch
from typing import Tuple

from ..dndarray import DNDarray
from ..manipulations import concatenate
from .. import factories
from .. import communication
from ..types import float32, float64

__all__ = ["qr"]


def qr(
    A: DNDarray,
    mode: str = "reduced",
    procs_to_merge: int = 2,
) -> Tuple[DNDarray, DNDarray]:
    r"""
    Calculates the QR decomposition of a 2D ``DNDarray``.
    Factor the matrix ``A`` as *QR*, where ``Q`` is orthonormal and ``R`` is upper-triangular.
    If ``mode = "reduced``, function returns ``QR(Q=Q, R=R)``, if ``mode = "r"`` function returns ``QR(Q=None, R=R)``

    This function also works for batches of matrices; in this case, the last two dimensions of the input array are considered as the matrix dimensions.
    The output arrays have the same leading batch dimensions as the input array.

    Parameters
    ----------
    A : DNDarray of shape (M, N), of shape (...,M,N) in the batched case
        Array which will be decomposed. So far only arrays with datatype float32 or float64 are supported
    mode : str, optional
        default "reduced" returns Q and R with dimensions (M, min(M,N)) and (min(M,N), N). Potential batch dimensions are not modified.
        "r" returns only R, with dimensions (min(M,N), N).
    procs_to_merge : int, optional
        This parameter is only relevant for split=0 (-2, in the batched case) and determines the number of processes to be merged at one step during the so-called TS-QR algorithm.
        The default is 2. Higher choices might be faster, but will probably result in higher memory consumption. 0 corresponds to merging all processes at once.
        We only recommend to modify this parameter if you are familiar with the TS-QR algorithm (see the references below).

    Notes
    -----
    The distribution schemes of ``Q`` and ``R`` depend on that of the input ``A``.

        - If ``A`` is distributed along the columns (A.split = 1), so will be ``Q`` and ``R``.

        - If ``A`` is distributed along the rows (A.split = 0), ``Q`` too will have  `split=0`. ``R`` won't be distributed, i.e. `R. split = None`, if ``A`` is tall-skinny, i.e., if
          the largest local chunk of data of ``A`` has at least as many rows as columns. Otherwise, ``R`` will be distributed along the rows as well, i.e., `R.split = 0`.

    Note that the argument `calc_q` allowed in earlier Heat versions is no longer supported; `calc_q = False` is equivalent to `mode = "r"`.
    Unlike ``numpy.linalg.qr()``, `ht.linalg.qr` only supports ``mode="reduced"`` or ``mode="r"`` for the moment, since "complete" may result in heavy memory usage.

    Heats QR function is built on top of PyTorchs QR function, ``torch.linalg.qr()``, using LAPACK (CPU) and MAGMA (CUDA) on
    the backend. Both cases split=0 and split=1 build on a column-block-wise version of stabilized Gram-Schmidt orthogonalization.
    For split=1 (-1, in the batched case), this is directly applied to the local arrays of the input array.
    For split=0, a tall-skinny QR (TS-QR) is implemented for the case of tall-skinny matrices (i.e., the largest local chunk of data has at least as many rows as columns),
    and extended to non tall-skinny matrices by applying a block-wise version of stabilized Gram-Schmidt orthogonalization.

    References
    ----------
    Basic information about QR factorization/decomposition can be found at, e.g.:

        - https://en.wikipedia.org/wiki/QR_factorization,

        - Gene H. Golub and Charles F. Van Loan. 1996. Matrix Computations (3rd Ed.).

    For an extensive overview on TS-QR and its variants we refer to, e.g.,

        - Demmel, James, et al. “Communication-Optimal Parallel and Sequential QR and LU Factorizations.” SIAM Journal on Scientific Computing, vol. 34, no. 1, 2 Feb. 2012, pp. A206–A239., doi:10.1137/080731992.
    """
    if not isinstance(A, DNDarray):
        raise TypeError(f"'A' must be a DNDarray, but is {type(A)}")
    if not isinstance(mode, str):
        raise TypeError(f"'mode' must be a str, but is {type(mode)}")
    if mode not in ["reduced", "r"]:
        if mode == "complete":
            raise NotImplementedError(
                "QR decomposition with 'mode'='complete' is not supported by heat yet. \n Please open an issue on GitHub if you require this feature. \n For now, you can use 'mode'='reduced' or 'r' instead."
            )
        elif mode == "raw":
            raise NotImplementedError(
                "QR decomposition with 'mode'='raw' is neither supported by Heat nor by PyTorch. \n"
            )
        else:
            raise ValueError(f"'mode' must be 'reduced' (default) or 'r', but is {mode}")
    if not isinstance(procs_to_merge, int):
        raise TypeError(f"procs_to_merge must be an int, but is currently {type(procs_to_merge)}")
    if procs_to_merge < 0 or procs_to_merge == 1:
        raise ValueError(
            f"procs_to_merge must be 0 (for merging all processes at once) or at least 2, but is currently {procs_to_merge}"
        )
    if procs_to_merge == 0:
        procs_to_merge = A.comm.size

    if A.dtype not in [float32, float64]:
        raise TypeError(f"Array 'A' must have a datatype of float32 or float64, but has {A.dtype}")

    QR = collections.namedtuple("QR", "Q, R")

    if A.ndim == 3:
        single_proc_qr = torch.vmap(torch.linalg.qr, in_dims=0, out_dims=0)
    else:
        single_proc_qr = torch.linalg.qr

    if not A.is_distributed() or A.split < A.ndim - 2:
        # handle the case of a single process or split=None: just PyTorch QR
        Q, R = single_proc_qr(A.larray, mode=mode)
        R = factories.array(R, is_split=A.split)
        if mode == "reduced":
            Q = factories.array(Q, is_split=A.split, device=A.device)
        else:
            Q = None
        return QR(Q, R)

    if A.split == A.ndim - 1:
        # handle the case that A is split along the columns
        # here, we apply a block-wise version of (stabilized) Gram-Schmidt orthogonalization
        # instead of orthogonalizing each column of A individually, we orthogonalize blocks of columns (i.e. the local arrays) at once

        lshapes = A.lshape_map[:, -1]
        lshapes_cum = torch.cumsum(lshapes, 0)
        nprocs = A.comm.size

        if A.shape[-2] >= A.shape[-1]:
            last_row_reached = nprocs
            k = A.shape[-1]
        else:
            last_row_reached = min(torch.argwhere(lshapes_cum >= A.shape[-2]))[0]
            k = A.shape[-2]

        if mode == "reduced":
            Q = factories.zeros(
                A.shape, dtype=A.dtype, split=A.ndim - 1, device=A.device, comm=A.comm
            )

        R = factories.zeros(
            (*A.shape[:-2], k, A.shape[-1]),
            dtype=A.dtype,
            split=A.ndim - 1,
            device=A.device,
            comm=A.comm,
        )
        R_shapes = torch.hstack(
            [
                torch.zeros(1, dtype=torch.int32, device=A.device.torch_device),
                torch.cumsum(R.lshape_map[:, -1], 0),
            ]
        )

        A_columns = A.larray.clone()

        for i in range(last_row_reached + 1):
            # this loop goes through all the column-blocks (i.e. local arrays) of the matrix
            # this corresponds to the loop over all columns in classical Gram-Schmidt

            if i < nprocs - 1:
                k_loc_i = min(A.shape[-2], A.lshape_map[i, -1])
                Q_buf = torch.zeros(
                    (*A.shape[:-1], k_loc_i),
                    dtype=A.larray.dtype,
                    device=A.device.torch_device,
                )

            if A.comm.rank == i:
                # orthogonalize the current block of columns by utilizing PyTorch QR
                Q_curr, R_loc = single_proc_qr(A_columns, mode="reduced")
                if i < nprocs - 1:
                    Q_buf = Q_curr.contiguous()
                if mode == "reduced":
                    Q.larray = Q_curr
                r_size = R.larray[..., R_shapes[i] : R_shapes[i + 1], :].shape[-2]
                R.larray[..., R_shapes[i] : R_shapes[i + 1], :] = R_loc[..., :r_size, :]

            if i < nprocs - 1:
                # broadcast the orthogonalized block of columns to all other processes
                A.comm.Bcast(Q_buf, root=i)

            if A.comm.rank > i:
                # subtract the contribution of the current block of columns from the remaining columns
                R_loc = torch.transpose(Q_buf, -2, -1) @ A_columns
                A_columns -= Q_buf @ R_loc
                r_size = R.larray[..., R_shapes[i] : R_shapes[i + 1], :].shape[-2]
                R.larray[..., R_shapes[i] : R_shapes[i + 1], :] = R_loc[..., :r_size, :]

        if mode == "reduced":
            Q = Q[..., :, :k].balance()
        else:
            Q = None

        return QR(Q, R)

    if A.split == A.ndim - 2:
        # check that data distribution is reasonable for TS-QR
        # we regard a matrix with split = 0 as suitable for TS-QR if its largest local chunk of data has at least as many rows as columns
        biggest_number_of_local_rows = A.lshape_map[:, -2].max().item()
        if biggest_number_of_local_rows < A.shape[-1]:
            column_idx = torch.cumsum(A.lshape_map[:, -2], 0)
            column_idx = column_idx[column_idx < A.shape[-1]]
            column_idx = torch.cat(
                (
                    torch.tensor([0], device=column_idx.device),
                    column_idx,
                    torch.tensor([A.shape[-1]], device=column_idx.device),
                )
            )
            A_copy = A.copy()
            R = A.copy()
            # Block-wise Gram-Schmidt orthogonalization, applied to groups of columns
            offset = 1 if A.shape[-1] <= A.shape[-2] else 2
            for k in range(len(column_idx) - offset):
                # since we only consider a group of columns, TS QR is applied to a tall-skinny matrix
                Qnew, Rnew = qr(
                    A_copy[..., :, column_idx[k] : column_idx[k + 1]],
                    mode="reduced",
                    procs_to_merge=procs_to_merge,
                )

                # usual update of the remaining columns
                if R.comm.rank == k:
                    R.larray[
                        ...,
                        : (column_idx[k + 1] - column_idx[k]),
                        column_idx[k] : column_idx[k + 1],
                    ] = Rnew.larray
                if R.comm.rank > k:
                    R.larray[..., :, column_idx[k] : column_idx[k + 1]] *= 0
                if k < len(column_idx) - 2:
                    coeffs = (
                        torch.transpose(Qnew.larray, -2, -1)
                        @ A_copy.larray[..., :, column_idx[k + 1] :]
                    )
                    R.comm.Allreduce(communication.MPI.IN_PLACE, coeffs)
                    if R.comm.rank == k:
                        R.larray[..., :, column_idx[k + 1] :] = coeffs
                    A_copy.larray[..., :, column_idx[k + 1] :] -= Qnew.larray @ coeffs
                if mode == "reduced":
                    Q = Qnew if k == 0 else concatenate((Q, Qnew), axis=-1)
            if A.shape[-1] < A.shape[-2]:
                R = R[..., : A.shape[-1], :].balance()
            if mode == "reduced":
                return QR(Q, R)
            else:
                return QR(None, R)

        else:
            # in this case the input is tall-skinny and we apply the TS-QR algorithm
            # it follows the implementation of TS-QR for split = 0
            current_procs = [i for i in range(A.comm.size)]
            current_comm = A.comm
            local_comm = current_comm.Split(current_comm.rank // procs_to_merge, A.comm.rank)
            Q_loc, R_loc = single_proc_qr(A.larray, mode=mode)
            R_loc = R_loc.contiguous()
            if mode == "reduced":
                leave_comm = current_comm.Split(current_comm.rank, A.comm.rank)

            level = 1
            while len(current_procs) > 1:
                if A.comm.rank in current_procs and local_comm.size > 1:
                    # create array to collect the R_loc's from all processes of the process group of at most n_procs_to_merge processes
                    shapes_R_loc = local_comm.gather(R_loc.shape[-2], root=0)
                    if local_comm.rank == 0:
                        gathered_R_loc = torch.zeros(
                            (*R_loc.shape[:-2], sum(shapes_R_loc), R_loc.shape[-1]),
                            device=R_loc.device,
                            dtype=R_loc.dtype,
                        )
                        counts = list(shapes_R_loc)
                        displs = torch.cumsum(
                            torch.tensor([0] + shapes_R_loc, dtype=torch.int32), 0
                        ).tolist()[:-1]
                    else:
                        gathered_R_loc = torch.empty(0, device=R_loc.device, dtype=R_loc.dtype)
                        counts = None
                        displs = None
                    # gather the R_loc's from all processes of the process group of at most n_procs_to_merge processes
                    local_comm.Gatherv(R_loc, (gathered_R_loc, counts, displs), root=0, axis=-2)
                    # perform QR decomposition on the concatenated, gathered R_loc's to obtain new R_loc
                    if local_comm.rank == 0:
                        previous_shape = R_loc.shape
                        Q_buf, R_loc = single_proc_qr(gathered_R_loc, mode=mode)
                        R_loc = R_loc.contiguous()
                    else:
                        Q_buf = torch.empty(0, device=R_loc.device, dtype=R_loc.dtype)
                    if mode == "reduced":
                        if local_comm.rank == 0:
                            Q_buf = Q_buf.contiguous()
                        scattered_Q_buf = torch.empty(
                            R_loc.shape if local_comm.rank != 0 else previous_shape,
                            device=R_loc.device,
                            dtype=R_loc.dtype,
                        )
                        # scatter the Q_buf to all processes of the process group
                        local_comm.Scatterv(
                            (Q_buf, counts, displs), scattered_Q_buf, root=0, axis=-2
                        )
                    del gathered_R_loc, Q_buf

                # for each process in the current processes, broadcast the scattered_Q_buf of this process
                # to all leaves (i.e. all original processes that merge to the current process)
                if mode == "reduced" and leave_comm.size > 1:
                    try:
                        scattered_Q_buf_shape = scattered_Q_buf.shape
                    except UnboundLocalError:
                        scattered_Q_buf_shape = None
                    scattered_Q_buf_shape = leave_comm.bcast(scattered_Q_buf_shape, root=0)
                    if scattered_Q_buf_shape is not None:
                        # this is needed to ensure that only those Q_loc get updates that are actually part of the current process group
                        if leave_comm.rank != 0:
                            scattered_Q_buf = torch.empty(
                                scattered_Q_buf_shape, device=Q_loc.device, dtype=Q_loc.dtype
                            )
                        leave_comm.Bcast(scattered_Q_buf, root=0)
                    # update the local Q_loc by multiplying it with the scattered_Q_buf
                try:
                    Q_loc = Q_loc @ scattered_Q_buf
                    del scattered_Q_buf
                except UnboundLocalError:
                    pass

                # update: determine processes to be active at next "merging" level, create new communicator and split it into groups for gathering
                current_procs = [
                    current_procs[i] for i in range(len(current_procs)) if i % procs_to_merge == 0
                ]
                if len(current_procs) > 1:
                    new_group = A.comm.group.Incl(current_procs)
                    current_comm = A.comm.Create_group(new_group)
                    if A.comm.rank in current_procs:
                        local_comm = communication.MPICommunication(
                            current_comm.Split(current_comm.rank // procs_to_merge, A.comm.rank)
                        )
                    if mode == "reduced":
                        leave_comm = A.comm.Split(A.comm.rank // procs_to_merge**level, A.comm.rank)
                level += 1
            # broadcast the final R_loc to all processes
            R_gshape = (*A.shape[:-2], A.shape[-1], A.shape[-1])
            if A.comm.rank != 0:
                R_loc = torch.empty(R_gshape, dtype=R_loc.dtype, device=R_loc.device)
            A.comm.Bcast(R_loc, root=0)
            R = DNDarray(
                R_loc,
                gshape=R_gshape,
                dtype=A.dtype,
                split=None,
                device=A.device,
                comm=A.comm,
                balanced=True,
            )
            if mode == "r":
                Q = None
            else:
                Q = DNDarray(
                    Q_loc,
                    gshape=A.shape,
                    dtype=A.dtype,
                    split=A.split,
                    device=A.device,
                    comm=A.comm,
                    balanced=True,
                )
            return QR(Q, R)
