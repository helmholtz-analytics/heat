"""
QR decomposition of (distributed) 2-D ``DNDarray``s.
"""

import collections
import torch
from typing import Tuple

from ..dndarray import DNDarray
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

    Parameters
    ----------
    A : DNDarray of shape (M, N)
        Array which will be decomposed. So far only 2D arrays with datatype float32 or float64 are supported
        For split=0, the matrix must be tall skinny, i.e. the local chunks of data must have at least as many rows as columns.
    mode : str, optional
        default "reduced" returns Q and R with dimensions (M, min(M,N)) and (min(M,N), N), respectively.
        "r" returns only R, with dimensions (min(M,N), N).
    procs_to_merge : int, optional
        This parameter is only relevant for split=0 and determines the number of processes to be merged at one step during the so-called TS-QR algorithm.
        The default is 2. Higher choices might be faster, but will probably result in higher memory consumption. 0 corresponds to merging all processes at once.
        We only recommend to modify this parameter if you are familiar with the TS-QR algorithm (see the references below).

    Notes
    -----
    The distribution schemes of ``Q`` and ``R`` depend on that of the input ``A``.

        - If ``A`` is distributed along the columns (A.split = 1), so will be ``Q`` and ``R``.

        - If ``A`` is distributed along the rows (A.split = 0), ``Q`` too will have  `split=0`, but ``R`` won't be distributed, i.e. `R. split = None` and a full copy of ``R`` will be stored on each process.

    Note that the argument `calc_q` allowed in earlier Heat versions is no longer supported; `calc_q = False` is equivalent to `mode = "r"`.
    Unlike ``numpy.linalg.qr()``, `ht.linalg.qr` only supports ``mode="reduced"`` or ``mode="r"`` for the moment, since "complete" may result in heavy memory usage.

    Heats QR function is built on top of PyTorchs QR function, ``torch.linalg.qr()``, using LAPACK (CPU) and MAGMA (CUDA) on
    the backend. For split=0, tall-skinny QR (TS-QR) is implemented, while for split=1 a block-wise version of stabilized Gram-Schmidt orthogonalization is used.

    References
    -----------
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

    if A.ndim != 2:
        raise ValueError(
            f"Array 'A' must be 2 dimensional, buts has {A.ndim} dimensions. \n Please open an issue on GitHub if you require QR for batches of matrices similar to PyTorch."
        )
    if A.dtype not in [float32, float64]:
        raise TypeError(f"Array 'A' must have a datatype of float32 or float64, but has {A.dtype}")

    QR = collections.namedtuple("QR", "Q, R")

    if not A.is_distributed():
        # handle the case of a single process or split=None: just PyTorch QR
        Q, R = torch.linalg.qr(A.larray, mode=mode)
        R = DNDarray(
            R,
            gshape=R.shape,
            dtype=A.dtype,
            split=A.split,
            device=A.device,
            comm=A.comm,
            balanced=True,
        )
        if mode == "reduced":
            Q = DNDarray(
                Q,
                gshape=Q.shape,
                dtype=A.dtype,
                split=A.split,
                device=A.device,
                comm=A.comm,
                balanced=True,
            )
        else:
            Q = None
        return QR(Q, R)

    if A.split == 1:
        # handle the case that A is split along the columns
        # here, we apply a block-wise version of (stabilized) Gram-Schmidt orthogonalization
        # instead of orthogonalizing each column of A individually, we orthogonalize blocks of columns (i.e. the local arrays) at once

        lshapes = A.lshape_map[:, 1]
        lshapes_cum = torch.cumsum(lshapes, 0)
        nprocs = A.comm.size

        if A.shape[0] >= A.shape[1]:
            last_row_reached = nprocs
            k = A.shape[1]
        else:
            last_row_reached = min(torch.argwhere(lshapes_cum >= A.shape[0]))[0]
            k = A.shape[0]

        if mode == "reduced":
            Q = factories.zeros(A.shape, dtype=A.dtype, split=1, device=A.device, comm=A.comm)

        R = factories.zeros((k, A.shape[1]), dtype=A.dtype, split=1, device=A.device, comm=A.comm)
        R_shapes = torch.hstack(
            [
                torch.zeros(1, dtype=torch.int32, device=A.device.torch_device),
                torch.cumsum(R.lshape_map[:, 1], 0),
            ]
        )

        A_columns = A.larray.clone()

        for i in range(last_row_reached + 1):
            # this loop goes through all the column-blocks (i.e. local arrays) of the matrix
            # this corresponds to the loop over all columns in classical Gram-Schmidt
            if i < nprocs - 1:
                k_loc_i = min(A.shape[0], A.lshape_map[i, 1])
                Q_buf = torch.zeros(
                    (A.shape[0], k_loc_i), dtype=A.larray.dtype, device=A.device.torch_device
                )

            if A.comm.rank == i:
                # orthogonalize the current block of columns by utilizing PyTorch QR
                Q_curr, R_loc = torch.linalg.qr(A_columns, mode="reduced")
                if i < nprocs - 1:
                    Q_buf = Q_curr
                if mode == "reduced":
                    Q.larray = Q_curr
                r_size = R.larray[R_shapes[i] : R_shapes[i + 1], :].shape[0]
                R.larray[R_shapes[i] : R_shapes[i + 1], :] = R_loc[:r_size, :]

            if i < nprocs - 1:
                # broadcast the orthogonalized block of columns to all other processes
                req = A.comm.Ibcast(Q_buf, root=i)
                req.Wait()

            if A.comm.rank > i:
                # subtract the contribution of the current block of columns from the remaining columns
                R_loc = Q_buf.T @ A_columns
                A_columns -= Q_buf @ R_loc
                r_size = R.larray[R_shapes[i] : R_shapes[i + 1], :].shape[0]
                R.larray[R_shapes[i] : R_shapes[i + 1], :] = R_loc[:r_size, :]

        if mode == "reduced":
            Q = Q[:, :k].balance()
        else:
            Q = None

        return QR(Q, R)

    if A.split == 0:
        # implementation of TS-QR for split = 0
        # check that data distribution is reasonable for TS-QR (i.e. tall-skinny matrix with also tall-skinny local chunks of data)
        if A.lshape_map[:, 0].max().item() < A.shape[1]:
            raise ValueError(
                "A is split along the rows and the local chunks of data are rectangular with more rows than columns. \n Applying TS-QR in this situation is not reasonable w.r.t. runtime and memory consumption. \n We recomment to split A along the columns instead. \n In case this is not an option for you, please open an issue on GitHub."
            )

        current_procs = [i for i in range(A.comm.size)]
        current_comm = A.comm
        local_comm = current_comm.Split(current_comm.rank // procs_to_merge, A.comm.rank)
        Q_loc, R_loc = torch.linalg.qr(A.larray, mode=mode)
        R_loc = R_loc.contiguous()  # required for all the communication ops lateron
        if mode == "reduced":
            leave_comm = current_comm.Split(current_comm.rank, A.comm.rank)

        level = 1
        while len(current_procs) > 1:
            if A.comm.rank in current_procs and local_comm.size > 1:
                # create array to collect the R_loc's from all processes of the process group of at most n_procs_to_merge processes
                shapes_R_loc = local_comm.gather(R_loc.shape[0], root=0)
                if local_comm.rank == 0:
                    gathered_R_loc = torch.zeros(
                        (sum(shapes_R_loc), R_loc.shape[1]),
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
                local_comm.Gatherv(R_loc, (gathered_R_loc, counts, displs), root=0, axis=0)
                # perform QR decomposition on the concatenated, gathered R_loc's to obtain new R_loc
                if local_comm.rank == 0:
                    previous_shape = R_loc.shape
                    Q_buf, R_loc = torch.linalg.qr(gathered_R_loc, mode=mode)
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
                    local_comm.Scatterv((Q_buf, counts, displs), scattered_Q_buf, root=0, axis=0)
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
        R_gshape = (A.shape[1], A.shape[1])
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
                split=0,
                device=A.device,
                comm=A.comm,
                balanced=True,
            )
        return QR(Q, R)
