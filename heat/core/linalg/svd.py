import torch

from .qr import __split0_r_calc, __split0_q_loop, __split1_qr_loop
from .. import dndarray
from .. import factories
from .. import tiling
from . import utils

__all__ = ["block_diagonalize", "bulge_chasing"]


def block_diagonalize(arr, overwrite_arr=False, return_tiles=False, balance=True):
    """
    Create an upper block diagonal matrix with both left and right transformation matrices.
    This is done by doing QR factorization on `arr` in a standard fashion while interweaving
    LQ factorization on `arr.T` on the neighboring tile, i.e. the next tile column.

    If a matrix is too small to benefit from tiling, an error will be raised.

    This function is typically an intermediate step in other functions and it is not viewed
    as necessary to implement a version for the non-distributed case.

    This function will return 3 matrices, `U`, `B`, and `V`. The three of which combine to equal `arr`,
    `arr == U @ B @ V`

    Parameters
    ----------
    arr : ht.DNDarray
        2D input matrix (m x n)
    overwrite_arr : bool, Optional
        Default: False
        if True, will overwrite arr with the result
    return_tiles : bool, Optional
        Default: False
        if True, will return the tiling objects as well as the final matrices
        NOTE: if this is True and balance if False, the tiles will not be usable!
    balance : bool, Optional
        Default: True
        If True, will balance the output matrices

    Returns
    -------
    results : Tuple
        U (q0) : DNDarray
            Left transformation matrix (m x m)
        B (arr) : DNDarray
            block diagonal matrix (m x n)
        V (q1) : DNDarray
            right transormation matrix (n x n)
        tiles : Dict
            Dictionary with the entries of the tiling objects, keys are as follows
            `{"q0": q0_tiles, "arr": arr_tiles, "arr_t": arr_t_tiles, "q1": q1_tiles}`

    References
    ----------
    [0] A. Haidar, H. Ltaief, P. Luszczek and J. Dongarra, "A Comprehensive Study of Task Coalescing for Selecting
        Parallelism Granularity in a Two-Stage Bidiagonal Reduction," 2012 IEEE 26th International Parallel and
        Distributed Processing Symposium, Shanghai, 2012, pp. 25-35, doi: 10.1109/IPDPS.2012.13.
    """
    if not isinstance(arr, dndarray.DNDarray):
        raise TypeError("arr must be a DNDarray, not {}".format(type(arr)))
    if not arr.is_distributed():
        raise RuntimeError("Array must be distributed, see docs")
    if arr.split not in [0, 1]:
        raise NotImplementedError(
            "Split {} not implemented, arr must be 2D and split".format(arr.split)
        )
    tiles_per_proc = 2
    # 1. tile arr if needed
    if not overwrite_arr:
        arr = arr.copy()
    arr_tiles = tiling.SquareDiagTiles(arr, tiles_per_proc)

    q1 = factories.eye(
        (arr.gshape[1], arr.gshape[1]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    if arr.split == 0 and (
        arr_tiles.lshape_map[0, 0] >= arr.gshape[1]
        or min(arr.gshape) // (arr.comm.size * tiles_per_proc) <= 1
    ):
        # in this case forming a band diagonal by interlacing QR and LQ doesn work so its just QR
        qr = arr.qr(tiles_per_proc=tiles_per_proc)
        return qr.Q, qr.R, q1.T
    # 2. get transpose of arr
    arr_t = arr.T
    # 3. tile arr_t
    arr_t_tiles = tiling.SquareDiagTiles(arr_t, tiles_per_proc, no_tiles=True)

    # 4. match tiles to arr
    arr_t_tiles.match_tiles_qr_lq(arr_tiles)
    arr_t_tiles.set_arr(arr.T)

    q0 = factories.eye(
        (arr.gshape[0], arr.gshape[0]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q0_tiles = tiling.SquareDiagTiles(q0, tiles_per_proc)
    q0_tiles.match_tiles(arr_tiles)
    q1_tiles = tiling.SquareDiagTiles(q1, tiles_per_proc)
    q1_tiles.match_tiles(arr_t_tiles)

    if arr.split == 0:
        return __block_diagonalize_sp0(
            arr_tiles, arr_t_tiles, q0_tiles, q1_tiles, balance=balance, ret_tiles=return_tiles
        )
    else:
        return __block_diagonalize_sp1(
            arr_tiles, arr_t_tiles, q0_tiles, q1_tiles, balance=balance, ret_tiles=return_tiles
        )


def __block_diagonalize_sp0(
    arr_tiles, arr_t_tiles, q0_tiles, q1_tiles, balance=True, ret_tiles=False
):
    tile_columns = arr_tiles.tile_columns

    torch_device = arr_tiles.arr.device.torch_device
    q0_dict = {}
    q0_dict_waits = {}

    active_procs = torch.arange(arr_tiles.arr.comm.size)
    active_procs_t = active_procs.clone().detach()
    empties = torch.nonzero(
        input=arr_tiles.lshape_map[..., arr_tiles.arr.split] == 0, as_tuple=False
    )
    empties = empties[0] if empties.numel() > 0 else []
    for e in empties:
        active_procs = active_procs[active_procs != e]
    tile_rows_per_pr_trmd = arr_tiles.tile_rows_per_process[: active_procs[-1] + 1]

    empties_t = torch.nonzero(
        input=arr_t_tiles.lshape_map[..., arr_t_tiles.arr.split] == 0, as_tuple=False
    )
    empties_t = empties_t[0] if empties_t.numel() > 0 else []
    for e in empties_t:
        active_procs_t = active_procs_t[active_procs_t != e]

    proc_tile_start = torch.cumsum(torch.tensor(tile_rows_per_pr_trmd, device=torch_device), dim=0)

    # looping over number of tile columns - 1 (col)
    # 1. do QR on arr for column=col (standard QR as written)
    # 2. do LQ on arr_t for column=col+1 (standard QR again, the transpose makes it LQ)
    #       both of these steps overwrite arr (or an initial copy of it, optional)
    rank = arr_tiles.arr.comm.rank
    for col in range(tile_columns - 1):
        # 1. do QR on arr for column=col (standard QR as written) (this assumes split == 0)
        not_completed_processes = torch.nonzero(
            input=col < proc_tile_start, as_tuple=False
        ).flatten()
        diag_process = not_completed_processes[0].item()
        if rank in not_completed_processes and rank in active_procs:
            # if the process is done calculating R the break the loop
            __split0_r_calc(
                r_tiles=arr_tiles,
                q_dict=q0_dict,
                q_dict_waits=q0_dict_waits,
                dim1=col,
                diag_pr=diag_process,
                not_completed_prs=not_completed_processes,
            )
        # 2. do full QR on the next column for LQ on arr_t
        __split1_qr_loop(
            dim1=col,
            r_tiles=arr_t_tiles,
            q0_tiles=q1_tiles,
            calc_q=True,
            dim0=col + 1,
            empties=empties_t,
        )
    for col in range(tile_columns - 1):
        __split0_q_loop(
            r_tiles=arr_tiles,
            q0_tiles=q0_tiles,
            dim1=col,
            proc_tile_start=proc_tile_start,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            active_procs=active_procs,
        )

    # do the last column now
    col = tile_columns - 1
    not_completed_processes = torch.nonzero(input=col < proc_tile_start, as_tuple=False).flatten()
    diag_process = not_completed_processes[0].item()
    if rank in not_completed_processes and rank in active_procs:
        __split0_r_calc(
            r_tiles=arr_tiles,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            dim1=col,
            diag_pr=diag_process,
            not_completed_prs=not_completed_processes,
        )
    __split0_q_loop(
        r_tiles=arr_tiles,
        q0_tiles=q0_tiles,
        dim1=col,
        proc_tile_start=proc_tile_start,
        q_dict=q0_dict,
        q_dict_waits=q0_dict_waits,
        active_procs=active_procs,
    )

    diag_diff = arr_t_tiles.row_indices[1]
    if arr_tiles.arr.gshape[0] < arr_tiles.arr.gshape[1] - diag_diff:
        __split1_qr_loop(
            dim1=col,
            r_tiles=arr_t_tiles,
            q0_tiles=q1_tiles,
            calc_q=True,
            dim0=col + 1,
            empties=empties_t,
        )

    if balance:
        arr_tiles.arr.balance_()
        q0_tiles.arr.balance_()
        q1_tiles.arr.balance_()
    ret = [q0_tiles.arr, arr_tiles.arr, q1_tiles.arr.T]
    if ret_tiles:
        tiles = {"q0": q0_tiles, "arr": arr_tiles, "arr_t": arr_t_tiles, "q1": q1_tiles}
        ret.append(tiles)
    return ret


def __block_diagonalize_sp1(
    arr_tiles, arr_t_tiles, q0_tiles, q1_tiles, balance=True, ret_tiles=False
):
    # -------------------------- split = 1 stuff (att) ---------------------------------------------
    tile_columns = arr_tiles.tile_columns
    tile_rows = arr_tiles.tile_rows

    torch_device = arr_tiles.arr.device.torch_device

    active_procs = torch.arange(arr_tiles.arr.comm.size)
    empties = torch.nonzero(
        input=arr_tiles.lshape_map[..., arr_tiles.arr.split] == 0, as_tuple=False
    )
    empties = empties[0] if empties.numel() > 0 else []
    for e in empties:
        active_procs = active_procs[active_procs != e]
    # -------------------------- split = 0 stuff (arr_t) -------------------------------------------
    active_procs_t = torch.arange(arr_t_tiles.arr.comm.size)
    empties_t = torch.nonzero(input=arr_t_tiles.lshape_map[..., 0] == 0, as_tuple=False)
    empties_t = empties_t[0] if empties_t.numel() > 0 else []
    for e in empties_t:
        active_procs_t = active_procs_t[active_procs_t != e]
    tile_rows_per_pr_trmd_t = arr_t_tiles.tile_rows_per_process[: active_procs_t[-1] + 1]

    q1_dict = {}
    q1_dict_waits = {}
    proc_tile_start_t = torch.cumsum(
        torch.tensor(tile_rows_per_pr_trmd_t, device=torch_device), dim=0
    )
    # ----------------------------------------------------------------------------------------------
    # looping over number of tile columns - 1 (col)
    # 1. do QR on arr for column=col (standard QR as written)
    # 2. do LQ on arr_t for column=col+1 (standard QR again, the transpose makes it LQ)
    #       both of these steps overwrite arr (or an initial copy of it, optional)
    rank = arr_tiles.arr.comm.rank
    lp_cols = tile_columns if arr_tiles.arr.gshape[0] > arr_tiles.arr.gshape[1] else tile_rows
    for col in range(lp_cols - 1):
        # 1. QR (split = 1) on col
        __split1_qr_loop(
            dim1=col, r_tiles=arr_tiles, q0_tiles=q0_tiles, calc_q=True, dim0=col, empties=empties
        )
        # 2. QR (split = 0) on col + 1
        not_completed_processes = torch.nonzero(
            input=col + 1 < proc_tile_start_t, as_tuple=False
        ).flatten()
        if rank in not_completed_processes and rank in active_procs_t:
            diag_process = not_completed_processes[0].item()
            __split0_r_calc(
                r_tiles=arr_t_tiles,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                dim1=col,
                diag_pr=diag_process,
                not_completed_prs=not_completed_processes,
                dim0=col + 1,
            )

        __split0_q_loop(
            r_tiles=arr_t_tiles,
            q0_tiles=q1_tiles,
            dim1=col,
            proc_tile_start=proc_tile_start_t,
            q_dict=q1_dict,
            q_dict_waits=q1_dict_waits,
            active_procs=active_procs_t,
            dim0=col + 1,
        )
    # do the last column
    col = lp_cols - 1
    __split1_qr_loop(
        dim1=col, r_tiles=arr_tiles, q0_tiles=q0_tiles, calc_q=True, dim0=col, empties=empties
    )

    if arr_tiles.arr.gshape[0] < arr_tiles.arr.gshape[1]:
        # if m < n then need to do another round of LQ.
        not_completed_processes = torch.nonzero(
            input=col + 1 < proc_tile_start_t, as_tuple=False
        ).flatten()
        if rank in not_completed_processes and rank in active_procs_t:
            diag_process = not_completed_processes[0].item()
            __split0_r_calc(
                r_tiles=arr_t_tiles,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                dim1=col,
                diag_pr=diag_process,
                not_completed_prs=not_completed_processes,
                dim0=col + 1,
            )

        if len(not_completed_processes) > 0:
            __split0_q_loop(
                r_tiles=arr_t_tiles,
                q0_tiles=q1_tiles,
                dim1=col,
                proc_tile_start=proc_tile_start_t,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                active_procs=active_procs_t,
                dim0=col + 1,
            )

    if balance:
        arr_tiles.arr.balance_()
        q0_tiles.arr.balance_()
        q1_tiles.arr.balance_()

    if ret_tiles:
        tiles = {"q0": q0_tiles, "arr": arr_tiles, "arr_t": arr_t_tiles, "q1": q1_tiles}
        return q0_tiles.arr, arr_tiles.arr, q1_tiles.arr.T, tiles
    else:
        return q0_tiles.arr, arr_tiles.arr, q1_tiles.arr.T


def bulge_chasing(arr, q0, q1, tiles=None):
    # assuming that the matrix is upper band diagonal
    # need the tile shape first
    rank = arr.comm.rank
    # size = arr.comm.size
    if tiles is not None:
        band_width = tiles["arr"].col_indices[1] + 1
    else:
        band_width = arr.gshape[1]

    lshape_map = tiles["arr"].lshape_map
    splits = lshape_map[..., arr.split].cumsum(0)
    # need to start on the first process and continue onward...
    # todo: should ldp come from arr_tiles or arr_t_tiles (depends on the lshape)
    ldp = tiles["arr"].last_diagonal_process
    active_procs = list(range(ldp + 1))
    lcl_array = arr._DNDarray__array
    # lcl_right_apply = {}
    # lcl_left_apply = {}
    if rank <= ldp:
        # todo: each column or row (smallest)
        ncols = min(arr.gshape) - 1 if arr.gshape[0] >= arr.gshape[1] else min(arr.gshape) + 1
        for col in range(ncols):
            if any(col >= splits[active_procs[0] :]):
                if rank == active_procs[0]:
                    break
                del active_procs[0]
            # get the first row
            if rank == active_procs[0]:
                # generate the vector to eliminate the row
                v, t = utils.gen_house_vec(lcl_array[col, col + 1 : col + band_width])
                # lcl_right_apply[(col, col + band_width)]
                print(col, v)
