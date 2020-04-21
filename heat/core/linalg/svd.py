import torch
from typing import Tuple

from .qr import __split0_r_calc, __split0_q_loop, __split1_qr_loop

from .. import factories
from .. import tiling

__all__ = ["block_diagonalize"]


def block_diagonalize(arr, overwrite_arr=False, return_tiles=False, balance=True):
    if arr.split == 0:
        return block_diagonalize_sp0(arr, overwrite_arr, balance=balance, ret_tiles=return_tiles)
    elif arr.split == 1:
        return block_diagonalize_sp1(arr, overwrite_arr, balance=balance, ret_tiles=return_tiles)
    else:
        raise NotImplementedError(
            "Split {} not implemented, arr must be 2D and split".format(arr.split)
        )


def block_diagonalize_sp0(arr, overwrite_arr=False, balance=True, ret_tiles=False):
    # no copies!
    # steps to get ready for loop:
    # 1. tile arr if needed
    # 2. get transpose of arr
    # 3. tile arr_t
    # 4. match tiles to arr
    # ----------------------------------------------------------------------------------------------

    tiles_per_proc = 1
    # 1. tile arr if needed
    if not overwrite_arr:
        arr = arr.copy()
    # if arr.tiles is None:
    #     arr.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    arr_tiles = tiling.SquareDiagTiles(arr, tiles_per_proc)
    q1 = factories.eye(
        (arr.gshape[1], arr.gshape[1]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )

    if arr_tiles.lshape_map[0, 0] >= arr.gshape[1]:
        # in this case forming a band diagonal by interlacing QR and LQ doesn work so its just QR
        qr = arr.qr(tiles_per_proc=tiles_per_proc)
        return qr.Q, qr.R, q1

    # 2. get transpose of arr
    arr_t = arr.T
    # 3. tile arr_t
    # arr_t.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    arr_t_tiles = tiling.SquareDiagTiles(arr_t, tiles_per_proc)

    # 4. match tiles to arr
    arr_t_tiles.match_tiles_qr_lq(arr_tiles)
    # print(arr_t_tiles.tile_map)

    arr_t_tiles.__DNDarray = arr.T

    q0 = factories.eye(
        (arr.gshape[0], arr.gshape[0]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q0_tiles = tiling.SquareDiagTiles(q0, tiles_per_proc)
    q0_tiles.match_tiles(arr_tiles)
    q1_tiles = tiling.SquareDiagTiles(q1, tiles_per_proc)
    q1_tiles.match_tiles(arr_t_tiles)
    # ----------------------------------------------------------------------------------------------
    tile_columns = arr_tiles.tile_columns

    torch_device = arr._DNDarray__array.device
    q0_dict = {}
    q0_dict_waits = {}

    active_procs = torch.arange(arr.comm.size)
    active_procs_t = active_procs.clone().detach()
    empties = torch.nonzero(arr_tiles.lshape_map[..., arr.split] == 0)
    empties = empties[0] if empties.numel() > 0 else []
    for e in empties:
        active_procs = active_procs[active_procs != e]
    tile_rows_per_pr_trmd = arr_tiles.tile_rows_per_process[: active_procs[-1] + 1]

    empties_t = torch.nonzero(arr_t_tiles.lshape_map[..., arr_t.split] == 0)
    empties_t = empties_t[0] if empties_t.numel() > 0 else []
    for e in empties_t:
        active_procs_t = active_procs_t[active_procs_t != e]

    proc_tile_start = torch.cumsum(torch.tensor(tile_rows_per_pr_trmd, device=torch_device), dim=0)

    # looping over number of tile columns - 1 (col)
    # 1. do QR on arr for column=col (standard QR as written)
    # 2. do LQ on arr_t for column=col+1 (standard QR again, the transpose makes it LQ)
    #       both of these steps overwrite arr (or an initial copy of it, optional)
    rank = arr.comm.rank
    for col in range(tile_columns - 1):
        # 1. do QR on arr for column=col (standard QR as written) (this assumes split == 0)
        not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
        diag_process = not_completed_processes[0].item()
        if rank in not_completed_processes and rank in active_procs:
            # if the process is done calculating R the break the loop
            # break
            __split0_r_calc(
                r_tiles=arr_tiles,
                q_dict=q0_dict,
                q_dict_waits=q0_dict_waits,
                dim1=col,
                diag_pr=diag_process,
                not_completed_prs=not_completed_processes,
            )
        # arr_t_tiles.set_arr(arr_tiles.arr.T)
        # 2. do full QR on the next column for LQ on arr_t
        # print(col + 1, col)
        __split1_qr_loop(
            dim1=col,
            r_tiles=arr_t_tiles,
            q0_tiles=q1_tiles,
            calc_q=True,
            dim0=col + 1,
            empties=empties_t,
        )
    # arr_t_tiles.set_arr(arr_tiles.arr.T)
    for col in range(tile_columns - 1):
        # not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
        # diag_process = not_completed_processes[0].item()
        __split0_q_loop(
            r_tiles=arr_tiles,
            q0_tiles=q0_tiles,
            dim1=col,
            proc_tile_start=proc_tile_start,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            active_procs=active_procs,
        )
    # arr_tiles.set_arr(arr_t_tiles.arr.T)

    # do the last column now
    col = tile_columns - 1
    not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
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
    arr_t_tiles.set_arr(arr_tiles.arr.T)
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
    if arr.gshape[0] < arr.gshape[1] - diag_diff:
        __split1_qr_loop(
            dim1=col,
            r_tiles=arr_t_tiles,
            q0_tiles=q1_tiles,
            calc_q=True,
            dim0=col + 1,
            empties=empties_t,
        )
    # arr_tiles.set_arr(arr_t_tiles.arr.T)

    # print(arr)
    # q1 = q1.T
    # q1.balance_()
    # arr.balance_()
    if balance:
        arr_tiles.arr.balance_()
        arr_t_tiles.arr.balance_()
        q0_tiles.arr.balance_()
        q1_tiles.arr.balance_()
    # print(arr_t_tiles.arr.T - arr_tiles.arr)
    ret = [q0_tiles.arr, arr_tiles.arr, q1_tiles.arr.T]
    if ret_tiles:
        tiles = {"q0": q0_tiles, "arr": arr_tiles, "arr_t": arr_t_tiles, "q1": q1_tiles}
        ret.append(tiles)
    return ret


def block_diagonalize_sp1(arr, overwrite_arr=False, balance=True, ret_tiles=False):
    tiles_per_proc = 1
    # no copies!
    # steps to get ready for loop:
    # 1. tile arr if needed
    # 2. get transpose of arr
    # 3. tile arr_t
    # 4. match tiles to arr
    # ----------------------------------------------------------------------------------------------
    # 1. tile arr if needed
    if not overwrite_arr:
        arr = arr.copy()
    arr_tiles = tiling.SquareDiagTiles(arr, tiles_per_proc=tiles_per_proc)
    # 2. get transpose of arr
    arr_t = arr.T.copy()
    # 3. tile arr_t
    arr_t_tiles = tiling.SquareDiagTiles(arr_t, tiles_per_proc=tiles_per_proc)
    # 4. match tiles to arr
    arr_t_tiles.match_tiles_qr_lq(arr_tiles)

    q0 = factories.eye(
        (arr.gshape[0], arr.gshape[0]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    # q0.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q0_tiles = tiling.SquareDiagTiles(q0, tiles_per_proc=tiles_per_proc)
    q0_tiles.match_tiles(arr_tiles)

    q1 = factories.eye(
        (arr.gshape[1], arr.gshape[1]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    # q1.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q1_tiles = tiling.SquareDiagTiles(q1, tiles_per_proc=tiles_per_proc)
    q1_tiles.match_tiles(arr_t_tiles)
    # -------------------------- split = 1 stuff (att) ---------------------------------------------
    tile_columns = arr_tiles.tile_columns
    tile_rows = arr_tiles.tile_rows

    torch_device = arr._DNDarray__array.device

    active_procs = torch.arange(arr.comm.size)
    empties = torch.nonzero(arr_tiles.lshape_map[..., arr.split] == 0)
    empties = empties[0] if empties.numel() > 0 else []
    for e in empties:
        active_procs = active_procs[active_procs != e]

    # proc_tile_start = torch.cumsum(
    #     torch.tensor(arr_tiles.tile_columns_per_process, device=torch_device), dim=0
    # )
    # -------------------------- split = 0 stuff (arr_t) -------------------------------------------
    active_procs_t = torch.arange(arr_t.comm.size)
    empties_t = torch.nonzero(arr_t_tiles.lshape_map[..., 0] == 0)
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
    rank = arr.comm.rank
    lp_cols = tile_columns if arr.gshape[0] > arr.gshape[1] else tile_rows
    for col in range(lp_cols - 1):
        # 1. QR (split = 1) on col
        # 2. QR (split = 0) on col + 1
        __split1_qr_loop(
            dim1=col, r_tiles=arr_tiles, q0_tiles=q0_tiles, calc_q=True, dim0=col, empties=empties
        )
        # __split1_qr_loop(dim1=col, r=arr, q0=q0, calc_q=True, empties=empties)

        arr_t_tiles.set_arr(arr_tiles.arr.T)

        not_completed_processes = torch.nonzero(col + 1 < proc_tile_start_t).flatten()
        # print(not_completed_processes, active_procs_t)
        diag_process = not_completed_processes[0].item()
        if rank in not_completed_processes and rank in active_procs_t:
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
        arr_tiles.set_arr(arr_t_tiles.arr.T)
    # do the last column now
    col = lp_cols - 1
    __split1_qr_loop(
        dim1=col, r_tiles=arr_tiles, q0_tiles=q0_tiles, calc_q=True, dim0=col, empties=empties
    )
    # __split1_qr_loop(dim1=col, r=arr, q0=q0, calc_q=True, empties=empties)

    arr_t_tiles.set_arr(arr.T)
    if arr.gshape[0] < arr.gshape[1]:
        # if m < n then need to do another round of LQ
        not_completed_processes = torch.nonzero(col + 1 < proc_tile_start_t).flatten()
        diag_process = not_completed_processes[0].item()
        if rank in not_completed_processes and rank in active_procs_t:
            __split0_r_calc(
                r_tiles=arr_t_tiles,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                dim1=col,
                diag_pr=diag_process,
                not_completed_prs=not_completed_processes,
                dim0=col + 1,
            )
        arr_tiles.set_arr(arr_t_tiles.arr.T)

        if len(not_completed_processes) > 0:
            __split0_q_loop(
                r_tiles=arr_t_tiles,
                q0_tiles=q1_tiles,
                dim1=col,
                diag_process=diag_process,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                active_procs=active_procs_t,
                dim0=col + 1,
            )

    q1 = q1.T
    if balance:
        arr_tiles.arr.balance_()
        q0.balance_()
        q1.balance_()

    ret = (q0, arr_tiles.arr, q1.T)
    if ret_tiles:
        tiles = {"q0": q0_tiles, "arr": arr_tiles, "arr_t": arr_t_tiles, "q1": q1_tiles}
        ret = (q0, arr_tiles.arr, q1.T, tiles)
    return ret


@torch.jit.script
def __apply_larft(side, v, tau, c1, c2):
    # type: (str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    """
    replacement for plasma core__dlarfx2
    applies the matrix H = I - tau * v * v.T to c1 and c2
    if side == left: return H @ c1 and H @ c2
    elif side == right: return c1 @ H and c2 @ H
    Parameters
    ----------
    side
    v
    tau
    c1
    c2

    Returns
    -------


    Notes
    -----
    http://icl.cs.utk.edu/plasma/docs/core__dlarfx__tbrd_8c.html#a80d7223148dcbf874885d5bb0707f231

    """

    if side == "left":
        n = c1.shape[0]
    elif side == "right":
        n = c1.shape[1]
    else:
        raise ValueError("side must be either 'left' or 'right', currently: {}".format(side))

    h = torch.eye(n, dtype=v.dtype, device=v.device)
    if tau == 0:
        return c1, c2
    h -= tau * torch.dot(v, v.t())
    if side == "left":
        torch.matmul(h, c1, out=c1)
        torch.matmul(h, c2, out=c2)
    elif side == "right":
        torch.matmul(c1, h, out=c1)
        torch.matmul(c2, h, out=c2)
    return c1, c2


@torch.jit.script
def __gbelr(n, arr, st, end):
    # type: (int, torch.Tensor, int, int) -> Tuple[torch.Tensor, torch.Tensor]
    """
    partial function for bulge chasing, designed for the case that the matrix is upper block diagonal
    LOWEr case

    Parameters
    ----------
    n : int
        order of matrix arr
    arr : torch.Tensor
        tensor on which to do the work, will be overwritten
    st : starting index
    end : ending index

    Returns
    -------
    arr : torch.Tensor
        the same tile which was passed is returned, modified by function
    v : torch.Tensor
        the scalar elementary reflectors
    tau : torch.Tensor
        scalar factors of the elementary reflectors

    Notes
    -----
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6267821

    """
    # notes: n is the order of the matrix, doing UPPER case
    #   nb :
    #   j1 :
    #   j2 :
    #   len1 :
    #   len2 :
    #   t1ed : tile 1 end?
    #   t2st : tile 2 start?
    #   i :
    pass
