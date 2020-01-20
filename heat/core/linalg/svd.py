import torch

from .qr import *
from .qr import __r_calc_split0, __qr_split1_loop, __q_calc_split0

from .. import factories
from .. import tiling

__all__ = ["block_diagoalize", "svd"]


def block_diagoalize(arr, tiles_per_proc=2, overwrite_arr=False):
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
    if arr.tiles is None:
        arr.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)

    # 2. get transpose of arr
    arr_t = arr.T
    # 3. tile arr_t
    arr_t.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    # 4. match tiles to arr
    arr_t.tiles.match_tiles(arr.tiles)
    arr_t.tiles.__DNDarray = arr.T

    # print(arr.tiles.tile_map)
    # print(arr_t.tiles.tile_map)

    # todo: change the split dynamically -----------------------------------------------------------
    q0 = factories.eye(
        (arr.gshape[0], arr.gshape[0]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q0.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q0.tiles.match_tiles(arr.tiles)

    q1 = factories.eye(
        (arr.gshape[1], arr.gshape[1]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q1.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q1.tiles.match_tiles(arr.tiles)
    # ----------------------------------------------------------------------------------------------
    tile_columns = arr.tiles.tile_columns

    torch_device = arr._DNDarray__array.device
    q0_dict = {}
    q0_dict_waits = {}

    active_procs = torch.arange(arr.comm.size)
    empties = torch.nonzero(arr.tiles.lshape_map[..., 0] == 0)
    empties = empties[0] if empties.numel() > 0 else []
    for e in empties:
        active_procs = active_procs[active_procs != e]
    tile_rows_per_pr_trmd = arr.tiles.tile_rows_per_process[: active_procs[-1] + 1]

    proc_tile_start = torch.cumsum(torch.tensor(tile_rows_per_pr_trmd, device=torch_device), dim=0)

    proc_tile_start_t = torch.cumsum(
        torch.tensor(arr_t.tiles.tile_columns_per_process, device=torch_device), dim=0
    )

    # looping over number of tile columns - 1 (col)
    # 1. do QR on arr for column=col (standard QR as written)
    # 2. do LQ on arr_t for column=col+1 (standard QR again, the transpose makes it LQ)
    #       both of these steps overwrite arr (or an initial copy of it, optional)
    # 3. do QR on the last column if m >= n
    for col in range(tile_columns - 1):
        # 1. do QR on arr for column=col (standard QR as written) (this assumes split == 0)
        not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
        diag_process = not_completed_processes[0].item()
        __r_calc_split0(
            a_tiles=arr.tiles,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            col_num=col,
            diag_process=diag_process,
            not_completed_prs=not_completed_processes,
        )
        arr_t.tiles.__DNDarray = arr.T
        # 2. do full QR on the next column for LQ on arr_t
        diag_process = torch.nonzero(col < proc_tile_start_t).flatten()[0].item()
        __qr_split1_loop(
            a_tiles=arr_t.tiles,
            q_tiles=q1.tiles,
            diag_pr=diag_process,
            dim0=col + 1,
            calc_q=False,
            dim1=col,
        )
    for col in range(tile_columns - 1):
        diag_process = (
            torch.nonzero(proc_tile_start > col)[0] if col != tile_columns else proc_tile_start[-1]
        )
        diag_process = diag_process.item()

        __q_calc_split0(
            a_tiles=arr.tiles,
            q_tiles=q0.tiles,
            col=col,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            diag_process=diag_process,
            active_procs=active_procs,
        )
    if arr.gshape[0] >= arr.gshape[1]:
        # todo: this is working on the 0th process as well for some reason :/
        # only need to do qr on the last column if m >= n
        col = tile_columns - 1
        not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
        diag_process = not_completed_processes[0].item()
        __r_calc_split0(
            a_tiles=arr.tiles,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            col_num=col,
            diag_process=diag_process,
            not_completed_prs=not_completed_processes,
        )

        diag_process = (
            torch.nonzero(proc_tile_start > col)[0] if col != tile_columns else proc_tile_start[-1]
        )
        diag_process = diag_process.item()

        __q_calc_split0(
            a_tiles=arr.tiles,
            q_tiles=q0.tiles,
            col=col,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            diag_process=diag_process,
            active_procs=active_procs,
        )

    q1 = q1.T
    arr.balance_()
    arr_t.balance_()
    q0.balance_()
    q1.balance_()
    # print((arr_t._DNDarray__array * 100).round())
    return q0, arr, q1


def svd(arr):
    pass
