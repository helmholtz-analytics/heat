import torch

from .qr import __split0_r_calc, __split0_q_loop, __split1_qr_loop

from heat.core import factories

__all__ = ["block_diagonalize"]


def block_diagonalize(arr, overwrite_arr=False):
    if arr.split == 0:
        out = block_diagonalize_sp0(arr, overwrite_arr)
    elif arr.split == 1:
        out = block_diagonalize_sp1(arr, overwrite_arr)
    return out


def block_diagonalize_sp0(arr, overwrite_arr=False):
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
    if arr.tiles is None:
        arr.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q1 = factories.eye(
        (arr.gshape[1], arr.gshape[1]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )

    if arr.tiles.lshape_map[0, 0] >= arr.gshape[1]:
        # in this case forming a band diagonal by interlacing QR and LQ doesn work so its just QR
        qr = arr.qr(tiles_per_proc=tiles_per_proc)
        return qr.Q, qr.R, q1

    # 2. get transpose of arr
    arr_t = arr.T
    # 3. tile arr_t
    arr_t.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    # 4. match tiles to arr
    arr_t.tiles.match_tiles_qr_lq(arr.tiles)
    arr_t.tiles.__DNDarray = arr.T

    q0 = factories.eye(
        (arr.gshape[0], arr.gshape[0]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q0.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q0.tiles.match_tiles(arr.tiles)

    q1.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q1.tiles.match_tiles(arr_t.tiles)
    # ----------------------------------------------------------------------------------------------
    tile_columns = arr.tiles.tile_columns

    torch_device = arr._DNDarray__array.device
    q0_dict = {}
    q0_dict_waits = {}

    active_procs = torch.arange(arr.comm.size)
    empties = torch.nonzero(arr.tiles.lshape_map[..., arr.split] == 0)
    empties = empties[0] if empties.numel() > 0 else []
    for e in empties:
        active_procs = active_procs[active_procs != e]
    tile_rows_per_pr_trmd = arr.tiles.tile_rows_per_process[: active_procs[-1] + 1]

    active_procs_t = torch.arange(arr_t.comm.size)
    empties_t = torch.nonzero(arr_t.tiles.lshape_map[..., arr_t.split] == 0)
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
        if rank in not_completed_processes and rank in active_procs:
            # if the process is done calculating R the break the loop
            # break
            diag_process = not_completed_processes[0].item()
            __split0_r_calc(
                r_tiles=arr.tiles,
                q_dict=q0_dict,
                q_dict_waits=q0_dict_waits,
                dim1=col,
                diag_process=diag_process,
                not_completed_prs=not_completed_processes,
            )
        arr_t.tiles.set_arr(arr.tiles.arr.T)
        # 2. do full QR on the next column for LQ on arr_t
        __split1_qr_loop(dim1=col, r=arr_t, q0=q1, calc_q=True, dim0=col + 1, empties=empties_t)
    for col in range(tile_columns - 1):
        __split0_q_loop(
            r=arr,
            q0=q0,
            dim1=col,
            proc_tile_start=proc_tile_start,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            active_procs=active_procs,
        )

    # do the last column now
    col = tile_columns - 1
    not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
    if rank in not_completed_processes and rank in active_procs:
        diag_process = not_completed_processes[0].item()
        __split0_r_calc(
            r_tiles=arr.tiles,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            dim1=col,
            diag_process=diag_process,
            not_completed_prs=not_completed_processes,
        )
    __split0_q_loop(
        r=arr,
        q0=q0,
        dim1=col,
        proc_tile_start=proc_tile_start,
        q_dict=q0_dict,
        q_dict_waits=q0_dict_waits,
        active_procs=active_procs,
    )

    arr_t.tiles.set_arr(arr.tiles.arr.T)
    diag_diff = arr_t.tiles.row_indices[1]
    if arr.gshape[0] < arr.gshape[1] - diag_diff:
        __split1_qr_loop(dim1=col, r=arr_t, q0=q1, calc_q=True, dim0=col + 1, empties=empties_t)

    q1 = q1.T
    arr.balance_()
    q0.balance_()
    q1.balance_()
    return q0, arr, q1


def block_diagonalize_sp1(arr, overwrite_arr=False):
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
    if arr.tiles is None:
        arr.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    # 2. get transpose of arr
    arr_t = arr.T.copy()
    # 3. tile arr_t
    arr_t.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    # 4. match tiles to arr
    arr_t.tiles.match_tiles_qr_lq(arr.tiles)

    q0 = factories.eye(
        (arr.gshape[0], arr.gshape[0]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q0.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q0.tiles.match_tiles(arr.tiles)

    q1 = factories.eye(
        (arr.gshape[1], arr.gshape[1]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q1.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q1.tiles.match_tiles(arr_t.tiles)
    # -------------------------- split = 1 stuff (att) ---------------------------------------------
    tile_columns = arr.tiles.tile_columns
    tile_rows = arr.tiles.tile_rows

    torch_device = arr._DNDarray__array.device

    active_procs = torch.arange(arr.comm.size)
    empties = torch.nonzero(arr.tiles.lshape_map[..., arr.split] == 0)
    empties = empties[0] if empties.numel() > 0 else []
    for e in empties:
        active_procs = active_procs[active_procs != e]

    proc_tile_start = torch.cumsum(
        torch.tensor(arr.tiles.tile_columns_per_process, device=torch_device), dim=0
    )
    # -------------------------- split = 0 stuff (arr_t) -------------------------------------------
    active_procs_t = torch.arange(arr_t.comm.size)
    empties_t = torch.nonzero(arr_t.tiles.lshape_map[..., 0] == 0)
    empties_t = empties_t[0] if empties_t.numel() > 0 else []
    for e in empties_t:
        active_procs_t = active_procs_t[active_procs_t != e]
    tile_rows_per_pr_trmd_t = arr_t.tiles.tile_rows_per_process[: active_procs_t[-1] + 1]

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
        __split1_qr_loop(dim1=col, r=arr, q0=q0, calc_q=True, empties=empties)

        arr_t.tiles.set_arr(arr.tiles.arr.T)

        not_completed_processes = torch.nonzero(col + 1 < proc_tile_start_t).flatten()
        if rank in not_completed_processes and rank in active_procs_t:
            # if the process is done calculating R the break the loop
            # break
            diag_process = not_completed_processes[0].item()
            __split0_r_calc(
                r_tiles=arr_t.tiles,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                dim1=col,
                diag_process=diag_process,
                not_completed_prs=not_completed_processes,
                dim0=col + 1,
            )

        __split0_q_loop(
            r=arr_t,
            q0=q1,
            dim1=col,
            proc_tile_start=proc_tile_start,
            q_dict=q1_dict,
            q_dict_waits=q1_dict_waits,
            active_procs=active_procs_t,
            dim0=col + 1,
        )
        arr.tiles.set_arr(arr_t.tiles.arr.T)
    # do the last column now
    col = lp_cols - 1
    __split1_qr_loop(dim1=col, r=arr, q0=q0, calc_q=True, empties=empties)

    arr_t.tiles.set_arr(arr.T)
    if arr.gshape[0] < arr.gshape[1]:
        # if m < n then need to do another round of LQ
        not_completed_processes = torch.nonzero(col + 1 < proc_tile_start_t).flatten()
        if rank in not_completed_processes and rank in active_procs_t:
            # if the process is done calculating R the break the loop
            # break
            diag_process = not_completed_processes[0].item()
            __split0_r_calc(
                r_tiles=arr_t.tiles,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                dim1=col,
                diag_process=diag_process,
                not_completed_prs=not_completed_processes,
                dim0=col + 1,
            )
        arr.tiles.set_arr(arr_t.tiles.arr.T)

        if len(not_completed_processes) > 0:
            __split0_q_loop(
                r=arr_t,
                q0=q1,
                dim1=col,
                proc_tile_start=proc_tile_start,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                active_procs=active_procs_t,
                dim0=col + 1,
            )

    q1 = q1.T
    arr.tiles.arr.balance_()
    q0.balance_()
    q1.balance_()
    return q0, arr.tiles.arr, q1
