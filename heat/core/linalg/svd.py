import torch

from .qr import __r_calc_split0, __qr_split1_loop, __q_calc_split0

from .. import factories

__all__ = ["block_diagonalize"]


def block_diagonalize(arr, overwrite_arr=False):
    if arr.split == 0:
        out = block_diagonalize_sp0(arr, overwrite_arr)
    if arr.split == 1:
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

    # 2. get transpose of arr
    arr_t = arr.T
    # 3. tile arr_t
    arr_t.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    arr_t.tiles.match_tiles_transpose(arr.tiles)
    # print(arr.lshape)
    # print(arr.shape, arr.tiles.tile_rows, arr.tiles.col_indices)
    # print(arr_t.shape, arr_t.tiles.tile_rows, arr_t.tiles.tile_columns)
    # print(arr.tiles.lshape_map)
    # print(arr.tiles.tile_map)

    # return None, None, None

    # 4. match tiles to arr
    arr_t.tiles.match_tiles_transpose(arr.tiles)
    arr_t.tiles.__DNDarray = arr.T

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
    # print(q1.tiles.lshape_map)
    # print(arr_t.tiles.lshape_map)
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

    proc_tile_start_t = torch.cumsum(
        torch.tensor(arr_t.tiles.tile_columns_per_process, device=torch_device), dim=0
    )

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
            __r_calc_split0(
                a_tiles=arr.tiles,
                q_dict=q0_dict,
                q_dict_waits=q0_dict_waits,
                dim1=col,
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
            calc_q=True,
            dim1=col,
            empties=empties_t,
        )
    for col in range(tile_columns - 1):
        diag_process = (
            torch.nonzero(proc_tile_start > col)[0] if col != tile_columns else proc_tile_start[-1]
        )
        diag_process = diag_process.item()

        __q_calc_split0(
            a_tiles=arr.tiles,
            q_tiles=q0.tiles,
            dim1=col,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            diag_process=diag_process,
            active_procs=active_procs,
        )
    # do the last column now
    col = tile_columns - 1
    not_completed_processes = torch.nonzero(col < proc_tile_start).flatten()
    if rank in not_completed_processes and rank in active_procs:
        diag_process = not_completed_processes[0].item()
        __r_calc_split0(
            a_tiles=arr.tiles,
            q_dict=q0_dict,
            q_dict_waits=q0_dict_waits,
            dim1=col,
            diag_process=diag_process,
            not_completed_prs=not_completed_processes,
        )
    diag_process = (
        torch.nonzero(proc_tile_start > col)[0] if col != tile_columns else proc_tile_start[-1]
    ).item()
    __q_calc_split0(
        a_tiles=arr.tiles,
        q_tiles=q0.tiles,
        dim1=col,
        q_dict=q0_dict,
        q_dict_waits=q0_dict_waits,
        diag_process=diag_process,
        active_procs=active_procs,
    )

    arr_t.tiles.__DNDarray = arr.T
    if arr.gshape[0] < arr.gshape[1]:
        # if m < n then need to do another round of LQ
        diag_process = torch.nonzero(col < proc_tile_start_t).flatten()[0].item()
        # print('last if loop', arr_t.tiles.tile_rows, arr_t.tiles.tile_columns)

        __qr_split1_loop(
            a_tiles=arr_t.tiles,
            q_tiles=q1.tiles,
            diag_pr=diag_process,
            dim0=col + 1,
            calc_q=True,
            dim1=col,
            empties=empties_t,
        )

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
    # print(arr.tiles.tile_map)
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
    # print('start')
    q1.tiles.match_tiles(arr_t.tiles)
    # print('stop')
    # print('t', arr_t.tiles.tile_map)
    # print(q1.tiles.tile_map)

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
        diag_process = torch.nonzero(col < proc_tile_start).flatten()[0].item()
        # __qr_split1_loop(
        #     a_tiles=arr.tiles,
        #     q_tiles=q0.tiles,
        #     diag_pr=diag_process,
        #     dim0=col,
        #     calc_q=True,
        #     empties=empties,
        #     arr_t=arr_t,
        # )
        arr_t.tiles.set_arr(arr.tiles.arr.T)

        not_completed_processes = torch.nonzero(col + 1 < proc_tile_start_t).flatten()
        if rank in not_completed_processes and rank in active_procs_t:
            diag_process = not_completed_processes[0].item()
            __r_calc_split0(
                a_tiles=arr_t.tiles,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                dim1=col,
                diag_process=diag_process,
                not_completed_prs=not_completed_processes,
                dim0=col + 1,
            )

        diag_process = (
            torch.nonzero(proc_tile_start_t > col + 1)[0]
            if col + 1 != tile_columns
            else proc_tile_start_t[-1]
        )
        diag_process = diag_process.item()
        __q_calc_split0(
            a_tiles=arr_t.tiles,
            q_tiles=q1.tiles,
            dim1=col,
            q_dict=q1_dict,
            q_dict_waits=q1_dict_waits,
            diag_process=diag_process,
            active_procs=active_procs_t,
            dim0=col + 1,
        )
        arr.tiles.set_arr(arr_t.tiles.arr.T)
    # do the last column now
    col = lp_cols - 1
    # diag_process = torch.nonzero(col < proc_tile_start).flatten()[0].item()
    # __qr_split1_loop(
    #     a_tiles=arr.tiles,
    #     q_tiles=q0.tiles,
    #     diag_pr=diag_process,
    #     dim0=col,
    #     calc_q=True,
    #     empties=empties,
    #     arr_t=arr_t,
    # )

    arr_t.tiles.__DNDarray = arr.T
    if arr.gshape[0] < arr.gshape[1]:  # and arr_t.tiles.lshape_map[0, 0]
        # if m < n then need to do another round of LQ
        # diag_process = torch.nonzero(col < proc_tile_start_t).flatten()[0].item()
        # print('last if loop', arr_t.tiles.tile_rows, arr_t.tiles.tile_columns)
        # col = arr_t.tiles.tile_columns - 1
        row = col + 1
        if tiles_per_proc == 1:
            # make col the last row after the last diagonal process
            row = sum(arr_t.tiles.tile_rows_per_process[: arr_t.tiles.last_diagonal_process + 1])
        not_completed_processes = torch.nonzero(row < proc_tile_start_t).flatten()
        # print(not_completed_processes)
        if rank in not_completed_processes and rank in active_procs_t:
            diag_process = not_completed_processes[0].item()
            __r_calc_split0(
                a_tiles=arr_t.tiles,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                dim1=col,
                diag_process=diag_process,
                not_completed_prs=not_completed_processes,
                dim0=row,
            )
            # print(arr_t.tiles.arr.T._DNDarray__array)

        # diag_process = (
        #     torch.nonzero(proc_tile_start_t > row)[0]
        #     if row != tile_columns
        #     else proc_tile_start_t[-1]
        # )
        if len(not_completed_processes) > 0:
            diag_process = not_completed_processes[0].item()
            __q_calc_split0(
                a_tiles=arr_t.tiles,
                q_tiles=q1.tiles,
                dim1=col,
                q_dict=q1_dict,
                q_dict_waits=q1_dict_waits,
                diag_process=diag_process,
                active_procs=active_procs_t,
                dim0=row,
            )
            arr.tiles.set_arr(arr_t.tiles.arr.T)
    # print(arr_t.tiles.lshape_map)
    # print(arr_t.gshape)
    # print(arr_t.tiles.arr.round())

    q1 = q1.T
    arr.tiles.arr.balance_()
    q0.balance_()
    q1.balance_()
    return q0, arr.tiles.arr, q1
