import torch

from .qr import __split0_r_calc, __split0_q_loop, __split1_qr_loop
from .. import dndarray
from .. import factories
from .. import tiling
from . import utils

from ..communication import MPI


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
    # todo: increase the tile number of possible (need to measure performance)
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

    # print(arr_tiles.row_indices, arr_tiles.col_indices, arr_tiles.tile_rows_per_process)
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
    # todo: is it needed to only run it on <= ldp?
    # get the lshape with the diagonal element
    start1 = splits[rank - 1].item() if rank > 0 else 0
    stop1 = splits[rank].item() + (2 * band_width - 3)
    # stop1 += band_width + 1
    lcl_array = arr._DNDarray__array[:, start1:stop1]
    nrows = min(arr.gshape) - 1 if arr.gshape[0] >= arr.gshape[1] else min(arr.gshape) + 1
    rcv_next = None
    st, sp = 0, 2 * band_width - 3
    rcv_shape = [sp - st] * 2
    out_array_inds = [0, splits[rank], start1, splits[rank] + band_width]  # relative to work arr
    data_to_send_next_inds = [None, None]  # relative to work arr
    data_to_send_prev_inds = [None, None]  # relative to work arr
    work_arr_inds = [None, None]  # global
    loc_data_inds = (slice(start1, splits[rank]), slice(start1, splits[rank] + band_width))
    if rank != arr.comm.size - 1:
        # send to rank + 1
        st1 = splits[rank].item() - (band_width - 2)
        st1 = st1 if rank == 0 else st1 - splits[rank - 1].item()
        sp0 = splits[rank].item()
        st0 = sp0 - sp
        st1_g = start1 + st1
        sp1_g = stop1
        data_to_send_next_inds = [st0, sp0, st1_g, sp1_g]
        # lcl_send_inds = (slice(sp * -1, None), slice(st1, None))
        # print(data_to_send_next_inds)
        arr.comm.send((st0, sp0, st1_g, sp1_g), dest=rank + 1)
        rcv_next = torch.zeros(
            tuple(rcv_shape), dtype=arr.dtype.torch_type(), device=arr.device.torch_device
        )
        arr.comm.Recv(rcv_next, source=rank + 1)
    if rank != 0:
        # rcv_from_prev_shape = [lshape_map[rank - 1, 0] - sp, sp1 - st1]
        # this happens at the beginning
        # send / recv data to the previous process
        data_to_send_prev_inds = [st, sp, start1 + st, stop1 - sp]
        print(data_to_send_prev_inds, st, sp, stop1)
        arr.comm.Send(lcl_array[st:sp, st:sp].clone(), dest=rank - 1)
        rcv_inds = arr.comm.recv(source=rank - 1)
    else:
        rcv_inds = None
    # todo: make the start and stop to convert the global index to the local index on work array
    work_arr_inds[0] = [start1, splits[rank].item()]
    work_arr_inds[1] = [start1, stop1]
    if rcv_next is not None and rcv_inds is not None:
        # rcv from both directions
        rcv_prev_shape = (rcv_inds[1] - rcv_inds[0], rcv_inds[3] - rcv_inds[2])
        work_arr_inds[0][0] = rcv_inds[0]
        work_arr_inds[1][0] = rcv_inds[2]
        work_arr = torch.zeros(
            (
                lcl_array.shape[0] + rcv_prev_shape[0] + 2 * band_width - 3,
                lcl_array.shape[1] + (band_width - 2) + (rcv_next.shape[1] - band_width) - 1,
            ),
            dtype=lcl_array.dtype,
            device=lcl_array.device,
        )
        # work_arr[: rcv_prev_shape[0], : rcv_prev_shape[1]] = rcv_prev
        rcv_set = (slice(None, rcv_prev_shape[0]), slice(None, rcv_prev_shape[1]))
        st0, sp0 = rcv_prev_shape[0], lcl_array.shape[0] + rcv_prev_shape[0]
        st1, sp1 = band_width - 2, lcl_array.shape[1] + band_width - 2
        work_arr[st0:sp0, st1:sp1] += lcl_array
        work_arr[rcv_next.shape[0] * -1 :, sp1 - band_width - 1 :] = rcv_next

        # data_to_send_prev_inds
        lcl_array_start = (st0, sp0, st1, sp1)
    else:
        # only one side receiving
        if rcv_next is not None:  # only get from rank + 1 (lcl data on top)
            # only the case for rank == 0
            work_arr = torch.zeros(
                (
                    lcl_array.shape[0] + rcv_next.shape[0],
                    lcl_array.shape[1] + rcv_next.shape[1] - band_width,
                ),
                dtype=lcl_array.dtype,
                device=lcl_array.device,
            )
            work_arr[: lcl_array.shape[0], : lcl_array.shape[1]] += lcl_array
            work_arr[
                lcl_array.shape[0] :, lcl_array.shape[0] : lcl_array.shape[0] + rcv_next.shape[1]
            ] = rcv_next

            lcl_array_start = (0, lcl_array.shape[0], 0, lcl_array.shape[1])
        else:  # only get from rank - 1 (lcl data on bottom)
            # only the case for rank == size - 1
            rcv_prev_shape = (rcv_inds[1] - rcv_inds[0], rcv_inds[3] - rcv_inds[2])
            work_arr_inds[0][0] = rcv_inds[0]
            work_arr_inds[1][0] = rcv_inds[2]
            work_arr = torch.zeros(
                (lcl_array.shape[0] + rcv_prev_shape[0], lcl_array.shape[1] + band_width - 2),
                dtype=lcl_array.dtype,
                device=lcl_array.device,
            )
            rcv_set = (slice(None, rcv_prev_shape[0]), slice(None, rcv_prev_shape[1]))
            # work_arr[: rcv_prev.shape[0], : rcv_prev_shape[1]] = rcv_prev
            # print(rcv_inds, lcl_array.shape)
            work_arr[rcv_prev_shape[0] :, lcl_array.shape[1] * -1 :] += lcl_array

            # loc_ind0[0] = work_arr.shape[0] - rcv_prev_shape[0] + 1
            # loc_ind1[0] = loc_ind1[1] - work_arr.shape[1]

            # lcl_array_start = (loc_ind0[0], None, lcl_array.shape[1] * -1, -1)
    work_arr_inds[0][1] = work_arr_inds[0][0] + work_arr.shape[0]
    work_arr_inds[1][1] = work_arr_inds[1][0] + work_arr.shape[1]
    if rank != arr.comm.size - 1:
        data_to_send_next_inds[0] -= work_arr_inds[0][0]
        data_to_send_next_inds[1] -= work_arr_inds[0][0]
        data_to_send_next_inds[2] -= work_arr_inds[1][0]
        data_to_send_next_inds[3] -= work_arr_inds[1][0]
    # data_to_send_next_inds indices are relative to work arr
    # work_arr_inds are relative to global
    work_arrs_map = torch.zeros((arr.comm.size, 4), dtype=torch.int, device=arr.device.torch_device)
    work_arrs_map[rank] = torch.tensor(
        work_arr_inds, dtype=torch.int, device=arr.device.torch_device
    ).flatten()
    arr.comm.Allreduce(MPI.IN_PLACE, work_arrs_map, MPI.SUM)
    print(work_arrs_map, work_arr.shape)
    # print(data_to_send_next_inds)
    app_sz = band_width - 1
    lcl_right_apply = {}
    lcl_left_apply = {}
    for row in range(1, nrows):
        # todo: loc_ind0/1 have the global start and stop for work_arr!
        # this is the loop for the start index
        # todo: put this back in
        # if any(row >= splits[active_procs[0] :]):
        #     if rank == active_procs[0]:
        #         break
        #     del active_procs[0]
        # # 1. get start element
        st_ind = (row, row + 1)
        end_ind_first = (row + 1 + app_sz, row + 1 + app_sz)
        # 2. determine where it will be applied to
        #       first round is a special case: it wont be the full thing, only 1 other element below the target point
        # get which processes are involved, need to figure out how to deal with the overlapping ones still
        inds = [st_ind]
        num = 0
        nxt_ind = list(st_ind)
        end = min(arr.gshape) - 1
        # todo: if the diagonal extends beyond the minimum dim (SF) + band_width - 2
        completed = False
        # determine all the indices
        # todo: i think that this can be moved outside the loop, after the first loop, all elements increase by 1
        while not completed:
            if nxt_ind[0] == nxt_ind[1]:
                nxt_ind[1] += band_width - 1
            else:
                nxt_ind[0] = nxt_ind[1]
            num += 1
            if any([n >= end for n in nxt_ind]):
                break
            inds.append(tuple(nxt_ind))
        # the side to apply the vectors on starts with right and then alternates until the end of inds
        side = "right"
        sent_data = False
        if rank > active_procs[0]:
            # todo: need to fix this
            # create the wait for the data from the rank - 1
            #   not sent yet, will be sent during the loop, the wait will be used as a barrier
            rcv_prev = torch.zeros(
                rcv_prev_shape, dtype=arr.dtype.torch_type(), device=arr.device.torch_device
            )
            arr.comm.Recv(rcv_prev, source=rank - 1, tag=row)
            work_arr[rcv_set] = rcv_prev
        for i in inds:
            if i == (2, 2):
                # print((work_arr * 100).round())
                break
            # print(i, side)
            # lp_end_inds == the indices for the end of the area effected
            if i == inds[0]:
                lp_end_inds = end_ind_first
            elif side == "right":
                lp_end_inds = [2 * app_sz + i[0], app_sz + i[1]]
            else:  # side == "left"
                lp_end_inds = [app_sz + i[0], 2 * app_sz + i[1]]

            if lp_end_inds[0] > arr.gshape[0]:
                lp_end_inds[0] = arr.gshape[0]
            if lp_end_inds[1] > arr.gshape[1]:
                lp_end_inds[1] = arr.gshape[1]

            # determine the processes which have this data, also have the
            #       use splits to determine if it overlaps processes
            if i[0] > splits[rank]:
                # send the local data to rank - 1
                break
            # do work if i[0] is on the rank or rank - 1
            print(i)
            # find the overlaps on the first two processes, wont work beyond that, this keeps everything in order
            #   but it may be a point where it is slower
            prs_dim0 = [
                torch.where((work_arrs_map[..., 0] <= i[0]) & (i[0] < work_arrs_map[..., 1]))[
                    0
                ].tolist()[:2],
                torch.where(
                    (work_arrs_map[..., 0] <= lp_end_inds[0])
                    & (lp_end_inds[0] < work_arrs_map[..., 1])
                )[0].tolist()[:2],
            ]
            prs_dim1 = [
                torch.where((work_arrs_map[..., 2] <= i[1]) & (i[1] < work_arrs_map[..., 3]))[
                    0
                ].tolist()[:2],
                torch.where(
                    (work_arrs_map[..., 2] <= lp_end_inds[1])
                    & (lp_end_inds[1] < work_arrs_map[..., 3])
                )[0].tolist()[:2],
            ]
            # print(prs_dim0, prs_dim1)
            overlap0, overlap1 = False, False
            if len(prs_dim0[0]) != len(prs_dim0[1]):
                overlap0 = True
            if len(prs_dim1[0]) != len(prs_dim1[1]):
                overlap1 = True
            # print(overlap0, overlap1)

            comm_result, comm_vec = False, False
            if overlap0 and overlap1:
                # overlap in both dims
                comm_result = True
                pass
            elif overlap0 and not overlap1:
                # overlap in the dim0
                if side == "right":
                    comm_result = True
                else:
                    comm_vec = True
            elif not overlap0 and overlap1:  # only overlap1 is true
                # overlap in dim1
                if side == "left":
                    comm_result = True
                else:
                    comm_vec = True
            else:  # both false
                first_pr = prs_dim0[0][0]
                second_pr = prs_dim0[0][1] if len(prs_dim0) > 1 else first_pr
            print(comm_result, comm_vec)
            if comm_vec or comm_result:
                # need to find the process with all the vector data
                if side == "right":
                    first_pr = list(set(prs_dim1[0]).intersection(prs_dim1[1]))
                    vec_pr = first_pr
                else:
                    first_pr = list(set(prs_dim0[0]).intersection(prs_dim0[1]))
                    vec_pr = first_pr
                    second_pr = prs_dim0[0][1] if len(prs_dim0) > 1 else prs_dim1[1][1]
            if comm_result:
                # need to find the indices of where to put the rest of the data
                # todo: this part...
                # if only working on two processes, then there will only be data passed to rank + 1

                pass

            # todo: below was working when there is not an overlapping
            # second_pr = torch.where(lp_end_inds[0] <= splits)[0]
            # second_pr = second_pr[0] if len(second_pr) > 0 else arr.comm.size - 1
            #
            # print(row, i, lp_end_inds, top_prs, bottom_prs, side)
            # if rank in [first_pr, second_pr]:
            #     # target row should be row or column based on the side
            #     work_start = (work_arr_inds[0][0], work_arr_inds[1][0])
            #     sl = (
            #         (i[0] - work_start[0], slice(i[1] - work_start[1], lp_end_inds[1] - work_start[1]))
            #         if side == "right"
            #         else (
            #             slice(i[0] - work_start[0], lp_end_inds[0] - work_start[0]),
            #             i[1] - work_start[1],
            #         )
            #     )
            #     # print(row, i, sl, work_start, work_arr[sl])
            #     # if i == (10, 13):
            #     #     print('here')
            #     #     break
            #     # todo: if the target vector is not all the way on the second process
            #     #       then send the result
            #     # need to know the start points of the work_arr on all processes
            #     if (side == "right" and i[1] - work_start[1] >= 0) or (
            #             side == "left" and i[0] - work_start[0] >= 0
            #     ):
            #         # target = lcl_array[i[0] - start1, i[1]: lp_end_inds[1]]
            #         v, t = utils.gen_house_vec(work_arr[sl])
            #         # save v and t for updating Q0/Q1
            #         if side == "right":
            #             lcl_right_apply[i] = (v, t)
            #         else:
            #             lcl_left_apply[i] = (v, t)
            #         # apply v ant t to from right (down the matrix)
            #         set_slice = (
            #             slice(i[0] - work_start[0], lp_end_inds[0] - work_start[0]),
            #             slice(i[1] - work_start[1], lp_end_inds[1] - work_start[1]),
            #         )
            #         # print(sl, '\t', set_slice)
            #         # if i == (2, 5):
            #         #     break
            #         work_arr[set_slice] = utils.apply_house(
            #             side=side, v=v, tau=t, c=work_arr[set_slice]
            #         )
            #         if (
            #                 not sent_data
            #                 and first_pr == second_pr == rank != arr.comm.size - 1
            #                 and lp_end_inds[1] > st1_g
            #                 and rank != arr.comm.size - 1
            #         ):
            #             # print('sending', i, row)
            #             # this sends the data to the next process to update the next
            #             # st0, sp0, st1, sp1 = data_to_send_next_inds
            #             hslice = (slice(data_to_send_next_inds[0], data_to_send_next_inds[1]),
            #                       slice(data_to_send_next_inds[2], data_to_send_next_inds[3]))
            #             # print('r', st0, sp0, st1, sp1, lcl_send_inds)
            #             arr.comm.Send(
            #                 work_arr[hslice].clone(), dest=rank + 1, tag=row
            #             )
            #             sent_data = True
            side = "right" if side == "left" else "left"
        # todo: at end of loop update the data sent to the previous processes
        if row == 1:
            break
    # print((work_arr * 100).round())

    # break
