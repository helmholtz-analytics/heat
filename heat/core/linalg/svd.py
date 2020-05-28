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
    lcl_right_apply = {}
    # lcl_left_apply = {}
    # todo: is it needed to only run it on <= ldp?
    # if rank <= ldp:
    # get the lshape with the diagonal element
    start1 = splits[rank - 1] if rank > 0 else 0
    stop1 = splits[rank + 1].clone() if rank < arr.comm.size - 1 else arr.gshape[arr.split - 1]
    stop1 += band_width + 1
    lcl_array = arr._DNDarray__array[:, start1:stop1]
    lcl_d_st = 0 if rank == 0 else splits[rank - 1]
    lcl_d_sp = splits[rank] + band_width + 1 if rank < arr.comm.size - 1 else arr.gshape[1]
    # todo: send a copy of the data on the processes to the next and previous ones

    nrows = min(arr.gshape) - 1 if arr.gshape[0] >= arr.gshape[1] else min(arr.gshape) + 1
    app_sz = band_width - 1
    # if rank <= 1:
    #     min_i = 0
    # else:
    #     min_i = splits[rank - 2].item()
    # first round:
    #   - gen vector (shape=(app_sz,)),
    #   - apply from right (down the matrix) on a (app_sz + 1 x app_sz) starting at (row, row + 1)
    # if rank == 0:
    #     # this only works for the first loop
    #     # need something like this for every loop
    #     v, t = utils.gen_house_vec(lcl_array[0, 1: 1 + band_width])
    #     st_ind = (0, 1)
    #     end_ind = (len(v) + 1, )

    for row in range(nrows):
        # todo: abstract this ==========================================================================
        wait_before = None
        wait_after = None
        if rank != arr.comm.size - 1:
            arr.comm.Send(lcl_array[:, lcl_d_st:lcl_d_sp].clone(), dest=rank + 1)
            rcv_shape = [lshape_map[rank + 1, 0], 0]
            rcv_shape[1] = splits[rank + 1] + band_width + 1 - splits[rank]
            rcv = torch.zeros(
                tuple(rcv_shape), dtype=arr.dtype.torch_type(), device=arr.device.torch_device
            )
            # todo: pad the lclarray to fit the stuff that will be recv'ed
            wait_after = arr.comm.Irecv(rcv, source=rank + 1)
        if rank != 0:
            # send / recv data to the previous process
            arr.comm.Send(lcl_array[:, lcl_d_st:lcl_d_sp].clone(), dest=rank - 1)
            rcv_shape = [splits[rank - 1], 0]
            rcv_shape[1] = (
                splits[rank - 1] + band_width + 1 - (splits[rank - 1] if rank != 1 else 0)
            )
            rcv = torch.zeros(
                tuple(rcv_shape), dtype=arr.dtype.torch_type(), device=arr.device.torch_device
            )
            # todo: pad the lclarray to fit the stuff that will be recv'ed
            wait_before = arr.comm.Irecv(rcv, source=rank - 1)
        if wait_before is not None:
            # todo: move this wait, need to extend the lcl array and add the received one to to
            wait_before.wait()
        if wait_after is not None:
            # todo: create a new matrix with more space and set the diag bit in the right place
            # new end of the array is that of the next process
            # todo: move this wait, need to extend the lcl array and add the received one to to
            wait_after.wait()
        # todo: abstract above =======================================================================

        # this is the loop for the start index
        if any(row >= splits[active_procs[0] :]):
            if rank == active_procs[0]:
                break
            del active_procs[0]
        # 1. get start element
        st_ind = (row, row + 1)
        end_ind_first = (row + 1 + app_sz, row + 1 + app_sz)
        # 2. determine where it will be applied to
        #       first round is a special case: it wont be the full thing, only 1 other element below the target point
        # get which processes are involved, need to figure out how to deal with the overlapping ones still
        inds = [st_ind]
        num = 0
        nxt_ind = list(st_ind)
        end = min(
            arr.gshape
        )  # todo: if the diagonal extends beyond the minimum dim (SF) + band_width - 2
        completed = False
        # determine all the indices
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
        for i in inds:
            # print(i, side)
            # determine the area effected in each loop
            if i == inds[0]:
                lp_end_inds = end_ind_first
            elif side == "right":
                lp_end_inds = [2 * app_sz + i[0], app_sz + i[1]]
            else:  # side == "left"

                lp_end_inds = [app_sz + i[0], 2 * app_sz + i[1]]
                if lp_end_inds[1] > arr.gshape[1]:
                    lp_end_inds[1] = arr.gshape[1]
                if lp_end_inds[0] > arr.gshape[0]:
                    lp_end_inds[0] = arr.gshape[0]

            # todo: determine the processes which have this data, also have the
            #       use splits to determine if it overlaps processes
            if i[0] > splits[rank]:
                break
            # do work if i[0] is on the rank or rank - 1
            first_pr = torch.where(i[0] <= splits)[0][0]
            second_pr = torch.where(lp_end_inds[0] <= splits)[0]
            second_pr = second_pr[0] if len(second_pr) > 0 else arr.comm.size - 1
            # print(i, lp_end_inds, first_pr, second_pr)
            if first_pr == second_pr == rank:
                # only need to work on this one
                # target row should be row or column based on the side
                sl = (
                    (i[0] - start1, slice(i[1], lp_end_inds[1]))
                    if side == "right"
                    else (slice(i[0], lp_end_inds[0]), i[1] - start1)
                )
                # target = lcl_array[i[0] - start1, i[1]: lp_end_inds[1]]
                # print(sl, side)
                # print(lcl_array[sl].shape)
                v, t = utils.gen_house_vec(lcl_array[sl])
                # save v and t for updating Q0/Q1
                lcl_right_apply[(row, row + band_width)] = (v, t)
                # apply v ant t to from right (down the matrix)
                print(i[0] - start1, lp_end_inds, i, splits)
                lcl_array[
                    i[0] - start1 : lp_end_inds[0] - start1, i[1] : lp_end_inds[1]
                ] = utils.apply_house(
                    side=side,
                    v=v,
                    tau=t,
                    c=lcl_array[i[0] - start1 : lp_end_inds[0] - start1, i[1] : lp_end_inds[1]],
                )
            elif rank in [first_pr, second_pr]:
                if rank == first_pr:
                    if wait_after is None:
                        wait_after.wait()  # rcv has the data from rank + 1
                        wait_after = None
                    # recv data from second
                    pass
                else:  # rank == second_pr
                    if wait_before is None:
                        wait_before.wait()  # rcv has the data from rank - 1
                    # recv data from first
                    pass
            if i == (1, 1):
                break
            # print(i, splits, )
            # print(i, lp_end_inds, )
            # if min_i <= i[0] <= splits[rank]:
            # translate global start index into local
            # todo: how many processes have the info? does that matter?

            # print(splits, i, i[0], lp_end_inds[0], min_i)

            # change right to left or vice versa
            side = "right" if side == "left" else "left"
            # break
        pass
        # if rank == active_procs[0]:
        #     # generate the vector to eliminate the row (working on split 0 now)
        #     # todo: split 1 version
        #     v, t = utils.gen_house_vec(lcl_array[row - splits[rank], row + 1 : row + band_width])
        #     # save v and t for updating Q0/Q1
        #     lcl_right_apply[(row, row + band_width)] = (v, t)
        #     # apply v ant t to from right (down the matrix)
        #     lcl_array[
        #         st_ind[0] : end_ind_first[0], st_ind[1] : end_ind_first[1]
        #     ] = utils.apply_house(
        #         side="right",
        #         v=v,
        #         tau=t,
        #         c=lcl_array[st_ind[0] : end_ind_first[0], st_ind[1] : end_ind_first[1]],
        #     )
        break
