"""
QR decomposition of (distributed) 2-D ``DNDarray``s.
"""
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple

from ..communication import MPICommunication
from ..types import datatype
from ..tiling import SquareDiagTiles
from ..dndarray import DNDarray
from .. import factories

__all__ = ["qr"]


def qr(
    a: DNDarray,
    tiles_per_proc: Union[int, torch.Tensor] = 1,
    calc_q: bool = True,
    overwrite_a: bool = False,
) -> Tuple[DNDarray, DNDarray]:
    r"""
    Calculates the QR decomposition of a 2D ``DNDarray``.
    Factor the matrix ``a`` as *QR*, where ``Q`` is orthonormal and ``R`` is upper-triangular.
    If ``calc_q==True``, function returns ``QR(Q=Q, R=R)``, else function returns ``QR(Q=None, R=R)``

    Parameters
    ----------
    a : DNDarray
        Array which will be decomposed
    tiles_per_proc : int or torch.Tensor, optional
        Number of tiles per process to operate on,
    calc_q : bool, optional
        Whether or not to calculate Q.
        If ``True``, function returns ``(Q, R)``.
        If ``False``, function returns ``(None, R)``.
    overwrite_a : bool, optional
        If ``True``, function overwrites ``a`` with R
        If ``False``, a new array will be created for R

    Notes
    -----
    This function is built on top of PyTorch's QR function. ``torch.linalg.qr()`` using LAPACK on
    the backend.
    Basic information about QR factorization/decomposition can be found at
    https://en.wikipedia.org/wiki/QR_factorization. The algorithms are based on the CAQR and TSQRalgorithms. For more information see references.

    References
    ----------
    [0] W. Zheng, F. Song, L. Lin, and Z. Chen, “Scaling Up Parallel Computation of Tiled QR
    Factorizations by a Distributed Scheduling Runtime System and Analytical Modeling,”
    Parallel Processing Letters, vol. 28, no. 01, p. 1850004, 2018. \n
    [1] Bilel Hadri, Hatem Ltaief, Emmanuel Agullo, Jack Dongarra. Tile QR Factorization with
    Parallel Panel Processing for Multicore Architectures. 24th IEEE International Parallel
    and DistributedProcessing Symposium (IPDPS 2010), Apr 2010, Atlanta, United States.
    inria-00548899 \n
    [2] Gene H. Golub and Charles F. Van Loan. 1996. Matrix Computations (3rd Ed.).

    Examples
    --------
    >>> a = ht.random.randn(9, 6, split=0)
    >>> qr = ht.linalg.qr(a)
    >>> print(ht.allclose(a, ht.dot(qr.Q, qr.R)))
    [0/1] True
    [1/1] True
    >>> st = torch.randn(9, 6)
    >>> a = ht.array(st, split=1)
    >>> a_comp = ht.array(st, split=0)
    >>> q, r = ht.linalg.qr(a)
    >>> print(ht.allclose(a_comp, ht.dot(q, r)))
    [0/1] True
    [1/1] True
    """
    if not isinstance(a, DNDarray):
        raise TypeError("'a' must be a DNDarray")
    if not isinstance(tiles_per_proc, (int, torch.Tensor)):
        raise TypeError(
            "tiles_per_proc must be an int or a torch.Tensor, "
            "currently {}".format(type(tiles_per_proc))
        )
    if not isinstance(calc_q, bool):
        raise TypeError("calc_q must be a bool, currently {}".format(type(calc_q)))
    if not isinstance(overwrite_a, bool):
        raise TypeError("overwrite_a must be a bool, currently {}".format(type(overwrite_a)))
    if isinstance(tiles_per_proc, torch.Tensor):
        raise ValueError(
            "tiles_per_proc must be a single element torch.Tenor or int, "
            "currently has {} entries".format(tiles_per_proc.numel())
        )
    if len(a.shape) != 2:
        raise ValueError("Array 'a' must be 2 dimensional")

    QR = collections.namedtuple("QR", "Q, R")

    if a.split is None:
        try:
            q, r = torch.linalg.qr(a.larray, mode="complete")
        except AttributeError:
            q, r = a.larray.qr(some=False)

        q = factories.array(q, device=a.device)
        r = factories.array(r, device=a.device)
        ret = QR(q if calc_q else None, r)
        return ret
    # =============================== Prep work ====================================================
    r = a if overwrite_a else a.copy()
    # r.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    r_tiles = SquareDiagTiles(arr=r, tiles_per_proc=tiles_per_proc)
    tile_columns = r_tiles.tile_columns
    tile_rows = r_tiles.tile_rows
    if calc_q:
        q = factories.eye(
            (r.gshape[0], r.gshape[0]), split=0, dtype=r.dtype, comm=r.comm, device=r.device
        )
        q_tiles = SquareDiagTiles(arr=q, tiles_per_proc=tiles_per_proc)
        q_tiles.match_tiles(r_tiles)
    else:
        q, q_tiles = None, None
    # ==============================================================================================
    if a.split == 0:
        rank = r.comm.rank
        active_procs = torch.arange(r.comm.size, device=r.device.torch_device)
        empties = torch.nonzero(input=r_tiles.lshape_map[..., 0] == 0, as_tuple=False)
        empties = empties.flatten().tolist() if empties.numel() > 0 else []
        for e in empties:
            active_procs = active_procs[active_procs != e]
        tile_rows_per_pr_trmd = r_tiles.tile_rows_per_process[: active_procs[-1] + 1]

        q_dict = {}
        q_dict_waits = {}
        proc_tile_start = torch.cumsum(
            torch.tensor(tile_rows_per_pr_trmd, device=r.device.torch_device), dim=0
        )
        # ------------------------------------ R Calculation ---------------------------------------
        for col in range(tile_columns):
            # for each tile column (need to do the last rank separately)
            # for each process need to do local qr
            not_completed_processes = torch.nonzero(
                input=col < proc_tile_start, as_tuple=False
            ).flatten()
            if rank not in not_completed_processes or rank not in active_procs:
                # if the process is done calculating R the break the loop
                break
            diag_process = not_completed_processes[0]
            __split0_r_calc(
                r_tiles=r_tiles,
                q_dict=q_dict,
                q_dict_waits=q_dict_waits,
                diag_idx=col,
                diag_pr=diag_process,
                not_completed_prs=not_completed_processes,
            )
        # ------------------------------------- Q Calculation --------------------------------------
        if calc_q:
            for col in range(tile_columns):
                __split0_q_loop(
                    diag_idx=col,
                    r_tiles=r_tiles,
                    proc_tile_start=proc_tile_start,
                    active_procs=active_procs,
                    q0_tiles=q_tiles,
                    q_dict=q_dict,
                    q_dict_waits=q_dict_waits,
                )
    elif a.split == 1:
        # loop over the tile columns
        lp_cols = tile_columns if a.gshape[0] > a.gshape[1] else tile_rows
        for dcol in range(lp_cols):  # dcol is the diagonal column
            __split1_qr_loop(diag_idx=dcol, r_tiles=r_tiles, q0_tiles=q_tiles, calc_q=calc_q)

    r.balance_()
    if q is not None:
        q.balance_()

    ret = QR(q, r)
    return ret


DNDarray.qr: Callable[
    [DNDarray, Union[int, torch.Tensor], bool, bool], Tuple[DNDarray, DNDarray]
] = lambda self, tiles_per_proc=1, calc_q=True, overwrite_a=False: qr(
    self, tiles_per_proc, calc_q, overwrite_a
)
DNDarray.qr.__doc__ = qr.__doc__


def __split0_global_q_dict_set(
    q_dict_col: Dict,
    diag_idx: Union[int, torch.Tensor, tuple],
    r_tiles: SquareDiagTiles,
    q_tiles: SquareDiagTiles,
    global_merge_dict: Dict = None,
) -> None:
    """
    The function takes the original Q tensors from the global QR calculation and sets them to
    the keys which corresponds with their tile coordinates in Q. this returns a separate dictionary,
    it does NOT set the values of Q

    Parameters
    ----------
    q_dict_col : Dict
        The dictionary of the Q values for a given column, should be given as q_dict[col]
    diag_idx : int, tuple
        index of the tile which will be diagonalized.
        If the input is an int, it is assumed that it is the diagonal element, i.e. 4 == (4,4)
    r_tiles : SquareDiagTiles
        tiling object for ``r``
    q_tiles : SquareDiagTiles
        tiling object for Q0
    global_merge_dict : Dict, optional
        the output of the function will be in this dictionary
        Form of output: key index : torch.Tensor
    row : int, single element torch.Tensor
        the target row index of the diagonal tile, used for off-diagonal QR factorization
    """
    if isinstance(diag_idx, int):
        diag_idx = (diag_idx, diag_idx)
    # q is already created, the job of this function is to create the group the merging q's together
    # it takes the merge qs, splits them, then puts them into a new dictionary
    # steps
    proc_tile_start = torch.cumsum(
        torch.tensor(r_tiles.tile_rows_per_process, device=r_tiles.arr.larray.device), dim=0
    )
    diag_proc = torch.nonzero(input=proc_tile_start > diag_idx[0], as_tuple=False)[0].item()
    proc_tile_start = torch.cat(
        (torch.tensor([0], device=r_tiles.arr.larray.device), proc_tile_start[:-1]), dim=0
    )

    # 1: create caqr dictionary
    # need to have empty lists for all tiles in q
    global_merge_dict = {} if global_merge_dict is None else global_merge_dict

    # intended to be used as [row][column] -> data
    # 2: loop over keys in the dictionary
    merge_list = list(q_dict_col.keys())
    merge_list.sort()
    # todo: possible improvement -> make the keys have the process they are on as well,
    #  then can async get them if they are not on the diagonal process
    for key in merge_list:
        # this loops over all of the Qs for col and creates the dictionary for the pr Q merges
        p0 = key.find("p0")
        p1 = key.find("p1")
        end = key.find("e")
        r0 = int(key[p0 + 2 : p1])
        r1 = int(key[p1 + 2 : end])
        lp_q = q_dict_col[key][0]
        base_size = q_dict_col[key][1]
        # cut the q into 4 bits (end of base array)
        # todo: modify this so that it will get what is needed from the process,
        #  instead of gathering all the qs
        top_left = lp_q[: base_size[0], : base_size[0]]
        top_right = lp_q[: base_size[0], base_size[0] :]
        bottom_left = lp_q[base_size[0] :, : base_size[0]]
        bottom_right = lp_q[base_size[0] :, base_size[0] :]
        # need to adjust the keys to be the global row
        if diag_proc == r0:
            col1 = diag_idx[0]
        else:
            col1 = proc_tile_start[r0].item()
        col2 = proc_tile_start[r1].item()
        # col0 and col1 are the columns numbers
        # r0 and r1 are the ranks
        if diag_idx[0] != diag_idx[1]:
            diff = col2 - col1
            jdim = (diag_idx[0], diag_idx[1])
            kdim = (diag_idx[0], diag_idx[1] + diff)
            ldim = (diag_idx[0] + diff, diag_idx[1])
            mdim = (diag_idx[0] + diff, diag_idx[1] + diff)
        else:
            jdim = (col1, col1)
            kdim = (col1, col2)
            ldim = (col2, col1)
            mdim = (col2, col2)

        # if there are no elements on that location than set it as the tile
        # 1. get keys of what already has data
        curr_keys = set(global_merge_dict.keys())
        # 2. determine which tiles need to be touched/created
        # these are the keys which are to be multiplied by the q in the current loop
        # for matrix of form: | J  K |
        #                     | L  M |
        mult_keys_00 = [(i, col1) for i in range(q_tiles.tile_columns)]  # (J)
        # (J) -> inds: (i, col0)(col0, col0) -> set at (i, col0)
        mult_keys_01 = [(i, col1) for i in range(q_tiles.tile_columns)]  # (K)
        # (K) -> inds: (i, col0)(col0, col1) -> set at (i, col1)
        mult_keys_10 = [(i, col2) for i in range(q_tiles.tile_columns)]  # (L)
        # (L) -> inds: (i, col1)(col1, col0) -> set at (i, col0)
        mult_keys_11 = [(i, col2) for i in range(q_tiles.tile_columns)]  # (M)
        # (M) -> inds: (i, col1)(col1, col1) -> set at (i, col1)

        # if there are no elements in the mult_keys then set the element to the same place
        s00 = set(mult_keys_00) & curr_keys
        s01 = set(mult_keys_01) & curr_keys
        s10 = set(mult_keys_10) & curr_keys
        s11 = set(mult_keys_11) & curr_keys
        hold_dict = global_merge_dict.copy()

        # (J)
        if not len(s00):
            global_merge_dict[jdim] = top_left
        else:  # -> do the mm for all of the mult keys
            for k in s00:
                global_merge_dict[k[0], jdim[1]] = hold_dict[k] @ top_left
        # (K)
        if not len(s01):
            # check that we are not overwriting here
            global_merge_dict[kdim] = top_right
        else:  # -> do the mm for all of the mult keys
            for k in s01:
                global_merge_dict[k[0], kdim[1]] = hold_dict[k] @ top_right
        # (L)
        if not len(s10):
            # check that we are not overwriting here
            global_merge_dict[ldim] = bottom_left
        else:  # -> do the mm for all of the mult keys
            for k in s10:
                global_merge_dict[k[0], ldim[1]] = hold_dict[k] @ bottom_left
        # (M)
        if not len(s11):
            # check that we are not overwriting here
            global_merge_dict[mdim] = bottom_right
        else:  # -> do the mm for all of the mult keys
            for k in s11:
                global_merge_dict[k[0], mdim[1]] = hold_dict[k] @ bottom_right
        del hold_dict
    return global_merge_dict


def __split0_r_calc(
    r_tiles: SquareDiagTiles,
    q_dict: Dict,
    q_dict_waits: Dict,
    diag_idx: Union[int, Tuple],
    diag_pr: int,
    not_completed_prs: torch.Tensor,
) -> None:
    """
    This function uses a binary merge structure in the global R merge.

    Parameters
    ----------
    r_tiles : SquareDiagTiles
        Tiling object for ``r``
    q_dict : Dict
        Dictionary to save the calculated Q matrices to
    q_dict_waits : Dict
        Dictionary to save the calculated Q matrices to which are
        not calculated on the diagonal process
    diag_idx : int, tuple
        index of the tile which will be diagonalized.
        If the input is an int, it is assumed that it is the diagonal element, i.e. 4 == (4,4)
    diag_pr : int
        Rank of the process which has the tile which lies along the diagonal
    not_completed_prs : torch.Tensor
        Tensor of the processes which have not yet finished calculating R
    """
    if isinstance(diag_idx, int):
        diag_idx = (diag_idx, diag_idx)
    tile_rows_proc = r_tiles.tile_rows_per_process
    comm = r_tiles.arr.comm
    rank = comm.rank
    lcl_tile_row = 0 if rank != diag_pr else diag_idx[0] - sum(tile_rows_proc[:rank])
    # only work on the processes which have not computed the final result
    q_dict[diag_idx[1]] = {}
    q_dict_waits[diag_idx[1]] = {}

    # --------------- local QR calc -----------------------------------------------------
    base_tile = r_tiles.local_get(key=(slice(lcl_tile_row, None), diag_idx[1]))
    try:
        q1, r1 = torch.linalg.qr(base_tile, mode="complete")
    except AttributeError:
        q1, r1 = base_tile.qr(some=False)

    q_dict[diag_idx[1]]["l0"] = [q1, base_tile.shape]
    r_tiles.local_set(key=(slice(lcl_tile_row, None), diag_idx[1]), value=r1)
    if diag_idx[1] != r_tiles.tile_columns - 1:
        base_rest = r_tiles.local_get((slice(lcl_tile_row, None), slice(diag_idx[1] + 1, None)))
        loc_rest = torch.matmul(q1.T, base_rest)
        r_tiles.local_set(
            key=(slice(lcl_tile_row, None), slice(diag_idx[1] + 1, None)), value=loc_rest
        )
    # --------------- global QR calc (binary merge) -------------------------------------
    rem1 = None
    rem2 = None
    offset = not_completed_prs[0]
    loop_size_remaining = not_completed_prs.clone()
    completed = False if loop_size_remaining.size()[0] > 1 else True
    procs_remaining = loop_size_remaining.size()[0]
    loop = 0
    # leftover = []
    while not completed:
        if procs_remaining % 2 == 1:
            # if the number of processes active is odd need to save the remainders
            if rem1 is None:
                rem1 = loop_size_remaining[-1]
                loop_size_remaining = loop_size_remaining[:-1]
            elif rem2 is None:
                rem2 = loop_size_remaining[-1]
                loop_size_remaining = loop_size_remaining[:-1]
        if rank not in loop_size_remaining and rank not in [rem1, rem2]:
            break  # if the rank is done then exit the loop
        # send the data to the corresponding processes
        try:
            half_prs_rem = torch.div(procs_remaining, 2, rounding_mode="floor")
        except TypeError:  # torch 1.7 version
            half_prs_rem = torch.floor_divide(procs_remaining, 2)

        zipped = zip(
            loop_size_remaining.flatten()[:half_prs_rem],
            loop_size_remaining.flatten()[half_prs_rem:],
        )
        for pr in zipped:
            pr0, pr1 = int(pr[0].item()), int(pr[1].item())
            #             if pr1 not in leftover:
            __split0_merge_tile_rows(
                pr0=pr0,
                pr1=pr1,
                diag_idx=diag_idx,
                rank=rank,
                r_tiles=r_tiles,
                diag_process=diag_pr,
                key=str(loop) + "p0" + str(pr0) + "p1" + str(pr1) + "e",
                q_dict=q_dict,
            )
            __split0_send_q_to_diag_pr(
                col=diag_idx[1],
                pr0=pr0,
                pr1=pr1,
                diag_process=diag_pr,
                comm=comm,
                q_dict=q_dict,
                key=str(loop) + "p0" + str(pr0) + "p1" + str(pr1) + "e",
                q_dict_waits=q_dict_waits,
                q_dtype=r_tiles.arr.dtype.torch_type(),
                q_device=r_tiles.arr._DNDarray__array.device,
            )

        loop_size_remaining = loop_size_remaining[: -1 * half_prs_rem]
        procs_remaining = loop_size_remaining.size()[0]

        if rem1 is not None and rem2 is not None:
            # combine rem1 and rem2 in the same way as the other nodes,
            # then save the results in rem1 to be used later
            rem1, rem2 = int(rem1), int(rem2)
            __split0_merge_tile_rows(
                pr0=rem2,
                pr1=rem1,
                diag_idx=diag_idx,
                rank=rank,
                r_tiles=r_tiles,
                diag_process=diag_pr,
                key=str(loop) + "p0" + str(rem1) + "p1" + str(rem2) + "e",
                q_dict=q_dict if q_dict is not None else {},
            )
            __split0_send_q_to_diag_pr(
                col=diag_idx[1],
                pr0=rem2,
                pr1=rem1,
                diag_process=diag_pr,
                key=str(loop) + "p0" + str(rem1) + "p1" + str(rem2) + "e",
                q_dict=q_dict if q_dict is not None else {},
                comm=comm,
                q_dict_waits=q_dict_waits,
                q_dtype=r_tiles.arr.dtype.torch_type(),
                q_device=r_tiles.arr.larray.device,
            )
            rem1 = rem2
            rem2 = None

        loop += 1
        if rem1 is not None and rem2 is None and procs_remaining == 1:
            # combine rem1 with process 0 (offset) and set completed to True
            # this should be the last thing that happens
            offset, rem1 = int(offset), int(rem1)
            __split0_merge_tile_rows(
                pr0=offset,
                pr1=rem1,
                diag_idx=diag_idx,
                rank=rank,
                r_tiles=r_tiles,
                diag_process=diag_pr,
                key=str(loop) + "p0" + str(offset) + "p1" + str(rem1) + "e",
                q_dict=q_dict,
            )
            __split0_send_q_to_diag_pr(
                col=diag_idx[1],
                pr0=offset,
                pr1=rem1,
                diag_process=diag_pr,
                key=str(loop) + "p0" + str(offset) + "p1" + str(rem1) + "e",
                q_dict=q_dict,
                comm=comm,
                q_dict_waits=q_dict_waits,
                q_dtype=r_tiles.arr.dtype.torch_type(),
                q_device=r_tiles.arr.larray.device,
            )
            rem1 = None
        completed = True if procs_remaining == 1 and rem1 is None and rem2 is None else False


def __split0_merge_tile_rows(
    pr0: int,
    pr1: int,
    diag_idx: Union[int, Tuple],
    rank: int,
    r_tiles: SquareDiagTiles,
    diag_process: int,
    key: str,
    q_dict: Dict,
) -> None:
    """
    Sets the value of ``q_dict[column][key]`` with ``[Q, upper.shape, lower.shape]``
    Merge two tile rows, take their QR, and apply it to the trailing process
    This will modify ``a`` and set the value of the ``q_dict[column][key]``
    with ``[Q, upper.shape, lower.shape]``.

    Parameters
    ----------
    pr0, pr1 : int, int
        Process ranks of the processes to be used
    diag_idx : int, tuple
        index of the tile which will be diagonalized.
        If the input is an int, it is assumed that it is the diagonal element, i.e. 4 == (4,4)
    rank : int
        The rank of the process
    r_tiles : SquareDiagTiles
        Tiling object used for getting/setting the tiles required
    diag_process : int
        The rank of the process which has the tile along the diagonal for the given column
    key: str
        where to set the value in the Q dictionary
    q_dict: dict
        the dictionary which holds the results for Q
        end result -> function sets q_dict[column][key] with [Q, upper.shape, lower.shape]

    Returns
    -------
    None, sets the value of q_dict[column][key] with [Q, upper.shape, lower.shape]
    """
    if rank not in [pr0, pr1]:
        return
    if isinstance(diag_idx, int):
        diag_idx = (diag_idx, diag_idx)
    pr0 = pr0.item() if isinstance(pr0, torch.Tensor) else pr0
    pr1 = pr1.item() if isinstance(pr1, torch.Tensor) else pr1
    comm = r_tiles.arr.comm
    upper_row = sum(r_tiles.tile_rows_per_process[:pr0]) if pr0 != diag_process else diag_idx[0]
    lower_row = sum(r_tiles.tile_rows_per_process[:pr1]) if pr1 != diag_process else diag_idx[0]

    upper_inds = r_tiles.get_start_stop(key=(upper_row, diag_idx[1]))
    lower_inds = r_tiles.get_start_stop(key=(lower_row, diag_idx[1]))

    upper_size = (upper_inds[1] - upper_inds[0], upper_inds[3] - upper_inds[2])
    lower_size = (lower_inds[1] - lower_inds[0], lower_inds[3] - lower_inds[2])

    a_torch_device = r_tiles.arr.larray.device

    # upper adjustments
    if upper_size[0] < upper_size[1]:
        if r_tiles.tile_rows_per_process[pr0] > 1:
            # end of dim0 (upper_inds[1]) is equal to the size in dim1
            upper_inds = list(upper_inds)
            upper_inds[1] = upper_inds[0] + upper_size[1]
            upper_size = (upper_inds[1] - upper_inds[0], upper_inds[3] - upper_inds[2])
        if r_tiles.tile_rows_per_process[pr1] > 1:
            # end of dim0 (upper_inds[1]) is equal to the size in dim1
            lower_inds = list(lower_inds)
            lower_inds[1] = lower_inds[0] + lower_size[1]
            lower_size = (lower_inds[1] - lower_inds[0], lower_inds[3] - lower_inds[2])

    if rank == pr0:
        # need to use lloc on r_tiles.arr with the indices
        upper = r_tiles.arr.lloc[upper_inds[0] : upper_inds[1], upper_inds[2] : upper_inds[3]]

        comm.Send(upper.clone(), dest=pr1, tag=986)
        lower = torch.zeros(lower_size, dtype=r_tiles.arr.dtype.torch_type(), device=a_torch_device)
        comm.Recv(lower, source=pr1, tag=4363)
    else:  # rank == pr1:
        lower = r_tiles.arr.lloc[lower_inds[0] : lower_inds[1], lower_inds[2] : lower_inds[3]]
        upper = torch.zeros(upper_size, dtype=r_tiles.arr.dtype.torch_type(), device=a_torch_device)
        comm.Recv(upper, source=pr0, tag=986)
        comm.Send(lower.clone(), dest=pr0, tag=4363)

    try:
        q_merge, r = torch.linalg.qr(torch.cat((upper, lower), dim=0), mode="complete")
    except AttributeError:
        q_merge, r = torch.cat((upper, lower), dim=0).qr(some=False)

    upp = r[: upper.shape[0]]
    low = r[upper.shape[0] :]
    if rank == pr0:
        r_tiles.arr.lloc[upper_inds[0] : upper_inds[1], upper_inds[2] : upper_inds[3]] = upp
    else:  # rank == pr1:
        r_tiles.arr.lloc[lower_inds[0] : lower_inds[1], lower_inds[2] : lower_inds[3]] = low

    if diag_idx[1] < r_tiles.tile_columns - 1:
        upper_rest_size = (upper_size[0], r_tiles.arr.gshape[1] - upper_inds[3])
        lower_rest_size = (lower_size[0], r_tiles.arr.gshape[1] - lower_inds[3])

        if rank == pr0:
            upper_rest = r_tiles.arr.lloc[upper_inds[0] : upper_inds[1], upper_inds[3] :]
            lower_rest = torch.zeros(
                lower_rest_size, dtype=r_tiles.arr.dtype.torch_type(), device=a_torch_device
            )
            comm.Send(upper_rest.clone(), dest=pr1, tag=98654)
            comm.Recv(lower_rest, source=pr1, tag=436364)
        else:  # rank == pr1:
            lower_rest = r_tiles.arr.lloc[lower_inds[0] : lower_inds[1], lower_inds[3] :]
            upper_rest = torch.zeros(
                upper_rest_size, dtype=r_tiles.arr.dtype.torch_type(), device=a_torch_device
            )
            comm.Recv(upper_rest, source=pr0, tag=98654)
            comm.Send(lower_rest.clone(), dest=pr0, tag=436364)

        cat_tensor = torch.cat((upper_rest, lower_rest), dim=0)
        new_rest = torch.matmul(q_merge.t(), cat_tensor)
        # the data for upper rest is a slice of the new_rest, need to slice only the 0th dim
        upp = new_rest[: upper_rest.shape[0]]
        low = new_rest[upper_rest.shape[0] :]
        if rank == pr0:
            r_tiles.arr.lloc[upper_inds[0] : upper_inds[1], upper_inds[3] :] = upp
        # set the lower rest
        else:  # rank == pr1:
            r_tiles.arr.lloc[lower_inds[0] : lower_inds[1], lower_inds[3] :] = low
    q_dict[diag_idx[1]][key] = [q_merge, upper.shape, lower.shape]


def __split0_send_q_to_diag_pr(
    col: int,
    pr0: int,
    pr1: int,
    diag_process: int,
    comm: MPICommunication,
    q_dict: Dict,
    key: str,
    q_dict_waits: Dict,
    q_dtype: Type[datatype],
    q_device: torch.device,
) -> None:
    """
    Sets the values of ``q_dict_waits`` with the with *waits* for the values of Q, ``upper.shape``,
    and ``lower.shape``
    This function sends the merged Q to the diagonal process. Buffered send it used for sending
    Q. This is needed for the Q calculation when two processes are merged and neither is the diagonal
    process.

    Parameters
    ----------
    col : int
        The current column used in the parent QR loop
    pr0, pr1 : int, int
        Rank of processes 0 and 1. These are the processes used in the calculation of q
    diag_process : int
        The rank of the process which has the tile along the diagonal for the given column
    comm : MPICommunication (ht.DNDarray.comm)
        The communicator used. (Intended as the communication of ``a`` given to qr)
    q_dict : Dict
        Dictionary containing the Q values calculated for finding R
    key : str
        Key for ``q_dict[col]`` which corresponds to the Q to send
    q_dict_waits : Dict
        Dictionary used in the collection of the Qs which are sent to the diagonal process
    q_dtype : torch.type
        Type of the Q tensor
    q_device : torch.device
        Torch device of the Q tensor
    """
    if comm.rank not in [pr0, pr1, diag_process] or pr0 == diag_process:
        return
    # this is to send the merged q to the diagonal process for the forming of q
    base_tag = "1" + str(pr1.item() if isinstance(pr1, torch.Tensor) else pr1)
    if comm.rank == pr1:
        q = q_dict[col][key][0]
        u_shape = q_dict[col][key][1]
        l_shape = q_dict[col][key][2]
        comm.send(tuple(q.shape), dest=diag_process, tag=int(base_tag + "1"))
        comm.Isend(q, dest=diag_process, tag=int(base_tag + "12"))
        comm.send(u_shape, dest=diag_process, tag=int(base_tag + "123"))
        comm.send(l_shape, dest=diag_process, tag=int(base_tag + "1234"))
    if comm.rank == diag_process:
        # q_dict_waits now looks like a
        q_sh = comm.recv(source=pr1, tag=int(base_tag + "1"))
        q_recv = torch.zeros(q_sh, dtype=q_dtype, device=q_device)
        k = "p0" + str(pr0) + "p1" + str(pr1)
        q_dict_waits[col][k] = []
        q_wait = comm.Irecv(q_recv, source=pr1, tag=int(base_tag + "12"))
        q_dict_waits[col][k].append([q_recv, q_wait])
        q_dict_waits[col][k].append(comm.irecv(source=pr1, tag=int(base_tag + "123")))
        q_dict_waits[col][k].append(comm.irecv(source=pr1, tag=int(base_tag + "1234")))
        q_dict_waits[col][k].append(key[0])


def __split0_q_loop(
    diag_idx: Union[int, tuple],
    r_tiles: SquareDiagTiles,
    proc_tile_start: torch.Tensor,
    active_procs: torch.Tensor,
    q0_tiles: SquareDiagTiles,
    q_dict: Dict,
    q_dict_waits: Dict,
) -> None:
    """
    Function for Calculating Q for ``split=0`` for QR. ``col`` is the index of the tile column.
    The assumption here is that the diagonal tile is ``(col, col)``.

    Parameters
    ----------
    diag_idx : int, tuple
        index of the tile which will be diagonalized.
        If the input is an int, it is assumed that it is the diagonal element, i.e. 4 == (4,4)
    r_tiles : SquareDiagTiles
        R tiles
    proc_tile_start : torch.Tensor
        Tensor containing the row tile start indices for each process
    active_procs : torch.Tensor
        Tensor containing the ranks of processes with have data
    q0_tiles : tiling.SquareDiagTiles
        tiling object for 'Q0'
    q_dict : Dictionary
        Dictionary created in the split=0 R calculation containing all of the Q matrices found
        transforming the matrix to upper triangular for each column. The keys of this dictionary are
        the column indices
    q_dict_waits : Dict
        Dictionary created while sending the Q matrices to the diagonal process

    Returns
    -------
    None
    """
    if isinstance(diag_idx, int):
        diag_idx = (diag_idx, diag_idx)
    tile_columns = r_tiles.tile_columns
    if diag_idx[0] != tile_columns:
        diag_process = torch.nonzero(input=proc_tile_start > diag_idx[0], as_tuple=False)
        if diag_process.numel() == 0:
            return
        else:
            diag_process = diag_process[0]
    else:
        diag_process = proc_tile_start[-1]

    comm = r_tiles.arr.comm
    rank = r_tiles.arr.comm.rank
    q0_dtype = q0_tiles.arr.dtype
    q0_torch_type = q0_dtype.torch_type()
    q0_torch_device = q0_tiles.arr.device.torch_device
    # wait for Q tensors sent during the R calculation -----------------------------------------
    if diag_idx[1] in q_dict_waits.keys():
        for key in q_dict_waits[diag_idx[1]].keys():
            new_key = q_dict_waits[diag_idx[1]][key][3] + key + "e"
            q_dict_waits[diag_idx[1]][key][0][1].wait()
            q_dict[diag_idx[1]][new_key] = [
                q_dict_waits[diag_idx[1]][key][0][0],
                q_dict_waits[diag_idx[1]][key][1].wait(),
                q_dict_waits[diag_idx[1]][key][2].wait(),
            ]
        del q_dict_waits[diag_idx[1]]
    # local Q calculation =====================================================================
    if diag_idx[1] in q_dict.keys():
        lcl_col_shape = r_tiles.local_get(key=(slice(None), diag_idx[1])).shape
        # get the start and stop of all local tiles
        #   -> get the rows_per_process[rank] and the row_indices
        row_ind = r_tiles.row_indices
        prev_rows_per_pr = sum(r_tiles.tile_rows_per_process[:rank])
        rows_per_pr = r_tiles.tile_rows_per_process[rank]
        if rows_per_pr == 1:
            # if there is only one tile on the process: return q_dict[col]['0']
            base_q = q_dict[diag_idx[1]]["l0"][0].clone()
            del q_dict[diag_idx[1]]["l0"]
        else:
            # 0. get the offset of the column start
            offset = (
                torch.tensor(
                    row_ind[diag_idx[0]] - row_ind[prev_rows_per_pr], device=q0_torch_device
                )
                if row_ind[diag_idx[0]] > row_ind[prev_rows_per_pr]
                else torch.tensor(0, device=q0_torch_device)
            )
            # 1: create an eye matrix of the row's zero'th dim^2
            q_lcl = q_dict[diag_idx[1]][
                "l0"
            ]  # [0] -> q, [1] -> shape of a use in q calc (q is square)
            del q_dict[diag_idx[1]]["l0"]
            base_q = torch.eye(
                lcl_col_shape[r_tiles.arr.split], dtype=q_lcl[0].dtype, device=q0_torch_device
            )
            # 2: set the area of the eye as Q
            base_q[offset : offset + q_lcl[1][0], offset : offset + q_lcl[1][0]] = q_lcl[0]

        local_merge_q = {rank: [base_q, None]}
    else:
        local_merge_q = {}
    # -------------- send local Q to all -------------------------------------------------------
    for pr in range(diag_process, active_procs[-1] + 1):
        if pr != rank:
            hld = torch.zeros(
                [q0_tiles.lshape_map[pr][q0_tiles.arr.split]] * 2,
                dtype=q0_torch_type,
                device=q0_torch_device,
            )
        else:
            hld = local_merge_q[pr][0].clone()
        wait = q0_tiles.arr.comm.Ibcast(hld, root=pr)
        local_merge_q[pr] = [hld, wait]

    # recv local Q + apply local Q to Q0 (from right)
    for pr in range(diag_process, active_procs[-1] + 1):
        if local_merge_q[pr][1] is not None:
            # receive q from the other processes
            local_merge_q[pr][1].Wait()
        if rank in active_procs:
            sum_row = sum(q0_tiles.tile_rows_per_process[:pr])
            end_row = q0_tiles.tile_rows_per_process[pr] + sum_row
            # slice of q_tiles -> [0: -> end local, 1: start -> stop]
            q_rest_loc = q0_tiles.local_get(key=(slice(None), slice(sum_row, end_row)))
            # apply the local merge to q0 then update q0`
            q_rest_loc = q_rest_loc @ local_merge_q[pr][0]
            q0_tiles.local_set(key=(slice(None), slice(sum_row, end_row)), value=q_rest_loc)
            del local_merge_q[pr]

    # global Q calculation =====================================================================
    # split up the Q's from the global QR calculation and set them in a dict w/ proper keys
    global_merge_dict = (
        __split0_global_q_dict_set(
            q_dict_col=q_dict[diag_idx[1]],
            diag_idx=(diag_idx[0], diag_idx[0]),
            r_tiles=r_tiles,
            q_tiles=q0_tiles,
        )
        if rank == diag_process
        else {}
    )

    if rank == diag_process:
        merge_dict_keys = set(global_merge_dict.keys())
    else:
        merge_dict_keys = None
    merge_dict_keys = comm.bcast(merge_dict_keys, root=diag_process)

    # send the global merge dictionary to all processes
    for k in merge_dict_keys:
        if rank == diag_process:
            snd = global_merge_dict[k].clone()
            snd_shape = snd.shape
            comm.bcast(snd_shape, root=diag_process)
        else:
            snd_shape = None
            snd_shape = comm.bcast(snd_shape, root=diag_process)
            snd = torch.empty(
                snd_shape, dtype=q0_tiles.arr.dtype.torch_type(), device=q0_torch_device
            )

        wait = comm.Ibcast(snd, root=diag_process)
        global_merge_dict[k] = [snd, wait]
    if rank in active_procs:
        # create a dictionary which says what tiles are in each column of the global merge Q
        qi_mult = {}
        for c in range(q0_tiles.tile_columns):
            # this loop is to slice the merge_dict keys along each column + create the
            qi_mult_set = set([(i, c) for i in range(diag_idx[1], q0_tiles.tile_columns)])
            if len(qi_mult_set & merge_dict_keys) != 0:
                qi_mult[c] = list(qi_mult_set & merge_dict_keys)

        # have all the q_merge in one place, now just do the mm with q0
        # get all the keys which are in a column (qi_mult[column])
        row_inds = q0_tiles.row_indices + [q0_tiles.arr.gshape[0]]
        q_copy = q0_tiles.arr.larray.clone()
        for qi_col in qi_mult.keys():
            # multiply q0 rows with qi cols
            # the result of this will take the place of the row height and the column width
            out_sz = q0_tiles.local_get(key=(slice(None), qi_col)).shape
            mult_qi_col = torch.zeros(
                (q_copy.shape[1], out_sz[1]),
                dtype=q0_tiles.arr.dtype.torch_type(),
                device=q0_torch_device,
            )
            for ind in qi_mult[qi_col]:
                if global_merge_dict[ind][1] is not None:
                    global_merge_dict[ind][1].Wait()
                lp_q = global_merge_dict[ind][0]
                if mult_qi_col.shape[1] < lp_q.shape[1]:
                    new_mult = torch.zeros(
                        (mult_qi_col.shape[0], lp_q.shape[1]),
                        dtype=mult_qi_col.dtype,
                        device=q0_torch_device,
                    )
                    new_mult[:, : mult_qi_col.shape[1]] += mult_qi_col.clone()
                    mult_qi_col = new_mult

                mult_qi_col[
                    row_inds[ind[0]] : row_inds[ind[0]] + lp_q.shape[0], : lp_q.shape[1]
                ] = lp_q
            hold = torch.matmul(q_copy, mult_qi_col)

            write_inds = q0_tiles.get_start_stop(key=(0, qi_col))
            q0_tiles.arr.lloc[:, write_inds[2] : write_inds[2] + hold.shape[1]] = hold
    else:
        for ind in merge_dict_keys:
            global_merge_dict[ind][1].wait()
    if diag_idx[1] in q_dict.keys():
        del q_dict[diag_idx[1]]


def __split1_qr_loop(
    diag_idx: Union[int, tuple],
    r_tiles: SquareDiagTiles,
    q0_tiles: SquareDiagTiles,
    calc_q: bool,
    empties=None,
):
    """
    Helper function to do the QR factorization of the column 'diag_col'. This function assumes
    that the target tile is at (diag_col, diag_col). This is the standard case at it assumes that the
    diagonal tile holds the diagonal entries of the matrix.

    Parameters
    ----------
    diag_idx : int, tuple
        index of the tile which will be diagonalized.
        If the input is an int, it is assumed that it is the diagonal element, i.e. 4 == (4,4)
    r_tiles : tiling.SquareDiagTiles
        tiling object for 'r'
    q0_tiles : tiling.SquareDiagTiles
        tiling object for 'q0'
    calc_q : Boolean
        Flag for weather to calculate Q or not, if False, then Q=None
    empties : torch.Tensor, optional
        list of the processes without data, if None, it will be generated
    """
    if isinstance(diag_idx, int):
        diag_idx = (diag_idx, diag_idx)
    if empties is None:
        # this will return the empty processes, Requires that lshape map is not none
        # todo: should this be the q tiles or the r tiles???
        empties = torch.nonzero(input=r_tiles.lshape_map[..., 0] == 0, as_tuple=False)
        empties = empties.flatten().tolist() if empties.numel() > 0 else []
    r_torch_device = r_tiles.arr.device.torch_device
    r_torch_type = r_tiles.arr.dtype.torch_type()
    q0_torch_device = q0_tiles.arr.device.torch_device if calc_q else None

    # ==================================== R Calculation - single tile =========================
    # loop over each column, need to do the QR for each tile in the column(should be rows)
    # need to get the diagonal process
    comm = r_tiles.arr.comm
    rank = comm.rank
    # find out which tile columns are on which ranks
    cols_on_proc = torch.cumsum(
        torch.tensor(r_tiles.tile_columns_per_process, device=r_torch_device), dim=0
    )
    # get a list of which processes must continue working
    not_completed_processes = torch.nonzero(
        input=diag_idx[1] < cols_on_proc, as_tuple=False
    ).flatten()
    diag_pr = not_completed_processes[0].item()
    tile_rows = r_tiles.tile_rows
    # get the diagonal tile and do qr on it
    # send q to the other processes
    # 1st qr: only on diagonal tile + apply to the row
    st_sp = r_tiles.get_start_stop(key=diag_idx)
    sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
    q1 = torch.zeros((sz[0], sz[0]), dtype=r_torch_type, device=r_torch_device)
    if rank == diag_pr:
        # do qr on diagonal process
        try:  # try/except here if for torch 1.7 compatibility
            qrank, r1 = torch.linalg.qr(r_tiles[diag_idx], mode="complete")
        except AttributeError:
            qrank, r1 = r_tiles[diag_idx].qr(some=False)

        q1 += qrank  # adding to a tensor of zeros to keep the memory object in-place (MPI)
        # send Q from the diagonal to all processes
        comm.Bcast(q1.clone(), root=diag_pr)
        r_tiles[diag_idx] = r1

        # apply q1 to the trailing matrix (other processes)
        # need to convert target column to a local tile index
        loc_col = diag_idx[1] - sum(r_tiles.tile_columns_per_process[:rank])
        # get all the tiles in the row, example -> [4, 4:]
        hold = r_tiles.local_get(key=(diag_idx[0], slice(loc_col + 1, None)))
        if hold is not None:  # if there is more data on that row after the diagonal tile
            r_tiles.local_set(
                key=(diag_idx[0], slice(loc_col + 1, None)), value=torch.matmul(q1.T, hold)
            )
    elif rank in empties or rank < diag_pr:
        # if the rank is empty (no data) or is lower than the diagonal process, the values are
        # already zeros, but we still need to catch q1 from the diagonal process
        comm.Bcast(q1, root=diag_pr)
    else:  # rank > diag_pr:
        # update the trailing matrix and then do q calc
        # receive the Q for the diagonal tile on all the non-empty processes
        comm.Bcast(q1, root=diag_pr)

        slices = r_tiles.local_to_global(key=(diag_idx[0], slice(0, None)), rank=rank)
        hold = r_tiles[slices]
        # apply the q1 from the diagonal process to the trailing matrix row
        r_tiles[slices] = torch.matmul(q1.T, hold)
    # ================================ Q Calculation - single tile =============================
    if calc_q:
        for row in range(q0_tiles.tile_rows_per_process[rank]):
            # q1 is applied to each tile of the column dcol of q0 then written there
            q0_tiles.local_set(
                key=(row, diag_idx[0]),
                value=torch.matmul(q0_tiles.local_get(key=(row, diag_idx[0])), q1),
            )
    del q1
    # loop over the rest of the rows, combine the tiles, then apply the result to the rest
    # 2nd step: merged QR on the rows
    # ================================ R Calculation - merged tiles ============================
    # this section is to eliminate the rows below the diagonal
    # the following for loop will concatenate two tiles, do a qr factorization to eliminate all
    #   the values in the lower tiles, then apply the q value to the whole row
    diag_tile = r_tiles[diag_idx]
    # (Q) need to get the start stop of diag tile
    diag_st_sp = r_tiles.get_start_stop(key=diag_idx)
    diag_sz = diag_st_sp[1] - diag_st_sp[0], diag_st_sp[3] - diag_st_sp[2]
    for row in range(diag_idx[0] + 1, tile_rows):
        lp_st_sp = r_tiles.get_start_stop(key=(row, diag_idx[1]))
        lp_sz = lp_st_sp[1] - lp_st_sp[0], lp_st_sp[3] - lp_st_sp[2]
        if rank == diag_pr:
            # cat diag tile and loop tile
            loop_tile = r_tiles[row, diag_idx[1]]
            loop_cat = torch.cat((diag_tile, loop_tile), dim=0)
            # qr
            try:
                ql, rl = torch.linalg.qr(loop_cat, mode="complete")
            except AttributeError:
                ql, rl = loop_cat.qr(some=False)
            # send ql to all
            comm.Bcast(ql.clone().contiguous(), root=diag_pr)
            # set 'r's
            r_tiles[diag_idx] = rl[: diag_sz[0]]
            r_tiles[row, diag_idx[1]] = rl[diag_sz[0] :]
            # apply q to remaining rows
            if loc_col + 1 < r_tiles.tile_columns_per_process[rank]:
                upp = r_tiles.local_get(key=(diag_idx[0], slice(loc_col + 1, None)))
                low = r_tiles.local_get(key=(row, slice(loc_col + 1, None)))
                hold = torch.matmul(ql.T, torch.cat((upp, low), dim=0))
                # set upper
                r_tiles.local_set(
                    key=(diag_idx[0], slice(loc_col + 1, None)), value=hold[: diag_sz[0]]
                )
                # set lower
                r_tiles.local_set(key=(row, slice(loc_col + 1, None)), value=hold[diag_sz[0] :])
            # todo: is this handled by the lp_sz being outside the loop?
            if len(empties) > 0:
                # send the shape to the empty process
                for i in empties:
                    comm.isend(ql.shape, dest=i, tag=222)
        elif rank in empties:
            sz = comm.recv(source=diag_pr, tag=222)
            ql = torch.zeros((sz[0], sz[0]), dtype=r_torch_type, device=r_torch_device)
            comm.Bcast(ql, root=diag_pr)
        elif rank > diag_pr:
            ql = torch.zeros([lp_sz[0] + diag_sz[0]] * 2, dtype=r_torch_type, device=r_torch_device)
            comm.Bcast(ql, root=diag_pr)
            upp = r_tiles.local_get(key=(diag_idx[0], slice(0, None)))
            low = r_tiles.local_get(key=(row, slice(0, None)))
            hold = torch.matmul(ql.T, torch.cat((upp, low), dim=0))
            # set upper
            r_tiles.local_set(key=(diag_idx[0], slice(0, None)), value=hold[: diag_sz[0]])
            # set lower
            r_tiles.local_set(key=(row, slice(0, None)), value=hold[diag_sz[0] :])
        else:
            ql = torch.zeros([lp_sz[0] + diag_sz[0]] * 2, dtype=r_torch_type, device=r_torch_device)
            comm.Bcast(ql, root=diag_pr)
        # ================================ Q Calculation - merged tiles ========================
        if calc_q:
            top_left = ql[: diag_sz[0], : diag_sz[0]]
            top_right = ql[: diag_sz[0], diag_sz[0] :]
            bottom_left = ql[diag_sz[0] :, : diag_sz[0]]
            bottom_right = ql[diag_sz[0] :, diag_sz[0] :]
            # two multiplications: one for the left tiles and one for the right
            # left tiles --------------------------------------------------------------------
            # create r column of the same size as the tile row of q0
            st_sp = r_tiles.get_start_stop(key=(slice(diag_idx[0], None), diag_idx[1]))
            qloop_col_left_sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
            qloop_col_left = torch.zeros(
                qloop_col_left_sz, dtype=q0_tiles.arr.dtype.torch_type(), device=q0_torch_device
            )
            # top left starts at 0 and goes until diag_sz[1]
            # print(dim0, dim1, row)
            qloop_col_left[: diag_sz[0]] = top_left
            # bottom left starts at ? and goes until ? (only care about 0th dim)
            st, sp, _, _ = r_tiles.get_start_stop(key=(row, 0))
            st -= diag_st_sp[0]  # adjust these by subtracting the start index of the diag tile
            sp -= diag_st_sp[0]
            qloop_col_left[st:sp] = bottom_left
            # right tiles --------------------------------------------------------------------
            # create r columns tensor of the size of the tile column of index 'row'
            st_sp = q0_tiles.get_start_stop(key=(row, slice(diag_idx[0], None)))
            # print(st_sp, row, dim0, q0_tiles.col_indices)
            sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
            qloop_col_right = torch.zeros(
                (sz[1], sz[0]), dtype=q0_tiles.arr.dtype.torch_type(), device=q0_torch_device
            )
            # top left starts at 0 and goes until diag_sz[1]
            qloop_col_right[: diag_sz[0]] = top_right
            # bottom left starts at ? and goes until ? (only care about 0th dim)
            st, sp, _, _ = r_tiles.get_start_stop(key=(row, 0))
            st -= diag_st_sp[0]  # adjust these by subtracting the start index of the diag tile
            sp -= diag_st_sp[0]
            qloop_col_right[st:sp] = bottom_right
            for qrow in range(q0_tiles.tile_rows_per_process[rank]):
                # q1 is applied to each tile of the column dcol of q0 then written there
                q0_row = q0_tiles.local_get(key=(qrow, slice(diag_idx[0], None))).clone()
                q0_tiles.local_set(
                    key=(qrow, diag_idx[0]), value=torch.matmul(q0_row, qloop_col_left)
                )
                q0_tiles.local_set(key=(qrow, row), value=torch.matmul(q0_row, qloop_col_right))
        del ql
