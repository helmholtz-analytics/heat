import torch

from .communication import MPI
from . import communication
from . import dndarray
from . import factories
from . import manipulations
from . import types


__all__ = [
    'mm_tiles',
    'SquareDiagTiles'
]


def mm_tiles(arr):
    if not isinstance(arr, dndarray.DNDarray):
        raise TypeError('arr must be a DNDarray, is currently a {}'.format(type(arr)))

    lshape_map = torch.zeros((arr.comm.size, len(arr.gshape)), dtype=int)
    lshape_map[arr.comm.rank, :] = torch.Tensor(arr.lshape)
    arr.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)
    return None, lshape_map


class SquareDiagTiles:
    # designed for QR tile scheme
    def __init__(self, arr, tile_rows=2):
        # tile_rows => in-process (local) divisions along the split axis
        # lshape_map -> rank (int), lshape (tuple of the local lshape, self.lshape)
        # if not isinstance(self, DNDarray):
        #     raise TypeError('self must be a DNDarray, is currently a {}'.format(type(self)))

        # todo: unbalance the array if there is *only* one row/column of the diagonal on a process (send it to pr - 1)

        lshape_map = torch.zeros((arr.comm.size, len(arr.gshape)), dtype=int)
        lshape_map[arr.comm.rank, :] = torch.Tensor(arr.lshape)
        arr.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

        # chunk map
        # is the diagonal crossed by a division between processes/where
        last_diag_pr = torch.where(lshape_map[..., arr.split].cumsum(dim=0) >= min(arr.gshape))[0][0]
        # adjust for small blocks on the last diag pr:
        rem_cols_last_pr = min(arr.gshape) - lshape_map[..., arr.split].cumsum(dim=0)[last_diag_pr - 1]  # end of the process before the split
        last_tile_cols = tile_rows
        # print(rem_cols_last_pr, last_tile_cols)
        while rem_cols_last_pr / last_tile_cols < 10:
            # if there cannot be tiles formed which are at list ten items large then need to reduce the number of tiles
            last_tile_cols -= 1
            if last_tile_cols == 1:
                break

        # need to determine the proper number of tile rows
        tile_columns = tile_rows * last_diag_pr + last_tile_cols
        diag_crossings = lshape_map[..., arr.split].cumsum(dim=0)[:last_diag_pr + 1]
        diag_crossings[-1] = diag_crossings[-1] if diag_crossings[-1] <= min(arr.gshape) else min(arr.gshape)
        diag_crossings = torch.cat((torch.tensor([0]), diag_crossings), dim=0)
        col_inds = []
        for col in range(tile_columns.item()):
            _, lshape, _ = arr.comm.chunk([diag_crossings[col // tile_rows + 1] - diag_crossings[col // tile_rows]], 0,
                                          rank=int(col % tile_rows), w_size=tile_rows if col // tile_rows != last_diag_pr else last_tile_cols)
            col_inds.append(lshape[0])

        # if there if are < 10 rows left after the diagonal need to reduce the number of tile rows for that process (same as columns)
        total_tile_rows = tile_rows * arr.comm.size
        row_inds = [0] * total_tile_rows
        for c, x in enumerate(col_inds):  # set the row indices to be the same for all of the column indices (however many there are)
            row_inds[c] = x

        last_diag_pr_rows = tile_rows  # tile rows in the last diagonal pr
        if last_diag_pr < arr.comm.size - 1 or (last_diag_pr == arr.comm.size - 1 and row_inds[-1] == 0):
            num_tiles_last_diag_pr = len(col_inds) - (tile_rows * last_diag_pr)  # number of tiles after the diagonal on the last process
            last_diag_pr_rows_rem = tile_rows - num_tiles_last_diag_pr  # number of rows remaining on the lshape
            # how many tiles can be put one the last process with diagonal elements?
            new_tile_rows_remaining = last_diag_pr_rows_rem // 2  # todo: determine if this should be changed to a larger number
            # delete entries from row_inds (need to delete tile_rows - (num_tiles_last_diag_pr + new_tile_rows_remaining))
            last_diag_pr_rows -= num_tiles_last_diag_pr + new_tile_rows_remaining
            del row_inds[-1 * last_diag_pr_rows:]
            if last_diag_pr_rows_rem < 2:
                # if the number of rows after the diagonal is 1 then need to rechunk in the 0th dimension
                for i in range(last_diag_pr_rows.item()):
                    _, lshape, _ = arr.comm.chunk(lshape_map[last_diag_pr], 0, rank=i, w_size=last_diag_pr_rows.item())
                    row_inds[(tile_rows * last_diag_pr).item() + i] = lshape[0]

        # need to determine the rest of the row indices
        nz = torch.nonzero(torch.Tensor(row_inds) == 0)
        for i in range(last_diag_pr.item() + 1, arr.comm.size):  # loop over all of the rest of the processes
            for t in range(tile_rows):
                _, lshape, _ = arr.comm.chunk(lshape_map[i], 0, rank=t, w_size=tile_rows)
                row_inds[nz[0].item()] = lshape[0]
                nz = nz[1:]
        # combine the last tiles into one if there is too little data on the last one
        if row_inds[-1] < 2:  # todo: determine if this should be larger
            row_inds[-2] += row_inds[-1]
            del row_inds[-1]

        tile_map = torch.zeros([len(row_inds), len(col_inds), 3], dtype=torch.int)
        # units -> row, column, size in each direction, process
        for num, c in enumerate(col_inds):  # set columns
            tile_map[:, num, 1] = c
        for num, r, in enumerate(row_inds):  # set rows
            tile_map[num, :, 0] = r
        for p in range(last_diag_pr.item()):  # set ranks
            tile_map[tile_rows * p:tile_rows * (p + 1), :, 2] = p
        # set last diag pr rank
        tile_map[tile_rows * last_diag_pr:tile_rows * last_diag_pr + last_diag_pr_rows, :, 2] = last_diag_pr
        # set the rest of the ranks
        st = tile_rows * last_diag_pr + last_diag_pr_rows
        for p in range(arr.comm.size - last_diag_pr.item() + 1):
            tile_map[st:st + tile_rows * (p + 1), :, 2] = p + last_diag_pr.item() + 1
            st += tile_rows

        # =================================================================================================
        self.__DNDarray = arr
        self.__lshape_map = lshape_map
        self.__tile_map = tile_map
        self.__tile_columns = len(col_inds)
        self.__tile_rows = len(row_inds)
        # =================================================================================================

    @property
    def lsahpe_map(self):
        return self.__lshape_map

    @property
    def tile_map(self):
        return self.__tile_map

    @property
    def tile_columns(self):
        return self.__tile_columns

    @property
    def tile_rows(self):
        return self.__tile_rows

    def __getitem__(self, key):
        # default getitem will return the data in the array!!
        # this is intended to return tiles which are local. it will return torch.Tensors which correspond to the tiles of the array
        # this is a global getter, if the tile is not on the process then it will return None
        arr = self.__DNDarray
        tile_map = self.__tile_map
        tile_rows = self.__tile_rows
        local_arr = self.__DNDarray._DNDarray__array
        # cases:
        # int,
        if isinstance(key, int):
            # get all the instances in a row (tile column 0 -> end)
            # todo: determine the rank with the diagonal element to determine the 1st coordinate
            if arr.split != 0:
                raise ValueError('Slicing across splits is not allowed')
            # print(arr.comm.rank, tile_map[key][..., 2].unique())
            if arr.comm.rank == int(tile_map[key][..., 2].unique()):
                rank_sliced = torch.where(tile_map[..., 2] == arr.comm.rank)[0].unique()
                st0 = tile_map[..., 1][rank_sliced][:key % rank_sliced.shape[0], 0].sum()
                sp0 = tile_map[..., 1][rank_sliced][key % rank_sliced.shape[0], 0] + st0
                return local_arr[st0:sp0]
            else:
                return None

        # tuple,
        if isinstance(key, tuple) and arr.split is not None:
            # need to
            if all(isinstance(x, int) for x in key):
                if arr.comm.rank == key[arr.split] // tile_rows:
                    key = list(key)
                    key[0] = key[0] % tile_rows
                    st0 = tile_map[arr.comm.rank, :key[0], key[1], 0].sum()
                    sp0 = tile_map[arr.comm.rank, :, :key[1], 0][key[0]] + st0

                    st1 = tile_map[arr.comm.rank, key[0], key[1], 1]
                    sp1 = tile_map[arr.comm.rank, key[0], :key[1], 1].sum() + st1
                    return arr._DNDarray__array[st0:sp0, st1:sp1]
                else:
                    return None
            elif isinstance(key[arr.split], slice):
                if key[arr.split].stop - key[arr.split].start > tile_rows:  # need to adjust the logic for split == 1
                    raise ValueError('Slicing across splits is not allowed')
            elif isinstance(key[arr.split], int):
                if arr.split == 0:
                    if isinstance(key[1], slice):
                        if arr.comm.rank == key[arr.split] // tile_rows:
                            key = list(key)
                            key[0] = key[0] % tile_rows  # adjust the row number to be the local number on that rank

                            st0 = tile_map[arr.comm.rank, :key[0], key[1], 0].sum()
                            sp0 = tile_map[arr.comm.rank, :, :key[1], 0][key[0]] + st0

                            st1 = tile_map[arr.comm.rank, key[0], key[1], 1]
                            sp1 = tile_map[arr.comm.rank, key[0], :key[1], 1].sum() + st1
                            return arr._DNDarray__array[st0:sp0, st1:sp1]
                    pass
                else:  # split == 1
                    pass
        # slice,
        if isinstance(key, slice):
            if arr.split > 0:
                raise ValueError('Slicing across splits is not allowed')
        # torch indices

    # need to get:
    # tile data
    # tile start
    # tile end
    # tile size
