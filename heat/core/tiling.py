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


# class LocalTileIndex:
#     """
#     Indexing class for local operations (primarily for lloc function)
#     For docs on __getitem__ and __setitem__ see lloc(self)
#     """
#
#     # def __init__(self, tiles):
#     #     self.tiles = tiles
#
#     def __getitem__(self, key):
#         return self.__getitem__(key=key)
#
#     def __setitem__(self, key, value):
#         self.__setitem__(key=key, value=value)


def mm_tiles(arr):
    if not isinstance(arr, dndarray.DNDarray):
        raise TypeError('arr must be a DNDarray, is currently a {}'.format(type(arr)))

    lshape_map = torch.zeros((arr.comm.size, len(arr.gshape)), dtype=int)
    lshape_map[arr.comm.rank, :] = torch.Tensor(arr.lshape)
    arr.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)
    return None, lshape_map


class SquareDiagTiles:
    # designed for QR tile scheme
    def __init__(self, arr, tile_per_proc=2, lshape_map=None):
        """
        Generate the tile map and the other objects which may be useful.
        The tiles generated here are based of square tiles along the diagonal. The size of these tiles along the diagonal dictate the divisions accross
        all processes. If gshape[0] >> gshape[1] then there will be extra tiles generated below the diagonal. If gshape[0] is close to gshape[1], then
        the last tile (as well as the other tiles which correspond with said tile) will be extended to cover the whole array. However, extra tiles are
        not generated above the diagonal in the case that gshape[0] << gshape[1].

        This tiling scheme was intended for use with the QR function.

        Parameters
        ----------
        arr : DNDarray
            the array to be tiled
        tile_per_proc : int
            Default = 2
            the number of divisions per process,
            if split = 0 then this is the starting number of tile rows
            if split = 1 then this is the starting number of tile columns

        Returns
        -------
        None

        Initializes
        -----------
        __col_per_proc_list : list
            list is length of the number of processes, each element has the number of tile columns on the process whos rank equals the index
        __DNDarray = arr : DNDarray
            the whole DNDarray
        __lshape_map : torch.Tensor
            tensor filled with the shapes of the local tensors
        __tile_map : torch.Tensor
            tensor filled with the global indices of the generated tiles
        __row_per_proc_list : list
            list is length of the number of processes, each element has the number of tile rows on the process whos rank equals the index
        __tile_columns : int
            number of tile columns
        __tile_rows : int
            number of tile rows
        """
        # lshape_map -> rank (int), lshape (tuple of the local lshape, self.lshape)
        if not isinstance(arr, dndarray.DNDarray):
            raise TypeError('self must be a DNDarray, is currently a {}'.format(type(self)))

        # todo: unbalance the array if there is *only* one row/column of the diagonal on a process (send it to pr - 1)
        # todo: small bug in edge case for very small matrices with < 10 elements on a process and split = 1 with gshape[0] > gshape[1]
        if lshape_map is None:
            #create lshape map
            lshape_map = torch.zeros((arr.comm.size, len(arr.gshape)), dtype=int)
            lshape_map[arr.comm.rank, :] = torch.Tensor(arr.lshape)
            arr.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

        # chunk map
        # is the diagonal crossed by a division between processes/where
        last_diag_pr = torch.where(lshape_map[..., arr.split].cumsum(dim=0) >= min(arr.gshape))[0][0]
        # adjust for small blocks on the last diag pr:
        rem_cols_last_pr = min(arr.gshape) - lshape_map[..., arr.split].cumsum(dim=0)[last_diag_pr - 1]  # end of the process before the split
        last_tile_cols = tile_per_proc
        while rem_cols_last_pr / last_tile_cols < 2:  # todo: determine best value for this (prev at 10)
            # if there cannot be tiles formed which are at list ten items large then need to reduce the number of tiles
            last_tile_cols -= 1
            if last_tile_cols == 1:
                break
        # create lists of columns and rows for each process
        col_per_proc_list = [tile_per_proc] * (last_diag_pr.item() + 1)
        col_per_proc_list[-1] = last_tile_cols
        row_per_proc_list = [tile_per_proc] * arr.comm.size

        # need to determine the proper number of tile rows/columns
        tile_columns = tile_per_proc * last_diag_pr + last_tile_cols
        diag_crossings = lshape_map[..., arr.split].cumsum(dim=0)[:last_diag_pr + 1]
        diag_crossings[-1] = diag_crossings[-1] if diag_crossings[-1] <= min(arr.gshape) else min(arr.gshape)
        diag_crossings = torch.cat((torch.tensor([0]), diag_crossings), dim=0)
        # create the tile columns sizes, saved to list
        col_inds = []
        for col in range(tile_columns.item()):
            _, lshape, _ = arr.comm.chunk([diag_crossings[col // tile_per_proc + 1] - diag_crossings[col // tile_per_proc]], 0,
                                          rank=int(col % tile_per_proc), w_size=tile_per_proc if col // tile_per_proc != last_diag_pr else last_tile_cols)
            col_inds.append(lshape[0])

        total_tile_rows = tile_per_proc * arr.comm.size
        row_inds = [0] * total_tile_rows
        for c, x in enumerate(col_inds):
            # set the row indices to be the same for all of the column indices (however many there are)
            row_inds[c] = x

        if arr.gshape[0] < arr.gshape[1] and arr.split == 0:
            # need to adjust the very last tile to be the remaining
            col_inds[-1] = arr.gshape[1] - sum(col_inds[:-1])

        last_diag_pr_rows = tile_per_proc  # tile rows in the last diagonal pr
        # adjust the rows on the last process which has diagonal elements
        if last_diag_pr < arr.comm.size - 1 or (last_diag_pr == arr.comm.size - 1 and row_inds[-1] == 0):
            num_tiles_last_diag_pr = len(col_inds) - (tile_per_proc * last_diag_pr)
            # num_tiles_last_diag_pr = number of tiles after the diagonal on the last process
            last_diag_pr_rows_rem = tile_per_proc - num_tiles_last_diag_pr
            # last_diag_pr_rows_rem = number of rows remaining on the lshape
            # how many tiles can be put one the last process with diagonal elements?
            new_tile_rows_remaining = last_diag_pr_rows_rem // 2
            # todo: determine if this should be changed to a larger number
            # delete entries from row_inds
            # (need to delete tile_per_proc - (num_tiles_last_diag_pr + new_tile_rows_remaining))
            last_diag_pr_rows -= num_tiles_last_diag_pr + new_tile_rows_remaining
            del row_inds[-1 * last_diag_pr_rows:]
            row_per_proc_list[last_diag_pr] = last_diag_pr_rows

            if last_diag_pr_rows_rem < 2 and arr.split == 0:
                # if the number of rows after the diagonal is 1
                # then need to rechunk in the 0th dimension
                for i in range(last_diag_pr_rows.item()):
                    _, lshape, _ = arr.comm.chunk(lshape_map[last_diag_pr], 0, rank=i, w_size=last_diag_pr_rows.item())
                    row_inds[(tile_per_proc * last_diag_pr).item() + i] = lshape[0]

        nz = torch.nonzero(torch.Tensor(row_inds) == 0)
        for i in range(last_diag_pr.item() + 1, arr.comm.size):
            # loop over all of the rest of the processes
            for t in range(tile_per_proc):
                _, lshape, _ = arr.comm.chunk(lshape_map[i], 0, rank=t, w_size=tile_per_proc)
                row_inds[nz[0].item()] = lshape[0]
                nz = nz[1:]

        # combine the last tiles into one if there is too little data on the last one
        if row_inds[-1] < 2 and arr.split == 0:  # todo: determine if this should be larger
            row_inds[-2] += row_inds[-1]
            del row_inds[-1]
            row_per_proc_list[-1] -= 1

        # add extra rows if there is place below the diagonal
        if arr.gshape[0] > arr.gshape[1] and arr.split == 1:
            # need to adjust the very last tile to be the remaining
            if arr.gshape[0] - arr.gshape[1] > 10:  # todo: determine best value for this
                # use chunk and a loop over the however many tiles are desired
                num_ex_row_tiles = 4  # todo: determine best value for this
                while (arr.gshape[0] - arr.gshape[1]) // num_ex_row_tiles < 2:
                    num_ex_row_tiles -= 1
                for i in range(num_ex_row_tiles):
                    _, lshape, _ = arr.comm.chunk((arr.gshape[0] - arr.gshape[1],), 0,
                                                  rank=i, w_size=num_ex_row_tiles)
                    row_inds.append(lshape[0])
            else:
                # if there is no place for multiple tiles, combine the remainder with the last row
                row_inds[-1] = arr.gshape[0] - sum(row_inds[:-1])

        tile_map = torch.zeros([len(row_inds), len(col_inds), 3], dtype=torch.int)
        # units -> row, column, start index in each direction, process
        # if arr.split == 0:  # adjust the 1st dim to be the cumsum
        col_inds = [0] + col_inds[:-1]
        col_inds = torch.tensor(col_inds).cumsum(dim=0)
        # if arr.split == 1:  # adjust the 0th dim to be the cumsum
        row_inds = [0] + row_inds[:-1]
        row_inds = torch.tensor(row_inds).cumsum(dim=0)

        for num, c in enumerate(col_inds):  # set columns
            tile_map[:, num, 1] = c
        for num, r, in enumerate(row_inds):  # set rows
            tile_map[num, :, 0] = r

        # setting of rank is different for split 0 and split 1
        if arr.split == 0:
            for p in range(last_diag_pr.item()):  # set ranks
                tile_map[tile_per_proc * p:tile_per_proc * (p + 1), :, 2] = p
            # set last diag pr rank
            tile_map[tile_per_proc * last_diag_pr:tile_per_proc * last_diag_pr + last_diag_pr_rows, :, 2] = last_diag_pr
            # set the rest of the ranks
            st = tile_per_proc * last_diag_pr + last_diag_pr_rows
            for p in range(arr.comm.size - last_diag_pr.item() + 1):
                tile_map[st:st + tile_per_proc * (p + 1), :, 2] = p + last_diag_pr.item() + 1
                st += tile_per_proc
        elif arr.split == 1:
            for p in range(last_diag_pr.item()):  # set ranks
                tile_map[:, tile_per_proc * p:tile_per_proc * (p + 1), 2] = p
            # set last diag pr rank
            tile_map[:, tile_per_proc * last_diag_pr:tile_per_proc * last_diag_pr + last_diag_pr_rows, 2] = last_diag_pr
            # set the rest of the ranks
            st = tile_per_proc * last_diag_pr + last_diag_pr_rows
            for p in range(arr.comm.size - last_diag_pr.item() + 1):
                tile_map[:, st:st + tile_per_proc * (p + 1), 2] = p + last_diag_pr.item() + 1
                st += tile_per_proc

        for c, i in enumerate(row_per_proc_list):
            try:
                row_per_proc_list[c] = i.item()
            except AttributeError:
                pass
        for c, i in enumerate(col_per_proc_list):
            try:
                col_per_proc_list[c] = i.item()
            except AttributeError:
                pass

        # =================================================================================================
        self.__col_per_proc_list = col_per_proc_list if arr.split == 1 \
            else [sum(col_per_proc_list)] * len(col_per_proc_list)
        self.__DNDarray = arr
        self.__lshape_map = lshape_map
        self.__last_diag_pr = last_diag_pr.item()
        self.__row_per_proc_list = row_per_proc_list
        self.__tile_map = tile_map
        # print(type(row_inds), t?ype(col_inds))
        self.__row_inds = list(row_inds)
        self.__col_inds = list(col_inds)
        self.__tile_columns = len(col_inds)
        self.__tile_rows = len(row_inds)
        # =================================================================================================

    @property
    def arr(self):
        return self.__DNDarray

    @property
    def col_indices(self):
        return self.__col_inds

    # @property
    # def lloc(self):
    #     return LocalTileIndex()

    @property
    def lshape_map(self):
        """
        Returns
        -------
        torch.Tensor : map of the lshape tuples for the DNDarray given
             units -> rank (int), lshape (tuple of the local shape)
        """
        return self.__lshape_map

    @property
    def last_diagonal_process(self):
        """
        returns the rank of the last process with diagonal elements
        :return:
        """
        return self.__last_diag_pr

    @property
    def row_indices(self):
        return self.__row_inds

    @property
    def tile_columns(self):
        """
        Returns
        -------
        int : number of tile columns
        """
        return self.__tile_columns

    @property
    def tile_columns_per_process(self):
        """
        Returns
        -------
        list : list containing the number of columns on all processes
        """
        return self.__col_per_proc_list

    @property
    def tile_map(self):
        """
        Returns
        -------
        torch.Tensor : map of tiles
            tile_map contains the sizes of the tiles
            units -> row, column, start index in each direction, process
        """
        return self.__tile_map

    @property
    def tile_rows(self):
        """
        Returns
        -------
        int : number of tile rows
        """
        return self.__tile_rows

    @property
    def tile_rows_per_process(self):
        """
        Returns
        -------
        list : list containing the number of rows on all processes
        """
        return self.__row_per_proc_list

    # todo: use dummy class to get square bracket availability
    def async_get(self, key, dest):
        """
        Call to get a tile and then send it to the specified process (dest) using Send and Irecv

        :param key:
        :param dest:
        :return:
        """
        tile = self.__getitem__(key)
        comm = self.__DNDarray.comm
        src = self.tile_map[key][..., 2].unique().item()
        if tile is not None:  # this will only be on one process (required by getitem)
            comm.isend(tile.clone(), dest=dest)
        if comm.rank == dest:
            ret = comm.irecv(source=src)
            return ret
        else:
            return tile

    def async_set(self, key, data, home):
        """
        Function to set the specified tile's values on the original tile's process

        :param key:
        :param data:
        :param home:
        :return:
        """
        tile = self.__getitem__(key)
        comm = self.__DNDarray.comm
        dest = self.get_tile_proc(key)
        if data is not None:  # this will only be on one process (required by getitem)
            comm.isend(data.copy(), dest=dest)
        if comm.rank == dest:
            ret = comm.recv(source=home)
            tile.__setitem__(slice(None), ret)

    def convert_local_key_to_global(self, key, proc):
        """
        conver a key from local indices to global indices
        :param key: key to convert
        :param proc: process on which the key is indended for
        :return:
        """
        arr = self.__DNDarray
        # need to convert the key into local indices -> only done on the split dimension
        key = list(key)
        if len(key) == 1:
            key = [key, slice(0, arr.gshape[1])]

        if arr.split == 0:
            # need to adjust key[0] to be only on the local tensor
            prev_rows = sum(self.__row_per_proc_list[:proc])
            loc_rows = self.__row_per_proc_list[proc]
            if isinstance(key[0], int):
                key[0] += prev_rows
            elif isinstance(key[0], slice):
                start = key[0].start + prev_rows if key[0].start is not None else prev_rows
                stop = key[0].stop + prev_rows if key[0].stop is not None else start + loc_rows
                if stop - start > loc_rows:
                    stop = start + loc_rows
                key[0] = slice(start, stop)
        if arr.split == 1:
            # need to adjust key[0] to be only on the local tensor
            # need the number of columns *before* the process
            # todo: adjust slices in split=1 case
            prev_cols = sum(self.__col_per_proc_list[:proc])
            loc_cols = self.__col_per_proc_list[proc]
            if isinstance(key[1], int):
                key[1] += prev_cols
            elif isinstance(key[1], slice):
                start = key[1].start + prev_cols
                stop = key[1].stop + prev_cols
                if stop - start > loc_cols:
                    stop = start + loc_cols
                key[1] = slice(start, stop)

        return tuple(key)

    def get_start_stop(self, key):
        """
        NOTE: this uses global indices!
        returns the start and stop indices which correspond to the selected tile

        Parameters
        ----------
        key : int, tuple, list, slice
            indices to select the tile
            STRIDES ARE NOT ALLOWED

        Returns
        -------
        tuple : dim0 start, dim0 stop, dim1 start, dim1 stop
        """
        # todo: change this to use the row/col indices
        # default getitem will return the data in the array!!
        # this is intended to return tiles which are local.
        # it will return torch.Tensors which correspond to the tiles of the array
        # this is a global getter, if the tile is not on the process then it will return None
        split = self.__DNDarray.split
        pr = self.tile_map[key][..., 2].unique()
        row_inds = self.row_indices + [self.__DNDarray.gshape[0]]
        col_inds = self.col_indices + [self.__DNDarray.gshape[1]]
        row_start = row_inds[sum(self.tile_rows_per_process[:pr]) if split == 0 else 0]
        col_start = col_inds[sum(self.tile_columns_per_process[:pr]) if split == 1 else 0]

        if not isinstance(key, (tuple, list, slice, int)):
            raise TypeError(
                'key must be an int, tuple, or slice, is currently {}'.format(type(key)))

        if isinstance(key, (slice, int)):
            key = tuple(key, slice(0, None))

        key = list(key)

        if isinstance(key[0], int):
            st0 = row_inds[key[0]] - row_start
            sp0 = row_inds[key[0] + 1] - row_start
        if isinstance(key[0], slice):
            start = row_inds[key[0].start] if key[0].start is not None else 0
            stop = row_inds[key[0].stop] if key[0].stop is not None else row_inds[-1]
            st0, sp0 = start - row_start, stop - row_start
        if isinstance(key[1], int):
            st1 = col_inds[key[1]] - col_start
            sp1 = col_inds[key[1] + 1] - col_start
        if isinstance(key[1], slice):
            start = col_inds[key[1].start] if key[1].start is not None else 0
            stop = col_inds[key[1].stop] if key[1].stop is not None else col_inds[-1]
            st1, sp1 = start - col_start, stop - col_start

        return st0, sp0, st1, sp1

    def get_tile_proc(self, key):
        # NOTE: this uses global indices!
        return self.tile_map[key][..., 2].unique()

    def get_tile_size(self, key):
        """
        NOTE: this uses global indices!
        Returns
        -------
        torch.Shape : uses the getitem routine then calls the torch shape function
        """
        tup = self.get_start_stop(key)
        return tup[1] - tup[0], tup[3] - tup[2]

    def __getitem__(self, key):
        """
        Standard getitem function for the tiles. The returned item is a view of the original DNDarray, operations which are done to this view will change
        the original array.
        **STRIDES ARE NOT AVAILABLE, NOR ARE CROSS-SPLIT SLICES**

        Parameters
        ----------
        key : int, slice, tuple, list
            indices of the tile/s desired

        Returns
        -------
        DNDarray_view : torch.Tensor
            A local selection of the DNDarray corresponding to the tile/s desired
        """
        # default getitem will return the data in the array!!
        # this is intended to return tiles which are local. it will return torch.Tensors which correspond to the tiles of the array
        arr = self.__DNDarray
        tile_map = self.__tile_map
        local_arr = self.__DNDarray._DNDarray__array
        if isinstance(key, int):
            # get all the instances in a row (tile column 0 -> end)
            # todo: determine the rank with the diagonal element to determine the 1st coordinate
            if arr.split != 0:
                raise ValueError('Slicing across splits is not allowed')
            if arr.comm.rank == tile_map[key][..., 2].unique():
                # above is the code to get the tile map for what is all on one tile
                prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank])
                st0 = tile_map[key, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                sp0 = tile_map[key + 1, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                return local_arr[st0:sp0]
            else:
                return None
        elif tile_map[key][..., 2].unique().nelement() > 1:
            raise ValueError('Slicing across splits is not allowed')
        else:
            if arr.comm.rank == tile_map[key][..., 2].unique():
                if not isinstance(key, (int, tuple, list, slice)):
                    raise TypeError('key must be an int, tuple, or slice, is currently {}'.format(type(key)))

                key = list(key)
                split = self.__DNDarray.split
                rank = arr.comm.rank
                # print(key)
                row_inds = self.row_indices + [self.__DNDarray.gshape[0]]
                col_inds = self.col_indices + [self.__DNDarray.gshape[1]]
                row_start = row_inds[sum(self.tile_rows_per_process[:rank]) if split == 0 else 0]
                col_start = col_inds[sum(self.tile_columns_per_process[:rank]) if split == 1 else 0]

                if len(key) == 1:
                    key.append(slice(0, None))

                if all(isinstance(x, int) for x in key):
                    st0 = row_inds[key[0]] - row_start
                    sp0 = row_inds[key[0] + 1] - row_start
                    st1 = col_inds[key[1]] - col_start
                    sp1 = col_inds[key[1] + 1] - col_start

                elif all(isinstance(x, slice) for x in key):
                    # need to set the values of start and stop if they are None
                    start = col_inds[key[1].start] if key[1].start is not None else 0
                    stop = col_inds[key[1].stop] if key[1].stop is not None else col_inds[-1]
                    st1, sp1 = start - col_start, stop - col_start

                    # need to adjust the indices from tiles to local for dim0
                    start = row_inds[key[0].start] if key[0].start is not None else 0
                    stop = row_inds[key[0].stop] if key[0].stop is not None else row_inds[-1]
                    st0, sp0 = start - row_start, stop - row_start

                elif isinstance(key[0], slice) and isinstance(key[1], int):
                    start = row_inds[key[0].start] if key[0].start is not None else 0
                    stop = row_inds[key[0].stop] if key[0].stop is not None else row_inds[-1]
                    st0, sp0 = start - row_start, stop - row_start
                    st1 = col_inds[key[1]] - col_start
                    sp1 = col_inds[key[1] + 1] - col_start

                elif isinstance(key[1], slice) and isinstance(key[0], int):
                    start = col_inds[key[1].start] if key[1].start is not None else 0
                    stop = col_inds[key[1].stop] if key[1].stop is not None else col_inds[-1]
                    st1, sp1 = start - col_start, stop - col_start
                    st0 = row_inds[key[0]] - row_start
                    sp0 = row_inds[key[0] + 1] - row_start

                print('getitem end', st0, sp0, st1, sp1)
                return local_arr[st0:sp0, st1:sp1]
            else:
                return None

    def local_get_async(self, key, src, dest):
        """
        this should be renamed... it doesnt correspond with the local get function
        This function sends the tile on src with the local indices given by the key
        :param key:
        :param src:
        :param dest:
        :return:
        """
        # note: this uses local indices!
        if not isinstance(src, int):
            try:
                src = src.item()
            except AttributeError:
                raise TypeError('src must be an int or a single element tensor, '
                                'is currently {}'.format(type(src)))
        if not isinstance(dest, int):
            try:
                dest = dest.item()
            except AttributeError:
                raise TypeError('dest must be an int or a single element tensor'
                                ', is currently {}'.format(type(src)))
        if not src < self.__DNDarray.comm.size or not dest < self.__DNDarray.comm.size:
            raise ValueError('src and dest must be less than the size, '
                             'currently {} and {} respectively'.format(src, dest))

        key = self.convert_local_key_to_global(key, proc=src)

        comm = self.__DNDarray.comm
        tile = self[key].clone() if comm.rank == src else None
        sz = self.get_tile_size(key=key)
        if comm.rank == src:  # this will only be on one process (required by getitem)
            comm.Isend(tile, dest=dest, tag=int("2222" + str(src) + str(dest)))
            return tile, None
        if comm.rank == dest:
            hld = torch.zeros(sz)
            wait = comm.Irecv(hld, source=src, tag=int("2222" + str(src) + str(dest)))
            return hld, wait

    def local_get(self, key, proc=None):
        """
        get the tile corresponding to the local indices given in the key
        :param key:
        :param proc:
        :return:
        """
        # this is to be used with only local indices!
        # convert from local to global
        proc = proc if proc is not None else self.__DNDarray.comm.rank
        if proc == self.__DNDarray.comm.rank:
            arr = self.__DNDarray
            # need to convert the key into local indices -> only done on the split dimension
            key = list(key)

            if arr.split == 0:
                if len(key) == 1:
                    key = [key, slice(0, None)]
                # need to adjust key[0] to be only on the local tensor
                prev_rows = sum(self.__row_per_proc_list[:proc])
                loc_rows = self.__row_per_proc_list[proc]
                if isinstance(key[0], int):
                    key[0] += prev_rows
                elif isinstance(key[0], slice):
                    start = key[0].start + prev_rows if key[0].start is not None else prev_rows
                    stop = key[0].stop + prev_rows if key[0].stop is not None else start + loc_rows
                    if stop - start > loc_rows:
                        # print(local_tile_map)
                        stop = start + loc_rows
                    # print(start, stop)
                    key[0] = slice(start, stop)

            if arr.split == 1:
                loc_cols = self.__col_per_proc_list[proc]
                prev_cols = sum(self.__col_per_proc_list[:proc])
                if len(key) == 1:
                    key = [key, slice(0, None)]
                # need to adjust key[0] to be only on the local tensor
                # need the number of columns *before* the process
                if isinstance(key[1], int):
                    key[1] += prev_cols
                elif isinstance(key[1], slice):
                    start = key[1].start + prev_cols if key[1].start is not None else prev_cols
                    stop = key[1].stop + prev_cols if key[1].stop is not None else start + prev_cols
                    if stop - start > loc_cols:
                        stop = start + loc_cols
                    key[1] = slice(start, stop)
            # print('moving to getitem')
            return self.__getitem__(tuple(key))

    def local_set(self, key, data, proc=None):
        """
        get the tile corresponding to the local
        :param key:
        :param proc:
        :return:
        """
        # this is to be used with only local indices!
        # convert from local to global?
        proc = proc if proc is not None else self.__DNDarray.comm.rank
        if proc == self.__DNDarray.comm.rank:
            arr = self.__DNDarray
            # rank_slice = torch.where(tile_map[..., 2] == proc)
            # lcl_tile_map = tile_map[rank_slice]
            # print(self.tile_columns_per_process)
            # need to convert the key into local indices -> only done on the split dimension
            key = list(key)
            key = [key, slice(0, arr.gshape[1])] if len(key) == 1 else key

            if arr.split == 0:
                # need to adjust key[0] to be only on the local tensor
                prev_rows = sum(self.__row_per_proc_list[:proc])
                loc_rows = self.__row_per_proc_list[proc]
                if isinstance(key[0], int):
                    key[0] += prev_rows
                elif isinstance(key[0], slice):
                    start = key[0].start + prev_rows if key[0].start is not None else prev_rows
                    stop = key[0].stop + prev_rows if key[0].stop is not None else start + loc_rows
                    if stop - start > loc_rows:
                        # print(local_tile_map)
                        stop = start + loc_rows
                    key[0] = slice(start, stop)
            if arr.split == 1:
                # need to adjust key[0] to be only on the local tensor
                # need the number of columns *before* the process
                prev_cols = sum(self.__col_per_proc_list[:proc])
                loc_cols = self.__col_per_proc_list[proc]
                if isinstance(key[1], int):
                    key[1] += prev_cols
                elif isinstance(key[1], slice):
                    start = key[1].start + prev_cols
                    stop = key[1].stop + prev_cols
                    if stop - start > loc_cols:
                        # print(local_tile_map)
                        stop = start + loc_cols
                    key[1] = slice(start, stop)
            # print('from local', key)
            # self.__setitem__(tuple(key), data)
            # print(len(torch.where(torch.isnan(data))[0]))
            self.__getitem__(tuple(key)).__setitem__(slice(0, None), data)

    def match_tiles(self, tiles_to_match):
        """
        function to match the tile sizes of another tile map
        NOTE: this is intended for use with the Q matrix, to match the tiling of a/R
        For this to work properly it is required that the 0th dim of both matrices is equal
        :param tiles_to_match:
        :return:
        """
        if not isinstance(tiles_to_match, SquareDiagTiles):
            raise TypeError("tiles_to_match must be a SquareDiagTiles object, currently: {}"
                            .format(type(tiles_to_match)))
        base_dnd = self.__DNDarray
        # this map will take the same tile row and column sizes up to the last diagonal row/column
        # the last row/column is determined by the number of rows/columns on the non-split dimension
        last_col_row = tiles_to_match.tile_rows_per_process[-1] if base_dnd.split == 1 \
            else tiles_to_match.tile_columns_per_process[-1]
        # working with only split=0 for now todo: split=1
        if base_dnd.split == tiles_to_match.__DNDarray.split == 0:
            # **the number of tiles on rows and columns will be equal for this new tile map**
            if base_dnd.split == 0:
                new_rows = tiles_to_match.tile_rows_per_process
            else:
                # this is assuming that the array is split
                new_rows = tiles_to_match.tile_columns_per_process

            # set the columns which are less than the last col_row to be the same as the last one
            # for i in range(last_col_row):
            # if split=0 then can just set the columns easily
            new_row_inds = []
            new_col_inds = []
            new_row_inds.extend(tiles_to_match.row_indices)
            new_col_inds.extend(tiles_to_match.col_indices)

            match_shape = tiles_to_match.__DNDarray.shape
            match_end_dim = 0 if match_shape[0] <= match_shape[1] else 1
            match_diag_end = tiles_to_match.__DNDarray.shape[match_end_dim]
            self_diag_end = sum(self.lshape_map[:tiles_to_match.last_diagonal_process + 1]
                              [..., base_dnd.split])
            # print(match_diag_end, self_lshape)

            # match_end = sum(match_lshape[..., base_dnd.split])
            # below only needs to run if there is enough space for another block (>=2 entries)
            # and only if the last diag pr is not the last one
            if (tiles_to_match.last_diagonal_process != base_dnd.comm.size - 1 and
                    base_dnd.split == 0):
                # print('here', self_diag_end, match_diag_end)
                if self_diag_end - match_diag_end >= 2:
                    new_row_inds.insert(last_col_row, match_diag_end)
                    new_rows[tiles_to_match.last_diagonal_process] += 1
                    # new_col_inds = new_row_inds.copy()
                if self_diag_end - match_end_dim < 2:
                    new_row_inds[last_col_row] = match_diag_end
                    # new_row_inds.insert(last_col_row - 1, torch.tensor(match_end))
                    # new_rows[tiles_to_match.last_diagonal_process] += 1
                new_col_inds = new_row_inds.copy()

            if (tiles_to_match.last_diagonal_process != base_dnd.comm.size - 1 and
                    base_dnd.split == 1 and
                    self_diag_end - match_diag_end >= 2):
                new_col_inds.insert(last_col_row, match_diag_end)
                new_rows[tiles_to_match.last_diagonal_process] += 1
                new_row_inds = new_col_inds.copy()

            # # create the new tile_map of all zeros
            # # units -> row, column, start index in each direction, process
            new_tile_map = torch.zeros([sum(new_rows), sum(new_rows), 3], dtype=torch.int)

            new_tile_map[0][..., 1] = 1
            proc_list = torch.cumsum(torch.tensor(new_rows), dim=0)
            pr, pr_hold = 0, 0
            # print(new_rows, new_col_inds)
            for c in range(sum(new_rows)):
                new_tile_map[..., 1][c] = torch.tensor(new_col_inds)
                new_tile_map[c][..., 0] = new_col_inds[c]
                if pr_hold == proc_list[0]:
                    pr += 1
                    proc_list = proc_list[1:]
                pr_hold += 1
                new_tile_map[c][..., 2] = pr
            self.__tile_map = new_tile_map
            # other things to set:
            self.__col_per_proc_list = new_rows
            self.__row_per_proc_list = new_rows
            self.__last_diag_pr = tiles_to_match.last_diagonal_process
            self.__row_inds = new_row_inds
            self.__col_inds = new_col_inds
            self.__tile_columns = len(new_col_inds)
            self.__tile_rows = len(new_row_inds)
        # case for different splits:
        if base_dnd.split != tiles_to_match.__DNDarray.split and \
                base_dnd.shape == tiles_to_match.__DNDarray.shape:
            # if the shapes are the same (also both are square)
            # then only need to change the processes indices
            # flipping rows and columns for an array that is the same size with a different split
            self.__col_per_proc_list = tiles_to_match.__row_per_proc_list
            self.__row_per_proc_list = tiles_to_match.__col_per_proc_list
            self.__row_inds = tiles_to_match.__col_inds
            self.__col_inds = tiles_to_match.__row_inds
            self.__last_diag_pr = tiles_to_match.last_diagonal_process  # shapes are equal
            # need to create the new tile map
            new_tile_map = tiles_to_match.tile_map.clone()
            tile_per_proc = self.__col_per_proc_list[0]
            last_diag_pr_rows = self.__row_per_proc_list[self.__last_diag_pr]
            for p in range(self.__last_diag_pr):  # set ranks
                new_tile_map[:, tile_per_proc * p:tile_per_proc * (p + 1), 2] = p
            # set last diag pr rank
            new_tile_map[:, tile_per_proc * self.__last_diag_pr:tile_per_proc * self.__last_diag_pr + last_diag_pr_rows, 2] = self.__last_diag_pr
            # set the rest of the ranks
            st = tile_per_proc * self.__last_diag_pr + last_diag_pr_rows
            for p in range(base_dnd.comm.size - self.__last_diag_pr + 1):
                new_tile_map[:, st:st + tile_per_proc * (p + 1), 2] = p + self.__last_diag_pr + 1
                st += tile_per_proc
            self.__tile_map = new_tile_map
            # other things to set:
            self.__tile_columns = tiles_to_match.__tile_rows
            self.__tile_rows = tiles_to_match.__tile_columns

        if base_dnd.gshape == tiles_to_match.__DNDarray.gshape and \
                base_dnd.split == tiles_to_match.__DNDarray.split:
            self.__tile_map = tiles_to_match.tile_map
            # other things to set:
            self.__col_per_proc_list = tiles_to_match.__col_per_proc_list
            self.__row_per_proc_list = tiles_to_match.__row_per_proc_list
            self.__last_diag_pr = tiles_to_match.last_diagonal_process
            self.__row_inds = tiles_to_match.__row_inds
            self.__col_inds = tiles_to_match.__col_inds
            self.__tile_columns = tiles_to_match.__tile_columns
            self.__tile_rows = tiles_to_match.__tile_rows

    def overwrite_arr(self, arr):
        self.__DNDarray = arr

    def send_and_set(self, key, data, home, irecv=False):
        """
        send data from process=home and set it on tile[key]
        :param key: place to set data on the process which that key corresponds to
        :param data:
        :param home:
        :return:
        """
        # send data shape (blocking)
        comm = self.__DNDarray.comm
        rank = comm.rank
        recv_proc = self.tile_map[key][..., 2].unique().item()
        shape = self.get_tile_size(key)
        # print('1', key, shape)
        if recv_proc == home == rank:
            self.__setitem__(key=key, value=data)
            return
        if rank == home:
            # print('2', key, recv_proc, data.shape)
            comm.Isend(data.clone(), dest=recv_proc, tag=8937)
        # create empty, then recv with wait
        elif rank == recv_proc:
            hld = torch.empty(shape)
            # print('3', key, self.__getitem__(key).shape)
            comm.Recv(hld, source=home, tag=8937)
            # self.__getitem__(key).__setitem__(slice(None), hld)
            # print(self.arr)
            self.__setitem__(key=key, value=hld.clone())

    def __setitem__(self, key, value):
        """
        Item setter,
        uses the torch item setter and the getitem routines to set the values of the original array
        (arr in __init__)

        Parameters
        ----------
        key : int, slice, tuple, list
            tile indices to identify the target tiles
        value : int, torch.Tensor, etc.
            values to be set

        Returns
        -------
        None
        """
        arr = self.__DNDarray
        tile_map = self.__tile_map
        if arr.comm.rank == tile_map[key][..., 2].unique():
            # this will set the tile values using the torch setitem function
            # print('here')
            print('entering getitem')
            arr = self.__getitem__(key)
            print('setting item')
            arr.__setitem__(slice(0, None), value)
            # print(self.__getitem__(key))

