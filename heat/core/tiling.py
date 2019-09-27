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
    def __init__(self, arr, tile_per_proc=2):
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
        __DNDarray = arr : DNDarray
            the whole DNDarray
        __lshape_map : torch.Tensor
            tensor filled with the shapes of the local tensors
        __tile_map : torch.Tensor
            tensor filled with the sizes of the generated tiles
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
        col_per_proc_list = [tile_per_proc] * arr.comm.size
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
        for c, x in enumerate(col_inds):  # set the row indices to be the same for all of the column indices (however many there are)
            row_inds[c] = x

        if arr.gshape[0] < arr.gshape[1] and arr.split == 0:  # need to adjust the very last tile to be the remaining
            col_inds[-1] = arr.gshape[1] - sum(col_inds[:-1])

        last_diag_pr_rows = tile_per_proc  # tile rows in the last diagonal pr
        # adjust the rows on the last process which has diagonal elements
        if last_diag_pr < arr.comm.size - 1 or (last_diag_pr == arr.comm.size - 1 and row_inds[-1] == 0):
            num_tiles_last_diag_pr = len(col_inds) - (tile_per_proc * last_diag_pr)  # number of tiles after the diagonal on the last process
            last_diag_pr_rows_rem = tile_per_proc - num_tiles_last_diag_pr  # number of rows remaining on the lshape
            # how many tiles can be put one the last process with diagonal elements?
            new_tile_rows_remaining = last_diag_pr_rows_rem // 2  # todo: determine if this should be changed to a larger number
            # delete entries from row_inds (need to delete tile_per_proc - (num_tiles_last_diag_pr + new_tile_rows_remaining))
            last_diag_pr_rows -= num_tiles_last_diag_pr + new_tile_rows_remaining
            del row_inds[-1 * last_diag_pr_rows:]
            row_per_proc_list[last_diag_pr] = last_diag_pr_rows

            if last_diag_pr_rows_rem < 2 and arr.split == 0:
                # if the number of rows after the diagonal is 1 then need to rechunk in the 0th dimension
                for i in range(last_diag_pr_rows.item()):
                    _, lshape, _ = arr.comm.chunk(lshape_map[last_diag_pr], 0, rank=i, w_size=last_diag_pr_rows.item())
                    row_inds[(tile_per_proc * last_diag_pr).item() + i] = lshape[0]

        nz = torch.nonzero(torch.Tensor(row_inds) == 0)
        for i in range(last_diag_pr.item() + 1, arr.comm.size):  # loop over all of the rest of the processes
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
        if arr.gshape[0] > arr.gshape[1] and arr.split == 1:  # need to adjust the very last tile to be the remaining
            if arr.gshape[0] - arr.gshape[1] > 10:  # todo: determine best value for this
                # use chunk and a loop over the however many tiles are desired
                num_ex_row_tiles = 4  # todo: determine best value for this
                while (arr.gshape[0] - arr.gshape[1]) // num_ex_row_tiles < 2:
                    num_ex_row_tiles -= 1
                for i in range(num_ex_row_tiles):
                    _, lshape, _ = arr.comm.chunk((arr.gshape[0] - arr.gshape[1],), 0, rank=i, w_size=num_ex_row_tiles)
                    row_inds.append(lshape[0])
            else:
                # if there is no place for multiple tiles then combine the remainder with the last row
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

        # =================================================================================================
        self.__col_per_proc_list = col_per_proc_list
        self.__DNDarray = arr
        self.__lshape_map = lshape_map
        self.__row_per_proc_list = row_per_proc_list
        self.__tile_map = tile_map
        self.__tile_columns = len(col_inds)
        self.__tile_rows = len(row_inds)
        # =================================================================================================

    @property
    def lsahpe_map(self):
        """
        Returns
        -------
        torch.Tensor : map of the lshape tuples for the DNDarray given
             units -> rank (int), lshape (tuple of the local shape)
        """
        return self.__lshape_map

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

    def async_get(self, key, dest):
        """
        Call to get a tile and then send it to the specified process (dest) using Send and Irecv

        :param key:
        :param dest:
        :return:
        """
        tile = self.__getitem__(key)
        comm = self.__DNDarray.comm
        src = self.tile_map[key][..., 2].unique()
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
            comm.isend(data, dest=dest)
        if comm.rank == dest:
            ret = comm.recv(source=home)
            tile.__setitem__(slice(None), ret)

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
                if not isinstance(key, (tuple, list, slice)):
                    raise TypeError('key must be an int, tuple, or slice, is currently {}'.format(type(key)))

                if isinstance(key, slice):
                    key = tuple(key, slice(0, None))
                    self.__getitem__(key)

                key = list(key)
                if all(isinstance(x, int) for x in key):
                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    st0 = tile_map[key[0], 0][..., 0] if arr.split == 1 else tile_map[key[0], 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    sp0 = tile_map[key[0] + 1, 0][..., 0] if arr.split == 1 else tile_map[key[0] + 1, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    st1 = tile_map[0, key[1]][..., 1] if arr.split == 0 else tile_map[0, key[1]][..., 1] - tile_map[0, prev_to_split][..., 1]
                    sp1 = tile_map[0, key[1] + 1][..., 1] if arr.split == 0 else tile_map[0, key[1] + 1][..., 1] - tile_map[0, prev_to_split][..., 1]

                elif isinstance(key[arr.split], slice) and isinstance(key[(arr.split + 1) % len(arr.gshape)], int):
                    # note: strides are not implemented! todo: add to docs
                    slice_dim = arr.split
                    # this is to change from global to local, take the mode of how many tiles are in the split dimension
                    start = key[slice_dim].start if key[slice_dim].start is not None else 0
                    stop = key[slice_dim].stop if key[slice_dim].stop is not None else arr.gshape[slice_dim]
                    key[slice_dim] = slice(start, stop)

                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    if arr.split == 0:
                        st0 = tile_map[key[0].start, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        try:
                            sp0 = tile_map[key[0].stop, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        except IndexError:
                            sp0 = key[0].stop
                        st1 = tile_map[0, key[1]][..., 1]
                        sp1 = tile_map[0, key[1] + 1][..., 1]
                    if arr.split == 1:
                        st0 = tile_map[key[0], 0][..., 0]
                        sp0 = tile_map[key[0] + 1, 0][..., 0]
                        st1 = tile_map[0, key[1].start][..., 1] - tile_map[0, prev_to_split][..., 1]
                        try:
                            sp1 = tile_map[0, key[1].stop][..., 1] - tile_map[0, prev_to_split][..., 1]
                        except IndexError:
                            sp1 = key[1].stop
                elif isinstance(key[arr.split], int) and isinstance(key[(arr.split + 1) % len(arr.gshape)], slice):
                    # this implies that the other axis is a slice -> key = (int, slice) for split = 0
                    slice_dim = (arr.split + 1) % len(arr.gshape)
                    # this is to change from global to local, take the mode of how many tiles are in the split dimension
                    start = key[slice_dim].start if key[slice_dim].start is not None else 0
                    stop = key[slice_dim].stop if key[slice_dim].stop is not None else arr.gshape[slice_dim]
                    key[slice_dim] = slice(start, stop)

                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    if arr.split == 0:
                        st0 = tile_map[key[0], 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        sp0 = tile_map[key[0] + 1, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        st1 = tile_map[0, key[1].start][..., 1]
                        try:
                            sp1 = tile_map[0, key[1].stop][..., 1]
                        except IndexError:
                            sp1 = key[1].stop
                    if arr.split == 1:
                        st0 = tile_map[key[0].start, 0][..., 0]
                        try:
                            sp0 = tile_map[key[0].stop, 0][..., 1]
                        except IndexError:
                            sp0 = key[1].stop
                        st1 = tile_map[0, key[1]][..., 1] - tile_map[0, prev_to_split][..., 1]
                        sp1 = tile_map[0, key[1] + 1][..., 1] - tile_map[0, prev_to_split][..., 1]
                else:  # all slices
                    start = key[arr.split].start if key[arr.split].start is not None else 0
                    stop = key[arr.split].stop if key[arr.split].stop is not None else arr.gshape[arr.split]
                    start2 = key[(arr.split + 1) % len(arr.gshape)].start if key[(arr.split + 1) % len(arr.gshape)].start is not None else 0
                    stop2 = key[(arr.split + 1) % len(arr.gshape)].stop if key[(arr.split + 1) % len(arr.gshape)].stop is not None \
                        else arr.gshape[(arr.split + 1) % len(arr.gshape)]
                    key[(arr.split + 1) % len(arr.gshape)] = slice(start2, stop2)
                    key[arr.split] = slice(start, stop)

                    # rank_slice = torch.where(tile_map[..., 2] == arr.comm.rank)
                    # only need to know how many columns are before the start of the key on the local column
                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    st0 = tile_map[key[0].start, 0][..., 0] if arr.split == 1 else tile_map[key[0].start, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    try:
                        sp0 = tile_map[key[0].stop, 0][..., 0] if arr.split == 1 else tile_map[key[0].stop, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    except IndexError:
                        sp0 = key[0].stop
                    st1 = tile_map[0, key[1].start][..., 1] if arr.split == 0 else tile_map[0, key[1].start][..., 1] - tile_map[0, prev_to_split][..., 1]
                    try:
                        sp1 = tile_map[0, key[1].stop][..., 1] if arr.split == 0 else tile_map[0, key[1].stop][..., 1] - tile_map[0, prev_to_split][..., 1]
                    except IndexError:
                        sp1 = key[1].stop

                # print(st0, sp0, st1, sp1)
                return local_arr[st0:sp0, st1:sp1]
            else:
                return None

    def local_get(self, key, proc=None):
        # this is to be used with only local indices!
        # convert from local to global?
        proc = proc if proc is not None else self.__DNDarray.comm.rank
        if proc == self.__DNDarray.comm.rank:
            arr = self.__DNDarray
            tile_map = self.__tile_map
            rank_slice = torch.where(tile_map[..., 2] == proc)

            # need to convert the key into local indices -> only needs to be done on the split dimension
            key = list(key)
            if isinstance(key, int):
                key = [key, slice(0, arr.gshape[1])]
            elif isinstance(key, slice):
                key = [key, slice(0, arr.gshape[1])]

            if arr.split == 0:
                # need to adjust key[0] to be only on the local tensor
                prev_rows = sum(self.__row_per_proc_list[:proc])
                loc_rows = self.__row_per_proc_list[proc]
                if isinstance(key[1], int):
                    key[0] += prev_rows
                elif isinstance(key[0], slice):
                    start = key[0].start + prev_rows
                    stop = key[0].stop + prev_rows
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
            self.__getitem__(key)

    def __setitem__(self, key, value):
        """
        Item setter, this uses the torch item setter and the getitem routines to set the values of the original array (arr in __init__)

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
        # print(tile_map[key][..., 2].unique())
        if arr.comm.rank == tile_map[key][..., 2].unique():
            # this will set the tile values using the torch setitem function
            self.__getitem__(key).__setitem__(slice(None), value)

    def get_start_stop(self, key):
        """
        Very similar to the getitem routine, returns the start and stop indices which correspond to the selected tile

        Parameters
        ----------
        key : int, tuple, list, slice
            indices to select the tile
            STRIDES ARE NOT ALLOWED

        Returns
        -------
        tuple : dim0 start, dim0 stop, dim1 start, dim1 stop
        """
        # default getitem will return the data in the array!!
        # this is intended to return tiles which are local. it will return torch.Tensors which correspond to the tiles of the array
        # this is a global getter, if the tile is not on the process then it will return None
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
                return st0, sp0, 0, arr.gshape[1]
        elif tile_map[key][..., 2].unique().nelement() > 1:
            raise ValueError('Slicing across splits is not allowed')
        else:
            if arr.comm.rank == tile_map[key][..., 2].unique():
                if not isinstance(key, (tuple, list, slice)):
                    raise TypeError('key must be an int, tuple, or slice, is currently {}'.format(type(key)))

                if isinstance(key, slice):
                    key = tuple(key, slice(0, None))
                    self.__getitem__(key)

                key = list(key)
                if all(isinstance(x, int) for x in key):
                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    st0 = tile_map[key[0], 0][..., 0] if arr.split == 1 else tile_map[key[0], 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    sp0 = tile_map[key[0] + 1, 0][..., 0] if arr.split == 1 else tile_map[key[0] + 1, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    st1 = tile_map[0, key[1]][..., 1] if arr.split == 0 else tile_map[0, key[1]][..., 1] - tile_map[0, prev_to_split][..., 1]
                    sp1 = tile_map[0, key[1] + 1][..., 1] if arr.split == 0 else tile_map[0, key[1] + 1][..., 1] - tile_map[0, prev_to_split][..., 1]

                elif isinstance(key[arr.split], slice) and isinstance(key[(arr.split + 1) % len(arr.gshape)], int):
                    # note: strides are not implemented! todo: add to docs
                    slice_dim = arr.split
                    # this is to change from global to local, take the mode of how many tiles are in the split dimension
                    start = key[slice_dim].start if key[slice_dim].start is not None else 0
                    stop = key[slice_dim].stop if key[slice_dim].stop is not None else arr.gshape[slice_dim]
                    key[slice_dim] = slice(start, stop)

                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    if arr.split == 0:
                        st0 = tile_map[key[0].start, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        try:
                            sp0 = tile_map[key[0].stop, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        except IndexError:
                            sp0 = key[0].stop
                        st1 = tile_map[0, key[1]][..., 1]
                        sp1 = tile_map[0, key[1] + 1][..., 1]
                    if arr.split == 1:
                        st0 = tile_map[key[0], 0][..., 0]
                        sp0 = tile_map[key[0] + 1, 0][..., 0]
                        st1 = tile_map[0, key[1].start][..., 1] - tile_map[0, prev_to_split][..., 1]
                        try:
                            sp1 = tile_map[0, key[1].stop][..., 1] - tile_map[0, prev_to_split][..., 1]
                        except IndexError:
                            sp1 = key[1].stop
                elif isinstance(key[arr.split], int) and isinstance(key[(arr.split + 1) % len(arr.gshape)], slice):
                    # this implies that the other axis is a slice -> key = (int, slice) for split = 0
                    slice_dim = (arr.split + 1) % len(arr.gshape)
                    # this is to change from global to local, take the mode of how many tiles are in the split dimension
                    start = key[slice_dim].start if key[slice_dim].start is not None else 0
                    stop = key[slice_dim].stop if key[slice_dim].stop is not None else arr.gshape[slice_dim]
                    key[slice_dim] = slice(start, stop)

                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    if arr.split == 0:
                        st0 = tile_map[key[0], 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        sp0 = tile_map[key[0] + 1, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                        st1 = tile_map[0, key[1].start][..., 1]
                        try:
                            sp1 = tile_map[0, key[1].stop][..., 1]
                        except IndexError:
                            sp1 = key[1].stop
                    if arr.split == 1:
                        st0 = tile_map[key[0].start, 0][..., 0]
                        try:
                            sp0 = tile_map[key[0].stop, 0][..., 1]
                        except IndexError:
                            sp0 = key[1].stop
                        st1 = tile_map[0, key[1]][..., 1] - tile_map[0, prev_to_split][..., 1]
                        sp1 = tile_map[0, key[1] + 1][..., 1] - tile_map[0, prev_to_split][..., 1]
                else:  # all slices
                    start = key[arr.split].start if key[arr.split].start is not None else 0
                    stop = key[arr.split].stop if key[arr.split].stop is not None else arr.gshape[arr.split]
                    start2 = key[(arr.split + 1) % len(arr.gshape)].start if key[(arr.split + 1) % len(arr.gshape)].start is not None else 0
                    stop2 = key[(arr.split + 1) % len(arr.gshape)].stop if key[(arr.split + 1) % len(arr.gshape)].stop is not None \
                        else arr.gshape[(arr.split + 1) % len(arr.gshape)]
                    key[(arr.split + 1) % len(arr.gshape)] = slice(start2, stop2)
                    key[arr.split] = slice(start, stop)

                    # rank_slice = torch.where(tile_map[..., 2] == arr.comm.rank)
                    # only need to know how many columns are before the start of the key on the local column
                    prev_to_split = sum(self.__row_per_proc_list[:arr.comm.rank]) if arr.split == 0 else sum(self.__col_per_proc_list[:arr.comm.rank])
                    st0 = tile_map[key[0].start, 0][..., 0] if arr.split == 1 else tile_map[key[0].start, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    try:
                        sp0 = tile_map[key[0].stop, 0][..., 0] if arr.split == 1 else tile_map[key[0].stop, 0][..., 0] - tile_map[prev_to_split, 0][..., 0]
                    except IndexError:
                        sp0 = key[0].stop
                    st1 = tile_map[0, key[1].start][..., 1] if arr.split == 0 else tile_map[0, key[1].start][..., 1] - tile_map[0, prev_to_split][..., 1]
                    try:
                        sp1 = tile_map[0, key[1].stop][..., 1] if arr.split == 0 else tile_map[0, key[1].stop][..., 1] - tile_map[0, prev_to_split][..., 1]
                    except IndexError:
                        sp1 = key[1].stop

                return st0, sp0, st1, sp1

    def get_tile_proc(self, key):
        return self.tile_map[key][..., 2].unique()

    def get_tile_size(self, key):
        """
        Returns
        -------
        torch.Shape : uses the getitem routine then calls the torch shape function
        """
        return self.__getitem__(key).shape

    # todo: get_start, get_end, asynce_get, async_set, docs, global->local convert
    # tile start
    # tile end
