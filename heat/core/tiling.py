import torch

from . import dndarray

__all__ = ["SquareDiagTiles"]


class SquareDiagTiles:
    def __init__(self, arr, tiles_per_proc=2):
        """
        Generate the tile map and the other objects which may be useful.
        The tiles generated here are based of square tiles along the diagonal. The size of these
        tiles along the diagonal dictate the divisions accross all processes. If
        gshape[0] >> gshape[1] then there will be extra tiles generated below the diagonal.
        If gshape[0] is close to gshape[1], then the last tile (as well as the other tiles which
        correspond with said tile) will be extended to cover the whole array. However, extra tiles
        are not generated above the diagonal in the case that gshape[0] << gshape[1].

        WARNING: The generation of these tiles may unbalance the original tensor!

        Note: This tiling scheme is intended for use with the QR function.

        Parameters
        ----------
        arr : DNDarray
            the array to be tiled
        tiles_per_proc : int
            Default = 2
            the number of divisions per process,

        Returns
        -------
        None

        Initializes
        -----------
        __col_per_proc_list : list
            list is length of the number of processes, each element has the number of tile
            columns on the process whos rank equals the index
        __DNDarray = arr : DNDarray
            the whole DNDarray
        __lshape_map : torch.Tensor
            unit -> [rank, row size, column size]
            tensor filled with the shapes of the local tensors
        __tile_map : torch.Tensor
            units -> row, column, start index in each direction, process
            tensor filled with the global indices of the generated tiles
        __row_per_proc_list : list
            list is length of the number of processes, each element has the number of tile
            rows on the process whos rank equals the index
        """
        # lshape_map -> rank (int), lshape (tuple of the local lshape, self.lshape)
        if not isinstance(arr, dndarray.DNDarray):
            raise TypeError("arr must be a DNDarray, is currently a {}".format(type(self)))
        if not isinstance(tiles_per_proc, int):
            raise TypeError("tiles_per_proc must be an int, is currently a {}".format(type(self)))
        if tiles_per_proc < 1:
            raise ValueError("Tiles per process must be >= 1, currently: {}".format(tiles_per_proc))
        if len(arr.shape) != 2:
            raise ValueError("Arr must be 2 dimensional, current shape {}".format(arr.shape))

        lshape_map = arr.create_lshape_map()

        # if there is only one element of the diagonal on the next process
        d = 1 if tiles_per_proc <= 2 else tiles_per_proc - 1
        redist = torch.where(
            torch.cumsum(lshape_map[..., arr.split], dim=0) >= arr.gshape[arr.split - 1] - d
        )[0]
        if redist.numel() > 0 and arr.gshape[0] > arr.gshape[1] and redist[0] != arr.comm.size - 1:
            target_map = lshape_map.clone()
            target_map[redist[0]] += d
            target_map[redist[0] + 1] -= d
            # print(target_map)
            arr.redistribute_(lshape_map=lshape_map, target_map=target_map)

        row_per_proc_list = [tiles_per_proc] * arr.comm.size

        last_diag_pr, col_per_proc_list, col_inds, tile_columns = self._create_cols(
            arr, lshape_map, tiles_per_proc
        )

        # need to adjust the lshape if the splits overlap...
        if arr.split == 0 and tiles_per_proc == 1:
            # if the split is 0 and the number of tiles per proc is 1
            # then the local data needs to be redistributed to fit the full diagonal on as many
            #       processes as possible....i think
            for i, c in enumerate(col_inds[:-1]):  # only need to loop until the second to last one
                for l in range(arr.comm.size - 1):
                    if lshape_map[..., 0][l] < c:
                        h = c - lshape_map[..., 0][l]
                        lshape_map[..., 0][l] += h
                        lshape_map[..., 0][l + 1] -= h
            arr.redistribute_(target_map=lshape_map)

            last_diag_pr, col_per_proc_list, col_inds, tile_columns = self._create_cols(
                arr, lshape_map, tiles_per_proc
            )

        total_tile_rows = tiles_per_proc * arr.comm.size
        row_inds = [0] * total_tile_rows
        for c, x in enumerate(col_inds):
            # set the row indices to be the same for all of the column indices
            #   (however many there are)
            row_inds[c] = x

        if arr.gshape[0] < arr.gshape[1] and arr.split == 0:
            # need to adjust the very last tile to be the remaining
            col_inds[-1] = arr.gshape[1] - sum(col_inds[:-1])

        # if there is too little data on the last tile then combine them
        if last_diag_pr < arr.comm.size - 1 and arr.split == 0:
            # these conditions imply that arr.gshape[0] > arr.gshape[1] (assuming balanced)
            # need to find the amount of data after the diagonal
            lshape_cumsum = torch.cumsum(lshape_map[..., 0], dim=0)
            diff = lshape_cumsum[last_diag_pr] - arr.gshape[1]
            if diff > lshape_map[last_diag_pr, 0] / 2:  # todo: tune this?
                # if the shape diff is > half the data on the process
                #   then add a row after the diagonal, todo: is multiple rows faster?
                row_inds.insert(tile_columns, diff)
                row_per_proc_list[last_diag_pr] += 1
            else:
                # if the diff is < half the data on the process
                #   then extend the last row inds to be the end of the process
                row_inds[tile_columns - 1] += diff

        if arr.split == 1 and arr.gshape[1] > arr.gshape[0]:
            # need to add to col inds witht the rest of the columns
            tile_columns = sum(col_per_proc_list)
            r = last_diag_pr + 1
            for i in range(len(col_inds), tile_columns):
                col_inds.append(lshape_map[r, 1])
                r += 1
            # if the 1st dim is > 0th dim then in split=1 the cols need to be extended
            col_proc_ind = torch.cumsum(torch.tensor(col_per_proc_list), dim=0)
            for pr in range(arr.comm.size):
                lshape_cumsum = torch.cumsum(lshape_map[..., 1], dim=0)
                col_cumsum = torch.cumsum(torch.tensor(col_inds), dim=0)
                diff = lshape_cumsum[pr] - col_cumsum[col_proc_ind[pr] - 1]
                if diff > 0 and pr <= last_diag_pr:
                    col_per_proc_list[pr] += 1
                    col_inds.insert(col_proc_ind[pr], diff)
                if pr > last_diag_pr and diff > 0:
                    col_inds.insert(col_proc_ind[pr], diff)

        if arr.gshape[0] > arr.gshape[1]:
            nz = torch.nonzero(torch.tensor(row_inds) == 0)
            for i in range(last_diag_pr.item() + 1, arr.comm.size):
                # loop over all of the rest of the processes
                for t in range(tiles_per_proc):
                    _, lshape, _ = arr.comm.chunk(lshape_map[i], 0, rank=t, w_size=tiles_per_proc)
                    row_inds[nz[0].item()] = lshape[0]
                    nz = nz[1:]

        # add extra rows if there is place below the diagonal for split == 1
        if arr.gshape[0] > arr.gshape[1] and arr.split == 1:
            # need to adjust the very last tile to be the remaining
            if arr.gshape[0] - arr.gshape[1] > 10:  # todo: determine best value for this
                # use chunk and a loop over the however many tiles are desired
                num_ex_row_tiles = 1  # todo: determine best value for this
                while (arr.gshape[0] - arr.gshape[1]) // num_ex_row_tiles < 2:
                    num_ex_row_tiles -= 1
                for i in range(num_ex_row_tiles):
                    _, lshape, _ = arr.comm.chunk(
                        (arr.gshape[0] - arr.gshape[1],), 0, rank=i, w_size=num_ex_row_tiles
                    )
                    row_inds.append(lshape[0])
            else:
                # if there is no place for multiple tiles, combine the remainder with the last row
                row_inds[-1] = arr.gshape[0] - sum(row_inds[:-1])

        # need to remove blank rows for arr.gshape[0] < arr.gshape[1]
        if arr.gshape[0] < arr.gshape[1]:
            row_inds_hold = []
            for i in torch.nonzero(torch.tensor(row_inds)).flatten():
                row_inds_hold.append(row_inds[i.item()])
            row_inds = row_inds_hold

        tile_map = torch.zeros([len(row_inds), len(col_inds), 3], dtype=torch.int)
        # if arr.split == 0:  # adjust the 1st dim to be the cumsum
        col_inds = [0] + col_inds[:-1]
        col_inds = torch.tensor(col_inds).cumsum(dim=0)
        # if arr.split == 1:  # adjust the 0th dim to be the cumsum
        row_inds = [0] + row_inds[:-1]
        row_inds = torch.tensor(row_inds).cumsum(dim=0)

        for num, c in enumerate(col_inds):  # set columns
            tile_map[:, num, 1] = c
        for num, r in enumerate(row_inds):  # set rows
            tile_map[num, :, 0] = r

        for i in range(arr.comm.size):
            st = sum(row_per_proc_list[:i])
            sp = st + row_per_proc_list[i]
            tile_map[..., 2][st:sp] = i
        # to adjust if the last process has more tiles
        i = arr.comm.size - 1
        tile_map[..., 2][sum(row_per_proc_list[:i]) :] = i

        if arr.split == 1:
            st = 0
            for pr, cols in enumerate(col_per_proc_list):
                tile_map[:, st : st + cols, 2] = pr
                st += cols

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
        # =========================================================================================
        self.__DNDarray = arr
        self.__col_per_proc_list = (
            col_per_proc_list if arr.split == 1 else [len(col_inds)] * len(col_per_proc_list)
        )
        self.__lshape_map = lshape_map
        self.__last_diag_pr = last_diag_pr.item()
        self.__row_per_proc_list = (
            row_per_proc_list if arr.split == 0 else [len(row_inds)] * len(row_per_proc_list)
        )
        self.__tile_map = tile_map
        self.__row_inds = list(row_inds)
        self.__col_inds = list(col_inds)
        # =========================================================================================

    def _create_cols(self, arr, lshape_map, tiles_per_proc):
        last_tile_cols = tiles_per_proc
        last_dia_pr = torch.where(lshape_map[..., arr.split].cumsum(dim=0) >= min(arr.gshape))[0][0]

        # adjust for small blocks on the last diag pr:
        last_pr_minus1 = last_dia_pr - 1 if last_dia_pr > 0 else 0
        rem_cols_last_pr = abs(
            min(arr.gshape) - lshape_map[..., arr.split].cumsum(dim=0)[last_pr_minus1]
        )
        # this is the number of rows/columns after the last diagonal on the last diagonal pr

        while 1 < rem_cols_last_pr / last_tile_cols < 2:
            # todo: determine best value for this (prev at 2)
            # if there cannot be tiles formed which are at list ten items larger than 2
            #   then need to reduce the number of tiles
            last_tile_cols -= 1
            if last_tile_cols == 1:
                break
        # create lists of columns and rows for each process
        col_per_proc_list = [tiles_per_proc] * (last_dia_pr.item() + 1)
        col_per_proc_list[-1] = last_tile_cols

        if last_dia_pr < arr.comm.size - 1 and arr.split == 1:
            # this is the case that the gshape[1] >> gshape[0]
            col_per_proc_list.extend([1] * (arr.comm.size - last_dia_pr - 1).item())
        # need to determine the proper number of tile rows/columns
        tile_columns = tiles_per_proc * last_dia_pr + last_tile_cols
        diag_crossings = lshape_map[..., arr.split].cumsum(dim=0)[: last_dia_pr + 1]
        diag_crossings[-1] = (
            diag_crossings[-1] if diag_crossings[-1] <= min(arr.gshape) else min(arr.gshape)
        )
        diag_crossings = torch.cat((torch.tensor([0]), diag_crossings), dim=0)
        # create the tile columns sizes, saved to list
        col_inds = []
        for col in range(tile_columns.item()):
            _, lshape, _ = arr.comm.chunk(
                [diag_crossings[col // tiles_per_proc + 1] - diag_crossings[col // tiles_per_proc]],
                0,
                rank=int(col % tiles_per_proc),
                w_size=tiles_per_proc if col // tiles_per_proc != last_dia_pr else last_tile_cols,
            )
            col_inds.append(lshape[0])
        return last_dia_pr, col_per_proc_list, col_inds, tile_columns

    @property
    def arr(self):
        """
        Returns
        -------
        DNDarray : the DNDarray for which the tiles are defined on
        """
        return self.__DNDarray

    @property
    def col_indices(self):
        """
        Returns
        -------
        list : list containing the indices of the tile columns
        """
        return self.__col_inds

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
        Returns
        -------
        int : the rank of the last process with diagonal elements
        """
        return self.__last_diag_pr

    @property
    def row_indices(self):
        """
        Returns
        -------
        list : list containing the indices of the tile rows
        """
        return self.__row_inds

    @property
    def tile_columns(self):
        """
        Returns
        -------
        int : number of tile columns
        """
        return len(self.__col_inds)

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

        Examples
        --------
        >>> a = ht.zeros((12, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)
        >>> print(a_tiles.tile_map)
        [(0 & 1)/1] tensor([[[0, 0, 0],
        [(0 & 1)/1]          [0, 3, 0],
        [(0 & 1)/1]          [0, 6, 0],
        [(0 & 1)/1]          [0, 8, 0]],
        [(0 & 1)/1]
        [(0 & 1)/1]         [[3, 0, 0],
        [(0 & 1)/1]          [3, 3, 0],
        [(0 & 1)/1]          [3, 6, 0],
        [(0 & 1)/1]          [3, 8, 0]],
        [(0 & 1)/1]
        [(0 & 1)/1]         [[6, 0, 1],
        [(0 & 1)/1]          [6, 3, 1],
        [(0 & 1)/1]          [6, 6, 1],
        [(0 & 1)/1]          [6, 8, 1]],
        [(0 & 1)/1]
        [(0 & 1)/1]         [[8, 0, 1],
        [(0 & 1)/1]          [8, 3, 1],
        [(0 & 1)/1]          [8, 6, 1],
        [(0 & 1)/1]          [8, 8, 1]]], dtype=torch.int32)
        >>> print(a_tiles.tile_map.shape)
        [0/1] torch.Size([4, 4, 3])
        [1/1] torch.Size([4, 4, 3])
        """
        return self.__tile_map

    @property
    def tile_rows(self):
        """
        Returns
        -------
        int : number of tile rows
        """
        return len(self.__row_inds)

    @property
    def tile_rows_per_process(self):
        """
        Returns
        -------
        list : list containing the number of rows on all processes
        """
        return self.__row_per_proc_list

    def get_start_stop(self, key):
        """
        Returns the start and stop indices which correspond to the tile/s which corresponds to the
        given key. The key MUST use global indices.

        Parameters
        ----------
        key : int, tuple, list, slice
            indices to select the tile
            STRIDES ARE NOT ALLOWED, MUST BE GLOBAL INDICES

        Returns
        -------
        tuple : (dim0 start, dim0 stop, dim1 start, dim1 stop)

        Examples
        --------
        >>> a = ht.zeros((12, 10), split=0)
        >>> a_tiles = ht.tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> print(a_tiles.get_start_stop(key=(slice(0, 2), 2)))
        [0/1] (tensor(0), tensor(6), tensor(6), tensor(8))
        [1/1] (tensor(0), tensor(6), tensor(6), tensor(8))
        >>> print(a_tiles.get_start_stop(key=(0, 2)))
        [0/1] (tensor(0), tensor(3), tensor(6), tensor(8))
        [1/1] (tensor(0), tensor(3), tensor(6), tensor(8))
        >>> print(a_tiles.get_start_stop(key=2))
        [0/1] (tensor(0), tensor(2), tensor(0), tensor(10))
        [1/1] (tensor(0), tensor(2), tensor(0), tensor(10))
        >>> print(a_tiles.get_start_stop(key=(3, 3)))
        [0/1] (tensor(2), tensor(6), tensor(8), tensor(10))
        [1/1] (tensor(2), tensor(6), tensor(8), tensor(10))
        """
        split = self.__DNDarray.split
        pr = self.tile_map[key][..., 2].unique()
        if pr.numel() > 1:
            raise ValueError("Tile/s must be located on one process. currently on: {}".format(pr))
        row_inds = self.row_indices + [self.__DNDarray.gshape[0]]
        col_inds = self.col_indices + [self.__DNDarray.gshape[1]]

        row_start = row_inds[sum(self.tile_rows_per_process[:pr]) if split == 0 else 0]
        col_start = col_inds[sum(self.tile_columns_per_process[:pr]) if split == 1 else 0]

        if isinstance(key, int):
            key = [key]
        else:
            key = list(key)

        if len(key) == 1:
            key.append(slice(0, None))

        # only need to do this in 2 dimensions (this class is only for 2D right now)
        if not isinstance(key[0], (int, slice)):
            raise TypeError(
                "Key elements must be ints, or slices; currently {}".format(type(key[0]))
            )
        if not isinstance(key[1], (int, slice)):
            raise TypeError(
                "Key elements must be ints, or slices; currently {}".format(type(key[1]))
            )

        key = list(key)
        if isinstance(key[0], int):
            st0 = row_inds[key[0]] - row_start
            sp0 = row_inds[key[0] + 1] - row_start
        elif isinstance(key[0], slice):
            start = row_inds[key[0].start] if key[0].start is not None else 0
            stop = row_inds[key[0].stop] if key[0].stop is not None else row_inds[-1]
            st0, sp0 = start - row_start, stop - row_start
        if isinstance(key[1], int):
            st1 = col_inds[key[1]] - col_start
            sp1 = col_inds[key[1] + 1] - col_start
        elif isinstance(key[1], slice):
            start = col_inds[key[1].start] if key[1].start is not None else 0
            stop = col_inds[key[1].stop] if key[1].stop is not None else col_inds[-1]
            st1, sp1 = start - col_start, stop - col_start

        return st0, sp0, st1, sp1

    def __getitem__(self, key):
        """
        Standard getitem function for the tiles. The returned item is a view of the original
        DNDarray, operations which are done to this view will change the original array.
        **STRIDES ARE NOT AVAILABLE, NOR ARE CROSS-SPLIT SLICES**

        Parameters
        ----------
        key : int, slice, tuple, list
            indices of the tile/s desired

        Returns
        -------
        DNDarray_view : torch.Tensor
            A local selection of the DNDarray corresponding to the tile/s desired

        Examples
        --------
        >>> a = ht.zeros((12, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> print(a_tiles[2, 3])
        [0/1] None
        [1/1] tensor([[0., 0.],
        [1/1]         [0., 0.]])
        >>> print(a_tiles[2])
        [0/1] None
        [1/1] tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1/1]         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        >>> print(a_tiles[0:2, 1])
        [0/1] tensor([[0., 0., 0.],
        [0/1]         [0., 0., 0.],
        [0/1]         [0., 0., 0.],
        [0/1]         [0., 0., 0.],
        [0/1]         [0., 0., 0.],
        [0/1]         [0., 0., 0.]])
        [1/1] None
        """
        arr = self.__DNDarray
        tile_map = self.__tile_map
        local_arr = arr._DNDarray__array
        if tile_map[key][..., 2].unique().nelement() > 1:
            raise ValueError("Slicing across splits is not allowed")
        # early outs (returns nothing if the tile does not exist on the process)
        if tile_map[key][..., 2].unique().nelement() == 0:
            return None
        if arr.comm.rank != tile_map[key][..., 2].unique():
            return None

        if not isinstance(key, (int, tuple, slice)):
            raise TypeError(
                "key must be an int, tuple, or slice, is currently {}".format(type(key))
            )
        st0, sp0, st1, sp1 = self.get_start_stop(key=key)
        return local_arr[st0:sp0, st1:sp1]

    def local_get(self, key):
        """
        Getitem routing using local indices, converts to global indices then uses getitem

        Parameters
        ----------
        key : int, slice, tuple, list
            indices of the tile/s desired
            if the stop index of a slice is larger than the end will be adjusted to the maximum
            allowed

        Returns
        -------
        torch.Tensor : the local tile/s corresponding to the key given

        Examples
        --------
        See local_set function.
        """
        rank = self.__DNDarray.comm.rank
        key = self.local_to_global(key=key, rank=rank)
        return self.__getitem__(key)

    def local_set(self, key, value):
        """
        Setitem routing to set data to a local tile (using local indices)

        Parameters
        ----------
        key : int, slice, tuple, list
            indices of the tile/s desired
            if the stop index of a slice is larger than the end will be adjusted to the maximum
            allowed
        value : torch.Tensor, int, float
            data to be written to the tile

        Returns
        -------
        None

        Examples
        --------
        >>> a = ht.zeros((11, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> local = a_tiles.local_get(key=slice(None))
        >>> a_tiles.local_set(key=slice(None), value=torch.arange(local.numel()).reshape(local.shape))
        >>> print(a)
        [0/1] tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
        [0/1]         [10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        [0/1]         [20., 21., 22., 23., 24., 25., 26., 27., 28., 29.],
        [0/1]         [30., 31., 32., 33., 34., 35., 36., 37., 38., 39.],
        [0/1]         [40., 41., 42., 43., 44., 45., 46., 47., 48., 49.],
        [0/1]         [50., 51., 52., 53., 54., 55., 56., 57., 58., 59.]])
        [1/1] tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
        [1/1]         [10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        [1/1]         [20., 21., 22., 23., 24., 25., 26., 27., 28., 29.],
        [1/1]         [30., 31., 32., 33., 34., 35., 36., 37., 38., 39.],
        [1/1]         [40., 41., 42., 43., 44., 45., 46., 47., 48., 49.]])
        >>> a.lloc[:] = 0
        >>> a_tiles.local_set(key=(0, 2), value=10)
        [0/1] tensor([[ 0.,  0.,  0.,  0.,  0.,  0., 10., 10.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0., 10., 10.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0., 10., 10.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        [1/1] tensor([[ 0.,  0.,  0.,  0.,  0.,  0., 10., 10.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0., 10., 10.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        >>> a_tiles.local_set(key=(slice(None), 1), value=10)
        [0/1] tensor([[ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.]])
        [1/1] tensor([[ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.],
        [1/1]         [ 0.,  0.,  0., 10., 10., 10.,  0.,  0.,  0.,  0.]])

        """
        rank = self.__DNDarray.comm.rank
        key = self.local_to_global(key=key, rank=rank)
        self.__getitem__(tuple(key)).__setitem__(slice(0, None), value)

    def local_to_global(self, key, rank):
        """
        Convert local indices to global indices

        Parameters
        ----------
        key : int, slice, tuple, list
            indices of the tile/s desired
            if the stop index of a slice is larger than the end will be adjusted to the maximum
            allowed
        rank : process rank

        Returns
        -------
        tuple : key with global indices

        Examples
        --------
        >>> a = ht.zeros((11, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> rank = a.comm.rank
        >>> print(a_tiles.local_to_global(key=(slice(None), 1), rank=rank))
        [0] (slice(0, 2, None), 1)
        [1] (slice(2, 4, None), 1)
        >>> print(a_tiles.local_to_global(key=(0, 2), rank=0))
        [0] (0, 2)
        [1] (0, 2)
        >>> print(a_tiles.local_to_global(key=(0, 2), rank=1))
        [0] (2, 2)
        [1] (2, 2)
        """
        arr = self.__DNDarray
        if isinstance(key, (int, slice)):
            key = [key, slice(0, None)]
        else:
            key = list(key)

        if arr.split == 0:
            # need to adjust key[0] to be only on the local tensor
            prev_rows = sum(self.__row_per_proc_list[:rank])
            loc_rows = self.__row_per_proc_list[rank]
            if isinstance(key[0], int):
                key[0] += prev_rows
            elif isinstance(key[0], slice):
                start = key[0].start + prev_rows if key[0].start is not None else prev_rows
                stop = key[0].stop + prev_rows if key[0].stop is not None else prev_rows + loc_rows
                stop = stop if stop - start < loc_rows else start + loc_rows
                key[0] = slice(start, stop)

        if arr.split == 1:
            loc_cols = self.__col_per_proc_list[rank]
            prev_cols = sum(self.__col_per_proc_list[:rank])
            # need to adjust key[0] to be only on the local tensor
            # need the number of columns *before* the process
            if isinstance(key[1], int):
                key[1] += prev_cols
            elif isinstance(key[1], slice):
                start = key[1].start + prev_cols if key[1].start is not None else prev_cols
                stop = key[1].stop + prev_cols if key[1].stop is not None else prev_cols + loc_cols
                stop = stop if stop - start < loc_cols else start + loc_cols
                key[1] = slice(start, stop)
        return tuple(key)

    def match_tiles(self, tiles_to_match):
        """
        function to match the tile sizes of another tile map
        NOTE: this is intended for use with the Q matrix, to match the tiling of a/R
        For this to work properly it is required that the 0th dim of both matrices is equal

        Parameters
        ----------
        tiles_to_match : SquareDiagTiles
            the tiles which should be matched by the current tiling scheme

        Returns
        -------
        None

        Notes
        -----
        This function overwrites most, if not all, of the elements of this class
        """
        if not isinstance(tiles_to_match, SquareDiagTiles):
            raise TypeError(
                "tiles_to_match must be a SquareDiagTiles object, currently: {}".format(
                    type(tiles_to_match)
                )
            )
        base_dnd = self.__DNDarray
        match_dnd = tiles_to_match.__DNDarray
        # this map will take the same tile row and column sizes up to the last diagonal row/column
        # the last row/column is determined by the number of rows/columns on the non-split dimension
        if base_dnd.split == match_dnd.split == 0:
            # this implies that the gshape[0]'s are equal
            # rows are the exact same, and the cols are also equal to the rows (square matrix)
            base_dnd.redistribute_(lshape_map=self.lshape_map, target_map=tiles_to_match.lshape_map)

            self.__row_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
            self.__col_per_proc_list = [tiles_to_match.tile_rows] * len(self.__row_per_proc_list)
            self.__row_inds = (
                tiles_to_match.__row_inds.copy()
                if base_dnd.gshape[0] >= base_dnd.gshape[1]
                else tiles_to_match.__col_inds.copy()
            )
            self.__col_inds = (
                tiles_to_match.__row_inds.copy()
                if base_dnd.gshape[0] >= base_dnd.gshape[1]
                else tiles_to_match.__col_inds.copy()
            )
            # self.__tile_rows = tiles_to_match.__tile_rows
            # self.__tile_columns = tiles_to_match.__tile_rows
            self.__tile_map = torch.zeros((self.tile_rows, self.tile_columns, 3), dtype=torch.int)
            for i in range(self.tile_rows):
                self.__tile_map[..., 0][i] = self.__row_inds[i]
            for i in range(self.tile_columns):
                self.__tile_map[..., 1][:, i] = self.__col_inds[i]
            for i in range(self.arr.comm.size - 1):
                st = sum(self.__row_per_proc_list[:i])
                sp = st + self.__row_per_proc_list[i]
                self.__tile_map[..., 2][st:sp] = i
            # to adjust if the last process has more tiles
            i = self.arr.comm.size - 1
            self.__tile_map[..., 2][sum(self.__row_per_proc_list[:i]) :] = i

        if base_dnd.split == 0 and match_dnd.split == 1:
            # rows determine the q sizes -> cols = rows
            self.__col_inds = (
                tiles_to_match.__row_inds.copy()
                if base_dnd.gshape[0] <= base_dnd.gshape[1]
                else tiles_to_match.__col_inds.copy()
            )
            self.__row_inds = (
                tiles_to_match.__row_inds.copy()
                if base_dnd.gshape[0] <= base_dnd.gshape[1]
                else tiles_to_match.__col_inds.copy()
            )

            rows_per = [x for x in self.__col_inds if x < base_dnd.shape[0]]
            # self.__tile_rows = len(rows_per)
            # self.__tile_columns = self.tile_rows

            target_0 = tiles_to_match.lshape_map[..., 1][: tiles_to_match.last_diagonal_process]
            end_tag0 = base_dnd.shape[0] - sum(target_0[: tiles_to_match.last_diagonal_process])
            end_tag0 = [end_tag0] + [0] * (
                base_dnd.comm.size - 1 - tiles_to_match.last_diagonal_process
            )
            target_0 = torch.cat((target_0, torch.tensor(end_tag0)), dim=0)

            targe_map = self.lshape_map.clone()
            targe_map[..., 0] = target_0
            target_0_c = torch.cumsum(target_0, dim=0)
            self.__row_per_proc_list = []
            st = 0
            rows_per = torch.tensor(rows_per + [base_dnd.shape[0]])
            for i in range(base_dnd.comm.size):
                # get the amount of data on each process, get the number of rows with
                # indices which are between the start and stop
                self.__row_per_proc_list.append(
                    torch.where((st < rows_per) & (rows_per <= target_0_c[i]))[0].numel()
                )
                st = target_0_c[i]

            base_dnd.redistribute_(lshape_map=self.lshape_map, target_map=targe_map)

            self.__tile_map = torch.zeros((self.tile_rows, self.tile_columns, 3), dtype=torch.int)
            for i in range(self.tile_rows):
                self.__tile_map[..., 0][i] = self.__row_inds[i]
            for i in range(self.tile_columns):
                self.__tile_map[..., 1][:, i] = self.__col_inds[i]
            for i in range(self.arr.comm.size):
                st = sum(self.__row_per_proc_list[:i])
                sp = st + self.__row_per_proc_list[i]
                self.__tile_map[..., 2][st:sp] = i
            # to adjust if the last process has more tiles
            i = self.arr.comm.size - 1

            self.__tile_map[..., 2][sum(self.__row_per_proc_list[:i]) :] = i
            self.__col_per_proc_list = [self.tile_columns] * base_dnd.comm.size
            self.__last_diag_pr = base_dnd.comm.size - 1

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

        Example
        -------
        >>> a = ht.zeros((12, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> a_tiles[0:2, 2] = 11
        >>> a_tiles[0, 0] = 22
        >>> a_tiles[2] = 33
        >>> a_tiles[3, 3] = 44
        >>> print(a)
        [0/1] tensor([[22., 22., 22.,  0.,  0.,  0., 11., 11.,  0.,  0.],
        [0/1]         [22., 22., 22.,  0.,  0.,  0., 11., 11.,  0.,  0.],
        [0/1]         [22., 22., 22.,  0.,  0.,  0., 11., 11.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0., 11., 11.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0., 11., 11.,  0.,  0.],
        [0/1]         [ 0.,  0.,  0.,  0.,  0.,  0., 11., 11.,  0.,  0.]])
        [1/1] tensor([[33., 33., 33., 33., 33., 33., 33., 33., 33., 33.],
        [1/1]         [33., 33., 33., 33., 33., 33., 33., 33., 33., 33.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 44., 44.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 44., 44.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 44., 44.],
        [1/1]         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 44., 44.]])

        """
        arr = self.__DNDarray
        tile_map = self.__tile_map
        if tile_map[key][..., 2].unique().numel() > 1:
            raise ValueError("setting across splits is not allowed")
        if arr.comm.rank == tile_map[key][..., 2].unique():
            # this will set the tile values using the torch setitem function
            arr = self.__getitem__(key)
            arr.__setitem__(slice(0, None), value)
