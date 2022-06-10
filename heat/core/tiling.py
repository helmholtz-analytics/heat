"""
Tiling functions/classes. With these classes, you can classes you can address blocks of data in a DNDarray
"""

from __future__ import annotations
import torch
from typing import List, Tuple, Union

from .dndarray import DNDarray

__all__ = ["SplitTiles", "SquareDiagTiles"]


class SplitTiles:
    """
    Initialize tiles with the tile divisions equal to the theoretical split dimensions in
    every dimension

    Parameters
    ----------
    arr : DNDarray
        Base array for which to create the tiles

    Attributes
    ----------
    __DNDarray : DNDarray
        the ``DNDarray`` associated with the tiles
    __lshape_map : torch.Tensor
        map of the shapes of the local torch tensors of arr
    __tile_locations : torch.Tensor
        locations of the tiles of ``arr``
    __tile_ends_g : torch.Tensor
        the global indices of the ends of the tiles
    __tile_dims : torch.Tensor
        the dimensions of all of the tiles

    Examples
    --------
    >>> a = ht.zeros((10, 11,), split=None)
    >>> a.create_split_tiles()
    >>> print(a.tiles.tile_ends_g)
    [0/2] tensor([[ 4,  7, 10],
    [0/2]         [ 4,  8, 11]], dtype=torch.int32)
    [1/2] tensor([[ 4,  7, 10],
    [1/2]         [ 4,  8, 11]], dtype=torch.int32)
    [2/2] tensor([[ 4,  7, 10],
    [2/2]         [ 4,  8, 11]], dtype=torch.int32)
    >>> print(a.tiles.tile_locations)
    [0/2] tensor([[0, 0, 0],
    [0/2]         [0, 0, 0],
    [0/2]         [0, 0, 0]], dtype=torch.int32)
    [1/2] tensor([[1, 1, 1],
    [1/2]         [1, 1, 1],
    [1/2]         [1, 1, 1]], dtype=torch.int32)
    [2/2] tensor([[2, 2, 2],
    [2/2]         [2, 2, 2],
    [2/2]         [2, 2, 2]], dtype=torch.int32)
    >>> a = ht.zeros((10, 11), split=1)
    >>> a.create_split_tiles()
    >>> print(a.tiles.tile_ends_g)
    [0/2] tensor([[ 4,  7, 10],
    [0/2]         [ 4,  8, 11]], dtype=torch.int32)
    [1/2] tensor([[ 4,  7, 10],
    [1/2]         [ 4,  8, 11]], dtype=torch.int32)
    [2/2] tensor([[ 4,  7, 10],
    [2/2]         [ 4,  8, 11]], dtype=torch.int32)
    >>> print(a.tiles.tile_locations)
    [0/2] tensor([[0, 1, 2],
    [0/2]         [0, 1, 2],
    [0/2]         [0, 1, 2]], dtype=torch.int32)
    [1/2] tensor([[0, 1, 2],
    [1/2]         [0, 1, 2],
    [1/2]         [0, 1, 2]], dtype=torch.int32)
    [2/2] tensor([[0, 1, 2],
    [2/2]         [0, 1, 2],
    [2/2]         [0, 1, 2]], dtype=torch.int32)
    """

    def __init__(self, arr: DNDarray) -> None:  # noqa: D107
        #  1. get the lshape map
        #  2. get the split axis numbers for the other axes
        #  3. build tile map
        lshape_map = arr.create_lshape_map()
        tile_dims = torch.zeros((arr.ndim, arr.comm.size), device=arr.device.torch_device)
        if arr.split is not None:
            tile_dims[arr.split] = lshape_map[..., arr.split]
        w_size = arr.comm.size
        for ax in range(arr.ndim):
            if arr.split is None or not ax == arr.split:
                size = arr.gshape[ax]
                chunk = size // w_size
                remainder = size % w_size
                tile_dims[ax] = chunk
                tile_dims[ax][:remainder] += 1

        tile_ends_g = torch.cumsum(tile_dims, dim=1).int()
        # tile_ends_g is the global end points of the tiles in each dimension
        # create a tensor for the process rank of all the tiles
        tile_locations = self.set_tile_locations(split=arr.split, tile_dims=tile_dims, arr=arr)

        self.__DNDarray = arr
        self.__lshape_map = lshape_map
        self.__tile_locations = tile_locations
        self.__tile_ends_g = tile_ends_g
        self.__tile_dims = tile_dims

    @staticmethod
    def set_tile_locations(split: int, tile_dims: torch.Tensor, arr: DNDarray) -> torch.Tensor:
        """
        Create a `torch.Tensor` which contains the locations of the tiles of ``arr`` for the given split

        Parameters
        ----------
        split : int
            Target split dimension. Does not need to be equal to ``arr.split``
        tile_dims : torch.Tensor
            Tensor containing the sizes of the each tile
        arr : DNDarray
            Array for which the tiles are being created for
        """
        # this is split off specifically for the resplit function
        tile_locations = torch.zeros(
            [tile_dims[x].numel() for x in range(arr.ndim)],
            dtype=torch.int64,
            device=arr.device.torch_device,
        )
        if split is None:
            tile_locations += arr.comm.rank
            return tile_locations
        arb_slice = [slice(None)] * arr.ndim
        for pr in range(1, arr.comm.size):
            arb_slice[split] = pr
            tile_locations[tuple(arb_slice)] = pr
        return tile_locations

    @property
    def arr(self) -> DNDarray:
        """
        Get the DNDarray associated with the tiling object
        """
        return self.__DNDarray

    @property
    def lshape_map(self) -> torch.Tensor:
        """
        Return the shape of all of the local torch.Tensors
        """
        return self.__lshape_map

    @property
    def tile_locations(self) -> torch.Tensor:
        """
        Get the ``torch.Tensor`` with the locations of the tiles for SplitTiles

        Examples
        --------
        see :class:`~SplitTiles`
        """
        return self.__tile_locations

    @property
    def tile_ends_g(self) -> torch.Tensor:
        """
        Returns a ``torch.Tensor`` with the global indices with the end points of the tiles in every dimension

        Examples
        --------
        see :func:`SplitTiles`
        """
        return self.__tile_ends_g

    @property
    def tile_dimensions(self) -> torch.Tensor:
        """
        Returns a ``torch.Tensor`` with the sizes of the tiles
        """
        return self.__tile_dims

    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice], ...]]) -> torch.Tensor:
        """
        Getitem function for getting tiles. Returns the tile which is specified is returned, but only on the process which it resides

        Parameters
        ----------
        key : int or Tuple or Slice
            Key which identifies the tile/s to get

        Examples
        --------
        >>> test = torch.arange(np.prod([i + 6 for i in range(2)])).reshape([i + 6 for i in range(2)])
        >>> a = ht.array(test, split=0).larray
        [0/2] tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
        [0/2]         [ 7.,  8.,  9., 10., 11., 12., 13.]])
        [1/2] tensor([[14., 15., 16., 17., 18., 19., 20.],
        [1/2]         [21., 22., 23., 24., 25., 26., 27.]])
        [2/2] tensor([[28., 29., 30., 31., 32., 33., 34.],
        [2/2]         [35., 36., 37., 38., 39., 40., 41.]])
        >>> a.create_split_tiles()
        >>> a.tiles[:2, 2]
        [0/2] tensor([[ 5.,  6.],
        [0/2]         [12., 13.]])
        [1/2] tensor([[19., 20.],
        [1/2]         [26., 27.]])
        [2/2] None
        >>> a = ht.array(test, split=1)
        >>> a.create_split_tiles()
        >>> a.tiles[1]
        [0/2] tensor([[14., 15., 16.],
        [0/2]         [21., 22., 23.]])
        [1/2] tensor([[17., 18.],
        [1/2]         [24., 25.]])
        [2/2] tensor([[19., 20.],
        [2/2]         [26., 27.]])
        """
        # todo: strides can be implemented with using a list of slices for each dimension
        if not isinstance(key, (tuple, slice, int, torch.tensor)):
            raise TypeError("key type not supported: {}".format(type(key)))
        arr = self.__DNDarray
        # if arr.comm.rank not in self.tile_locations[key]:
        #     return None
        # This filters out the processes which are not involved
        # next need to get the local indices
        # tile_ends_g has the end points, need to get the start and stop
        if arr.comm.rank not in self.tile_locations[key]:
            return None
        arb_slices = self.__get_tile_slices(key)
        return arr.larray[tuple(arb_slices)]

    def __get_tile_slices(
        self, key: Union[int, slice, Tuple[Union[int, slice], ...]]
    ) -> Tuple[slice, ...]:
        """
        Create and return slices to convert a key from the tile indices to the normal indices
        """
        arr = self.__DNDarray
        arb_slices = [None] * arr.ndim
        end_rank = (
            max(self.tile_locations[key].unique())
            if self.tile_locations[key].unique().numel() > 1
            else self.tile_locations[key]
        )

        if isinstance(key, int):
            key = [key]
        if len(key) < arr.ndim or key[-1] is None:
            lkey = list(key)
            lkey.extend([slice(0, None)] * (arr.ndim - len(key)))
            key = lkey
        for d in range(arr.ndim):
            # todo: implement advanced indexing (lists of positions to iterate through)
            lkey = key
            stop = self.tile_ends_g[d][lkey[d]].max().item()
            # print(stop, self.lshape_map[end_rank][d].max())
            stop = (
                stop
                if d != arr.split or stop is None
                else self.lshape_map[end_rank][d].max().item()
            )
            if (
                isinstance(lkey[d], slice)
                and d != arr.split
                and lkey[d].start != 0
                and lkey[d].start is not None
            ):
                # if the key is a slice in a dimension, and the start value of the slice is not 0,
                # and d is not the split dimension (-> the tiles start at 0 on all tiles in the split dim)
                start = self.tile_ends_g[d][lkey[d].start - 1].item()
            elif isinstance(lkey[d], int) and lkey[d] > 0 and d != arr.split:
                start = self.tile_ends_g[d][lkey[d] - 1].item()
            elif (
                isinstance(lkey[d], torch.Tensor)
                and lkey[d].numel() == 1
                and lkey[d] > 0
                and d != arr.split
            ):
                start = self.tile_ends_g[d][lkey[d] - 1].item()
            else:
                start = 0
            arb_slices[d] = slice(start, stop)
        return arb_slices

    def get_tile_size(
        self, key: Union[int, slice, Tuple[Union[int, slice], ...]]
    ) -> Tuple[int, ...]:
        """
        Get the size of a tile or tiles indicated by the given key

        Parameters
        ----------
        key : int or slice or tuple
            which tiles to get
        """
        arb_slices = self.__get_tile_slices(key)
        inds = []
        for sl in arb_slices:
            inds.append(sl.stop - sl.start)
        return tuple(inds)

    def __setitem__(
        self,
        key: Union[int, slice, Tuple[Union[int, slice], ...]],
        value: Union[int, float, torch.Tensor],
    ) -> None:
        """
        Set the values of a tile

        Parameters
        ----------
        key : int or Tuple or Slice
            Key which identifies the tile/s to get
        value : int or torch.Tensor
            Value to be set on the tile

        Examples
        --------
        see getitem function for this class
        """
        if not isinstance(key, (tuple, slice, int, torch.Tensor)):
            raise TypeError("key type not supported: {}".format(type(key)))
        if not isinstance(value, (torch.Tensor, int, float)):
            raise TypeError("value type not supported: {}".format(type(value)))
        # todo: is it okay for cross-split setting? this can be problematic,
        #   but it is fine if the data shapes match up
        if self.__DNDarray.comm.rank not in self.tile_locations[key]:
            return None
        # this will set the tile values using the torch setitem function
        arr = self.__getitem__(key)
        arr.__setitem__(slice(0, None), value)


class SquareDiagTiles:
    """
    These tiles are square along the diagonal of the given DNDarray. If the matrix is square-like
    then the diagonal will be stretched to the last process. For the definition of square-like
    see :func: `dndarray.DNDarray.matrix_shape_classifier()`. If the matrix is TS or SF with split == 1
    and split == 0 respectively then the diagonal reaches the end already and tiles are added to the
    processes after the diagonal ends. Otherwise the last tile corresponding to the end of the diagonal
    will be adjusted accordingly.

    WARNING: The generation of these tiles may unbalance the original tensor!
    Note: This tiling scheme is intended for use with the QR and SVD functions.


    Parameters
    ----------
    arr : DNDarray
        The array to be tiled
    tiles_per_proc : int, optional
        Default = 2
        the number of divisions per process,
    no_tiles : bool, optional
        Default = False
        This will initilize the class but will not do any logic,
        to be used when the tiles will be matched to another tiling class

    Attributes
    -----------
    arr : DNDarray
        the DNDarray which the tiles operate one
    col_indices : list
        global indices of the beginning of each column
    lshape_map : torch.Tensor
        map of all the lshapes of the DNDarray
    last_diagonal_process : int
        the last process with diagonal elements
    row_indices : list
        global indices of the beginning of each row
    tile_columns : int
        total number of tile columns
    tile_columns_per_process : list
        number of tile columns on each process
    tile_map : torch.Tensor
        the full map of the start and stop indices for all tiles
    tile_rows : int
        total number of tile rows
    tile_rows_per_process : list
        number of tile rows on each process
    """

    def __init__(self, arr, tiles_per_proc=2, no_tiles=False):
        # lshape_map -> rank (int), lshape (tuple of the local lshape, self.lshape)
        if not isinstance(arr, DNDarray):
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
            arr.redistribute_(lshape_map=lshape_map, target_map=target_map)

        row_per_proc_list = [tiles_per_proc] * arr.comm.size

        last_diag_pr, col_per_proc_list, col_inds, tile_columns = self.__create_cols(
            arr, lshape_map, tiles_per_proc
        )
        # need to adjust the lshape if the splits overlap
        if arr.split == 0 and tiles_per_proc == 1:
            # if the split is 0 and the number of tiles per proc is 1
            # then the local data needs to be redistributed to fit the full diagonal on as many
            #       processes as possible
            # if any(lshape_map[..., arr.split] == 1):
            (
                last_diag_pr,
                col_per_proc_list,
                col_inds,
                tile_columns,
            ) = self.__adjust_lshape_sp0_1tile(arr, col_inds, lshape_map, tiles_per_proc)
            # re-test for empty processes and remove empty rows
            empties = torch.where(lshape_map[..., 0] == 0)[0]
            if empties.numel() > 0:
                # need to remove the entry in the rows per process
                for e in empties:
                    row_per_proc_list[e] = 0

        row_inds = []
        for c in col_inds:
            # set the row indices to be the same for all of the column indices
            #   (however many there are)
            row_inds.append(c)

        if arr.split == 0 and arr.gshape[0] < arr.gshape[1]:
            # need to adjust the very last tile to be the remaining
            col_inds[-1] = arr.gshape[1] - sum(col_inds[:-1])

        # if there is too little data on the last tile then combine them
        if arr.split == 0 and last_diag_pr < arr.comm.size - 1:
            # these conditions imply that arr.gshape[0] > arr.gshape[1] (assuming balanced)
            self.__adjust_last_row_sp0_m_ge_n(
                arr, lshape_map, last_diag_pr, row_inds, row_per_proc_list, tile_columns
            )

        if arr.split == 0 and arr.gshape[0] > arr.gshape[1]:
            # adjust the last row to have the
            self.__def_end_row_inds_sp0_m_ge_n(
                arr, row_inds, last_diag_pr, tiles_per_proc, lshape_map
            )

        if arr.split == 1 and arr.gshape[0] < arr.gshape[1]:
            self.__adjust_cols_sp1_m_ls_n(
                arr, col_per_proc_list, last_diag_pr, col_inds, lshape_map
            )

        if arr.split == 1 and arr.gshape[0] > arr.gshape[1]:
            # add extra rows if there is place below the diagonal for split == 1
            # adjust the very last tile to be the remaining
            self.__last_tile_row_adjust_sp1(arr, row_inds)

        # need to remove blank rows for arr.gshape[0] < arr.gshape[1]
        if arr.gshape[0] < arr.gshape[1]:
            row_inds_hold = []
            for i in torch.nonzero(
                input=torch.tensor(row_inds, device=arr.larray.device), as_tuple=False
            ).flatten():
                row_inds_hold.append(row_inds[i.item()])
            row_inds = row_inds_hold

        tile_map = torch.zeros(
            [len(row_inds), len(col_inds), 3], dtype=torch.int, device=arr.larray.device
        )
        # if arr.split == 0:  # adjust the 1st dim to be the cumsum
        col_inds = [0] + col_inds[:-1]
        col_inds = torch.tensor(col_inds, device=arr.larray.device).cumsum(dim=0)
        # if arr.split == 1:  # adjust the 0th dim to be the cumsum
        row_inds = [0] + row_inds[:-1]
        row_inds = torch.tensor(row_inds, device=arr.larray.device).cumsum(dim=0)

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
        return tile_map

    @property
    def arr(self) -> DNDarray:
        """
        Returns
        -------
        DNDarray : the DNDarray for which the tiles are defined on
        """
        return self.__DNDarray

    @property
    def col_indices(self) -> List[int, ...]:
        """
        Returns
        -------
        list : list containing the indices of the tile columns
        """
        return self.__col_inds

    @property
    def lshape_map(self) -> torch.Tensor:
        """
        Returns the map of the lshape tuples for the ``DNDarray`` given.
        Units are ``(rank, lshape)`` (tuple of the local shape)
        """
        return self.__lshape_map

    @property
    def last_diagonal_process(self) -> int:
        """
        Returns
        -------
        int : the rank of the last process with diagonal elements
        """
        return self.__last_diag_pr

    @property
    def row_indices(self) -> List[int, ...]:
        """
        Returns
        -------
        list : list containing the indices of the tile rows
        """
        return self.__row_inds

    @property
    def tile_columns(self) -> int:
        """
        Returns
        -------
        int : number of tile columns
        """
        return len(self.__col_inds)

    @property
    def tile_columns_per_process(self) -> List[int, ...]:
        """
        Returns
        -------
        list : list containing the number of columns on all processes
        """
        return self.__col_per_proc_list

    @property
    def tile_map(self) -> torch.Tensor:
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
    def tile_rows(self) -> int:
        """
        Returns
        -------
        int : number of tile rows
        """
        return len(self.__row_inds)

    @property
    def tile_rows_per_process(self) -> List[int, ...]:
        """
        Returns
        -------
        list : list containing the number of rows on all processes
        """
        return self.__row_per_proc_list

    def get_start_stop(
        self, key: Union[int, slice, Tuple[int, slice, ...]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the start and stop indices which correspond to the tile/s which corresponds to the
        given key. The key MUST use global indices.

        Parameters
        ----------
        key : int or Tuple or List or slice
            Indices to select the tile
            STRIDES ARE NOT ALLOWED, MUST BE GLOBAL INDICES

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

    def __getitem__(self, key: Union[int, slice, Tuple[int, slice, ...]]) -> torch.Tensor:
        """
        Standard getitem function for the tiles. The returned item is a view of the original
        DNDarray, operations which are done to this view will change the original array.
        **STRIDES ARE NOT AVAILABLE, NOR ARE CROSS-SPLIT SLICES**

        Parameters
        ----------
        key : int, slice, tuple
            indices of the tile/s desired

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
        local_arr = arr.larray
        if not isinstance(key, (int, tuple, slice)):
            raise TypeError(
                "key must be an int, tuple, or slice, is currently {}".format(type(key))
            )
        involved_procs = tile_map[key][..., 2].unique()
        if involved_procs.nelement() == 1 and involved_procs == arr.comm.rank:
            st0, sp0, st1, sp1 = self.get_start_stop(key=key)
            return local_arr[st0:sp0, st1:sp1]
        elif involved_procs.nelement() > 1:
            raise ValueError("Slicing across splits is not allowed")
        else:
            return None

    def local_get(self, key: Union[int, slice, Tuple[int, slice, ...]]) -> torch.Tensor:
        """
        Getitem routing using local indices, converts to global indices then uses getitem

        Parameters
        ----------
        key : int, slice, tuple, list
            Indices of the tile/s desired.
            If the stop index of a slice is larger than the end will be adjusted to the maximum
            allowed

        Examples
        --------
        See local_set function.
        """
        rank = self.__DNDarray.comm.rank
        key = self.local_to_global(key=key, rank=rank)
        return self.__getitem__(key)

    def local_set(
        self, key: Union[int, slice, Tuple[int, slice, ...]], value: Union[int, float, torch.Tensor]
    ):
        """
        Setitem routing to set data to a local tile (using local indices)

        Parameters
        ----------
        key : int or slice or Tuple[int,...]
            Indices of the tile/s desired
            If the stop index of a slice is larger than the end will be adjusted to the maximum
            allowed
        value : torch.Tensor or int or float
            Data to be written to the tile

        Examples
        --------
        >>> a = ht.zeros((11, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> local = a_tiles.local_get(key=slice(None))
        >>> a_tiles.local_set(key=slice(None), value=torch.arange(local.numel()).reshape(local.shape))
        >>> print(a.larray)
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

    def local_to_global(
        self, key: Union[int, slice, Tuple[int, slice, ...]], rank: int
    ) -> Tuple[int, slice, ...]:
        """
        Convert local indices to global indices

        Parameters
        ----------
        key : int or slice or Tuple or List
            Indices of the tile/s desired.
            If the stop index of a slice is larger than the end will be adjusted to the maximum
            allowed
        rank : int
            Process rank

        Examples
        --------
        >>> a = ht.zeros((11, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> rank = a.comm.rank
        >>> print(a_tiles.local_to_global(key=(slice(None), 1), rank=rank))
        [0/1] (slice(0, 2, None), 1)
        [1/1] (slice(2, 4, None), 1)
        >>> print(a_tiles.local_to_global(key=(0, 2), rank=0))
        [0/1] (0, 2)
        [1/1] (0, 2)
        >>> print(a_tiles.local_to_global(key=(0, 2), rank=1))
        [0/1] (2, 2)
        [1/1] (2, 2)
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

    def match_tiles(self, tiles_to_match: SquareDiagTiles) -> None:
        """
        Match the tiles of the Q matrix associated with R for QR factorization.
        tiles_to_match must be the tiles for the R matrix. Q must be split == 0.
        NOTE: this will redistribute Q. After redistribution, the split axis will equal that of R
            up to the point that there are elements of Q to distribute

        Parameters
        ----------
        tiles_to_match : SquareDiagTiles
            The tiles which should be matched by the current tiling scheme

        Notes
        -----
        This function overwrites most, if not all, of the elements of this class. Intended for use with the Q matrix,
        to match the tiling of a/R. For this to work properly it is required that the 0th dim of both matrices is equal
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
            self.__tile_map = torch.zeros(
                (self.tile_rows, self.tile_columns, 3),
                dtype=torch.int,
                device=match_dnd.larray.device,
            )
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

    def change_row_and_column_index(self, row: int, column: int, position_from_end: int = 0):
        """
        Add a row/column to the tiles with a relative position from the end of BOTH dims

        Parameters
        ----------
        row : int
            row index to add
        column: int
            column index to add
        position_from_end: int, optional
            where to replace the index in the list of row/col indexes

        Returns
        -------
        None
        """
        if position_from_end == 0:
            self.__row_inds.append(row)
            self.__col_inds.append(column)
        else:
            self.__row_inds[position_from_end] = row
            self.__col_inds[position_from_end] = column
        self.__tile_map = self.__create_tile_map(
            self.__row_inds,
            self.__col_inds,
            self.__row_per_proc_list,
            self.__col_per_proc_list,
            self.arr,
        )

    def set_arr(self, arr: DNDarray):
        """
        Set the DNDarray of self to be arr

        Parameters
        ----------
        arr : DNDarray
            array to be set
        """
        if not isinstance(arr, DNDarray):
            raise TypeError("arr must be a DNDarray, currently is {}".format(type(arr)))
        self.__DNDarray = arr

    def __setitem__(
        self, key: Union[int, slice, Tuple[int, slice, ...]], value: Union[int, float, torch.Tensor]
    ) -> None:
        """
        Item setter,
        uses the torch item setter and the getitem routines to set the values of the original array
        (arr in __init__)

        Parameters
        ----------
        key : int or slice or Tuple[int,...]
            Tile indices to identify the target tiles
        value : int or torch.Tensor
            Values to be set

        Example
        -------
        >>> a = ht.zeros((12, 10), split=0)
        >>> a_tiles = tiling.SquareDiagTiles(a, tiles_per_proc=2)  # type: tiling.SquareDiagTiles
        >>> a_tiles[0:2, 2] = 11
        >>> a_tiles[0, 0] = 22
        >>> a_tiles[2] = 33
        >>> a_tiles[3, 3] = 44
        >>> print(a.larray)
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
