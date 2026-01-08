Module heat.core.tiling
=======================
Tiling functions/classes. With these classes, you can classes you can address blocks of data in a DNDarray

Classes
-------

`SplitTiles(arr: DNDarray)`
:   Initialize tiles with the tile divisions equal to the theoretical split dimensions in
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
    >>> a = ht.zeros(
    ...     (
    ...         10,
    ...         11,
    ...     ),
    ...     split=None,
    ... )
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

    ### Static methods

    `set_tile_locations(split: int, tile_dims: torch.Tensor, arr: DNDarray) ‑> torch.Tensor`
    :   Create a `torch.Tensor` which contains the locations of the tiles of ``arr`` for the given split

        Parameters
        ----------
        split : int
            Target split dimension. Does not need to be equal to ``arr.split``
        tile_dims : torch.Tensor
            Tensor containing the sizes of the each tile
        arr : DNDarray
            Array for which the tiles are being created for

    ### Instance variables

    `arr: DNDarray`
    :   Get the DNDarray associated with the tiling object

    `lshape_map: torch.Tensor`
    :   Return the shape of all of the local torch.Tensors

    `tile_dimensions: torch.Tensor`
    :   Returns a ``torch.Tensor`` with the sizes of the tiles

    `tile_ends_g: torch.Tensor`
    :   Returns a ``torch.Tensor`` with the global indices with the end points of the tiles in every dimension

        Examples
        --------
        see :func:`SplitTiles`

    `tile_locations: torch.Tensor`
    :   Get the ``torch.Tensor`` with the locations of the tiles for SplitTiles

        Examples
        --------
        see :class:`~SplitTiles`

    ### Methods

    `get_subarray_params(self, from_axis: int, to_axis: int) ‑> List[Tuple[List[int], List[int], List[int]]]`
    :   Create subarray types of the local array along a new split axis. For use with Alltoallw.

        Based on the work by Dalcin et al. (https://arxiv.org/abs/1804.09536)
        Return type is a list of tuples, each tuple containing the shape of the local array, the shape of the subarray, and the start index of the subarray.

        Parameters
        ----------
        from_axis : int
            Current split axis of global array.
        to_axis : int
            New split axis of of subarrays array.

    `get_tile_size(self, key: Union[int, slice, Tuple[Union[int, slice], ...]]) ‑> Tuple[int, ...]`
    :   Get the size of a tile or tiles indicated by the given key

        Parameters
        ----------
        key : int or slice or tuple
            which tiles to get

`SquareDiagTiles(arr: DNDarray, tiles_per_proc: int = 2)`
:   Generate the tile map and the other objects which may be useful.
    The tiles generated here are based of square tiles along the diagonal. The size of these
    tiles along the diagonal dictate the divisions across all processes. If
    ``gshape[0]>>gshape[1]`` then there will be extra tiles generated below the diagonal.
    If ``gshape[0]`` is close to ``gshape[1]``, then the last tile (as well as the other tiles which
    correspond with said tile) will be extended to cover the whole array. However, extra tiles
    are not generated above the diagonal in the case that ``gshape[0]<<gshape[1]``.

    Parameters
    ----------
    arr : DNDarray
        The array to be tiled
    tiles_per_proc : int, optional
        The number of divisions per process
        Default: 2

    Attributes
    ----------
    __col_per_proc_list : List
        List is length of the number of processes, each element has the number of tile
        columns on the process whos rank equals the index
    __DNDarray: DNDarray
        The whole DNDarray
    __lshape_map : torch.Tensor
        ``unit -> [rank, row size, column size]``
        Tensor filled with the shapes of the local tensors
    __tile_map : torch.Tensor
        ``units -> row, column, start index in each direction, process``
        Tensor filled with the global indices of the generated tiles
    __row_per_proc_list : List
        List is length of the number of processes, each element has the number of tile
        rows on the process whos rank equals the index

    Warnings
    -----------
    The generation of these tiles may unbalance the original ``DNDarray``!

    Notes
    -----
    This tiling scheme is intended for use with the :func:`~heat.core.linalg.qr.qr` function.

    ### Instance variables

    `arr: DNDarray`
    :   Returns the ``DNDarray`` for which the tiles are defined on

    `col_indices: List[int, ...]`
    :   Returns a list containing the indices of the tile columns

    `last_diagonal_process: int`
    :   Returns the rank of the last process with diagonal elements

    `lshape_map: torch.Tensor`
    :   Returns the map of the lshape tuples for the ``DNDarray`` given.
        Units are ``(rank, lshape)`` (tuple of the local shape)

    `row_indices: List[int, ...]`
    :   Returns a list containing the indices of the tile rows

    `tile_columns: int`
    :   Returns the number of tile columns

    `tile_columns_per_process: List[int, ...]`
    :   Returns a list containing the number of columns on all processes

    `tile_map: torch.Tensor`
    :   Returns tile_map which contains the sizes of the tiles
        units are ``(row, column, start index in each direction, process)``

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

    `tile_rows: int`
    :   Returns the number of tile rows

    `tile_rows_per_process: List[int, ...]`
    :   Returns a list containing the number of rows on all processes

    ### Methods

    `get_start_stop(self, key: Union[int, slice, Tuple[int, slice, ...]]) ‑> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
    :   Returns the start and stop indices in form of ``(dim0 start, dim0 stop, dim1 start, dim1 stop)``
        which correspond to the tile/s which corresponds to the given key. The key MUST use global indices.

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

    `local_get(self, key: Union[int, slice, Tuple[int, slice, ...]]) ‑> torch.Tensor`
    :   Returns the local tile/s corresponding to the key given
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

    `local_set(self, key: Union[int, slice, Tuple[int, slice, ...]], value: Union[int, float, torch.Tensor])`
    :   Setitem routing to set data to a local tile (using local indices)

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
        >>> a_tiles.local_set(
        ...     key=slice(None), value=torch.arange(local.numel()).reshape(local.shape)
        ... )
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

    `local_to_global(self, key: Union[int, slice, Tuple[int, slice, ...]], rank: int) ‑> Tuple[int, slice, ...]`
    :   Convert local indices to global indices

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

    `match_tiles(self, tiles_to_match: SquareDiagTiles) ‑> None`
    :   Function to match the tile sizes of another tile map

        Parameters
        ----------
        tiles_to_match : SquareDiagTiles
            The tiles which should be matched by the current tiling scheme

        Notes
        -----
        This function overwrites most, if not all, of the elements of this class. Intended for use with the Q matrix,
        to match the tiling of a/R. For this to work properly it is required that the 0th dim of both matrices is equal
