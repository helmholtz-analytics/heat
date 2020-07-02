import torch

from . import dndarray
from . import factories

__all__ = ["SplitTiles", "SquareDiagTiles"]


class SplitTiles:
    def __init__(self, arr):
        """
        Initialize tiles with the tile divisions equal to the theoretical split dimensions in
        every dimension

        Parameters
        ----------
        arr : dndarray.DNDarray
            base array for which to create the tiles

        Examples
        --------
        (3 processes)
        >>> a = ht.zeros((10, 11,), split=None)
        >>> a.create_split_tiles()
        >>> print(a.tiles.tile_ends_g)
        [0] tensor([[ 4,  7, 10],
        [0]         [ 4,  8, 11]], dtype=torch.int32)
        [1] tensor([[ 4,  7, 10],
        [1]         [ 4,  8, 11]], dtype=torch.int32)
        [2] tensor([[ 4,  7, 10],
        [2]         [ 4,  8, 11]], dtype=torch.int32)
        >>> print(a.tiles.tile_locations)
        [0] tensor([[0, 0, 0],
        [0]         [0, 0, 0],
        [0]         [0, 0, 0]], dtype=torch.int32)
        [1] tensor([[1, 1, 1],
        [1]         [1, 1, 1],
        [1]         [1, 1, 1]], dtype=torch.int32)
        [2] tensor([[2, 2, 2],
        [2]         [2, 2, 2],
        [2]         [2, 2, 2]], dtype=torch.int32)
        >>> a = ht.zeros((10, 11), split=1)
        >>> a.create_split_tiles()
        >>> print(a.tiles.tile_ends_g)
        [0] tensor([[ 4,  7, 10],
        [0]         [ 4,  8, 11]], dtype=torch.int32)
        [1] tensor([[ 4,  7, 10],
        [1]         [ 4,  8, 11]], dtype=torch.int32)
        [2] tensor([[ 4,  7, 10],
        [2]         [ 4,  8, 11]], dtype=torch.int32)
        >>> print(a.tiles.tile_locations)
        [0] tensor([[0, 1, 2],
        [0]         [0, 1, 2],
        [0]         [0, 1, 2]], dtype=torch.int32)
        [1] tensor([[0, 1, 2],
        [1]         [0, 1, 2],
        [1]         [0, 1, 2]], dtype=torch.int32)
        [2] tensor([[0, 1, 2],
        [2]         [0, 1, 2],
        [2]         [0, 1, 2]], dtype=torch.int32)
        """
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
    def set_tile_locations(split, tile_dims, arr):
        """
        Create a torch Tensor with the locations of the tiles for SplitTiles

        Parameters
        ----------
        split : int
            target split dimension. does not need to be equal to arr.split
        tile_dims : torch.Tensor
            torch Tensor containing the sizes of the each tile
        arr : DNDarray
            array for which the tiles are being created for

        Returns
        -------
        tile_locations : torch.Tensor
            a tensor which contains the locations of the tiles of arr for the given split
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
    def arr(self):
        return self.__DNDarray

    @property
    def lshape_map(self):
        return self.__lshape_map

    @property
    def tile_locations(self):
        """
        Get the torch Tensor with the locations of the tiles for SplitTiles

        Examples
        --------
        see :func:`~SplitTiles.__init__`
        """
        return self.__tile_locations

    @property
    def tile_ends_g(self):
        """
        Returns
        -------
        end_of_tiles_global : torch.Tensor
            tensor wih the global indces with the end points of the tiles in every dimension

        Examples
        --------
        see :func:`~SplitTiles.__init__`
        """
        return self.__tile_ends_g

    @property
    def tile_dimensions(self):
        return self.__tile_dims

    def __getitem__(self, key):
        """
        Getitem function for getting tiles

        Parameters
        ----------
        key : int, tuple, slice
            key which identifies the tile/s to get

        Returns
        -------
        tile/s : torch.Tensor
             the tile which is specified is returned, but only on the process which it resides

        Examples
        --------
        >>> test = torch.arange(np.prod([i + 6 for i in range(2)])).reshape([i + 6 for i in range(2)])
        >>> a = ht.array(test, split=0)
        [0] tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
        [0]         [ 7.,  8.,  9., 10., 11., 12., 13.]])
        [1] tensor([[14., 15., 16., 17., 18., 19., 20.],
        [1]         [21., 22., 23., 24., 25., 26., 27.]])
        [2] tensor([[28., 29., 30., 31., 32., 33., 34.],
        [2]         [35., 36., 37., 38., 39., 40., 41.]])
        >>> a.create_split_tiles()
        >>> a.tiles[:2, 2]
        [0] tensor([[ 5.,  6.],
        [0]         [12., 13.]])
        [1] tensor([[19., 20.],
        [1]         [26., 27.]])
        [2] None
        >>> a = ht.array(test, split=1)
        >>> a.create_split_tiles()
        >>> a.tiles[1]
        [0] tensor([[14., 15., 16.],
        [0]         [21., 22., 23.]])
        [1] tensor([[17., 18.],
        [1]         [24., 25.]])
        [2] tensor([[19., 20.],
        [2]         [26., 27.]])
        """
        # todo: strides can be implemented with using a list of slices for each dimension
        if not isinstance(key, (tuple, slice, int, torch.Tensor)):
            raise TypeError("key type not supported: {}".format(type(key)))
        arr = self.__DNDarray
        # if arr.comm.rank not in self.tile_locations[key]:
        #     return None
        # This filters out the processes which are not involved
        # next need to get the local indices
        # tile_ends_g has the end points, need to get the start and stop
        if arr.comm.rank not in self.tile_locations[key]:
            return None
        arb_slices = self.get_tile_slices(key)
        return arr._DNDarray__array[tuple(arb_slices)]

    def get_tile_slices(self, key):
        arr = self.__DNDarray
        arb_slices = [None] * arr.ndim
        # print(self.tile_locations[key])
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

    def get_tile_size(self, key):
        arb_slices = self.get_tile_slices(key)
        inds = []
        for sl in arb_slices:
            inds.append(sl.stop - sl.start)
        return tuple(inds)

    def __setitem__(self, key, value):
        """
        Set the values of a tile

        Parameters
        ----------
        key : int, tuple, slice
            key which identifies the tile/s to get
        value : int, torch.Tensor
            Value to be set on the tile

        Returns
        -------
        None

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
        the array to be tiled
    tiles_per_proc : int, optional
        Default = 2
        the number of divisions per process,
    no_tiles : bool, optional
        Default = False
        This will initilize the class but will not do any logic,
        to be used when the tiles will be matched to another tiling class

    Properties
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
        if not isinstance(arr, dndarray.DNDarray):
            raise TypeError("arr must be a DNDarray, is currently a {}".format(type(self)))
        if not isinstance(tiles_per_proc, int):
            raise TypeError("tiles_per_proc must be an int, is currently a {}".format(type(self)))
        if tiles_per_proc < 1:
            raise ValueError("Tiles per process must be >= 1, currently: {}".format(tiles_per_proc))
        if len(arr.shape) != 2:
            raise ValueError("Arr must be 2 dimensional, current shape {}".format(arr.shape))
        if arr.split is None or arr.split > 1:
            raise ValueError("DNDarray must be distributed")

        lshape_map = arr.create_lshape_map()
        if not no_tiles:
            mat_shape_type = arr.matrix_shape_classifier()
            torch_dev = arr.device.torch_device
            divs_per_proc = [tiles_per_proc] * arr.comm.size
            if (mat_shape_type == "TS" and arr.split == 0) or (
                mat_shape_type == "SF" and arr.split == 1
            ):
                # for these cases the diagonal crosses all splits
                # the min(gshape) -> dim1 for TS, dim0 for SF
                mgshape = arr.gshape[1] if mat_shape_type == "TS" else arr.gshape[0]

                lshape_cs_sp = lshape_map[..., arr.split].cumsum(0)
                st_ldp = torch.where(lshape_cs_sp > mgshape)[0][0].item()
                last_diag_pr = st_ldp
                ntiles = (st_ldp + 1) * tiles_per_proc
                tile_shape, rem = mgshape // ntiles, mgshape % ntiles
                if tile_shape <= 1:  # this can be raised as these matrices are mostly square
                    raise ValueError(
                        "Dataset too small for tiles to be useful, resplit to None if possible"
                    )

                col_inds = [0] + torch.tensor(
                    [tile_shape] * ntiles, dtype=torch.int, device=torch_dev
                ).cumsum(0).tolist()
                if rem > 0:
                    col_inds[-1] += rem
                row_per_proc_list = [0] * arr.comm.size
                row_inds = col_inds.copy()
                col_inds = col_inds[:-1]
                col_per_proc_list = [len(col_inds)] * arr.comm.size

                redist_vec = lshape_map[..., arr.split].clone()
                if st_ldp > 0:
                    redist_vec[0] = tile_shape * tiles_per_proc
                    for i in range(st_ldp):
                        # this loop makes it so that the remainder of all processes is pushed to the last one
                        diff = redist_vec[i] - lshape_map[..., arr.split][i]
                        redist_vec[i] = tile_shape * tiles_per_proc
                        redist_vec[i + 1] += abs(diff)
                    diff = mgshape - redist_vec[: st_ldp + 1].sum()
                    if diff > 0:
                        redist_vec[st_ldp] += diff
                        redist_vec[-1] -= diff
                else:
                    diff = mgshape - redist_vec[0]
                    redist_vec[0] = mgshape
                    redist_vec[1] += abs(diff)

                if redist_vec[st_ldp] > tile_shape:
                    # include the process with however many shape will fit
                    diag_rem_el = mgshape - sum(redist_vec[:st_ldp])
                    # this is the number of diagonal elements on the process
                    proc_space_after_diag = redist_vec[st_ldp] - diag_rem_el
                    # this is the number of elements on the process after the diagonal
                    if proc_space_after_diag < 0:
                        print(redist_vec, diag_rem_el, arr.gshape, proc_space_after_diag)
                        raise ValueError(
                            "wtf? proc space after diag: {}".format(proc_space_after_diag)
                        )

                    if redist_vec[st_ldp] < diag_rem_el:
                        # not enough space for any diagonal tiles, send diag rem el to process before
                        redist_vec[st_ldp - 1] += diag_rem_el
                        last_diag_pr = st_ldp - 1
                    elif redist_vec[st_ldp] < tile_shape * tiles_per_proc:
                        # not enough space for all requested tiles
                        lcl_tiles = redist_vec[st_ldp] // tile_shape
                        lcl_rem = redist_vec[st_ldp] % tile_shape
                        if lcl_rem == 1:
                            redist_vec[st_ldp] -= 1
                            redist_vec[-1] += 1
                            divs_per_proc[st_ldp] = lcl_tiles
                        else:
                            # can fit one more tile on the process
                            divs_per_proc[st_ldp] = lcl_tiles + 1
                            row_inds.append(mgshape + proc_space_after_diag)
                    elif proc_space_after_diag == 1:
                        # 1 element after diagonal
                        redist_vec[st_ldp] -= 1
                        redist_vec[-1] += 1
                    elif proc_space_after_diag >= tile_shape:
                        # more space after the diagonal (more than 1 element) -> proc space > 1
                        divs_per_proc[st_ldp] += 1
                        row_inds.append(mgshape + proc_space_after_diag.item())
                    else:
                        redist_vec[st_ldp] -= proc_space_after_diag
                        redist_vec[-1] += proc_space_after_diag
                if redist_vec.sum() < arr.gshape[arr.split]:
                    redist_vec[-1] += arr.gshape[arr.split] - redist_vec.sum()

                target_map = lshape_map.clone()
                target_map[..., arr.split] = redist_vec
                arr.redistribute_(lshape_map, target_map)
                # next, chunk each process

                for pr in range(arr.comm.size):
                    if pr <= last_diag_pr:
                        row_per_proc_list[pr] = divs_per_proc[pr]
                    else:
                        # only 1 tile on the processes after the diagonal
                        row_inds.append(row_inds[-1] + redist_vec[pr].item())
                        row_per_proc_list[pr] = 1
                row_inds = row_inds[:-1]
                if mat_shape_type == "SF":
                    hld = row_inds
                    row_inds = col_inds
                    col_inds = hld
                    hld = col_per_proc_list
                    col_per_proc_list = row_per_proc_list
                    row_per_proc_list = hld
            else:
                # this covers the following cases:
                #       Square -> both splits
                #       TS -> split == 1  # diag covers whole size
                #       SF -> split == 0  # diag covers whole size
                divs_per_proc = [tiles_per_proc] * arr.comm.size
                mgshape = min(arr.gshape)
                ntiles = arr.comm.size * tiles_per_proc
                tile_shape, rem = mgshape // ntiles, mgshape % ntiles
                if tile_shape <= 1:  # this can be raised as these matrices are mostly square
                    raise ValueError(
                        "Dataset too small for tiles to be useful, resplit to None if possible"
                    )
                shape_rem = (
                    arr.gshape[0] - arr.gshape[1]
                    if arr.split == 0
                    else arr.gshape[1] - arr.gshape[0]
                )
                divs = [tile_shape] * ntiles
                redist_v_sp, c = [], 0
                for i in range(arr.comm.size):
                    st = c
                    ed = divs_per_proc[i] + c
                    redist_v_sp.append(sum(divs[st:ed]))
                    c = ed
                redist_v_sp = torch.tensor(redist_v_sp, dtype=torch.int, device=torch_dev)
                # original assumption is that there are tiles_per_proc tiles on each process with
                #       an maximum of ((m - n) % sz) * n * tiles_per_proc  additionally on the last process
                #       (for m x n)
                div_inds1 = torch.tensor(
                    divs, dtype=torch.int32, device=arr.device.torch_device
                ).cumsum(dim=0)
                row_inds = [0] + div_inds1[:-1].tolist()
                col_inds = [0] + div_inds1[:-1].tolist()

                if rem > 0:
                    divs[-1] += rem
                    redist_v_sp[-1] += rem
                if shape_rem > 0:
                    # put the remainder on the last process
                    divs.append(shape_rem)
                    redist_v_sp[-1] += shape_rem

                if mat_shape_type == "TS" and arr.split == 1:
                    # need to add a division after the diagonal here
                    row_inds.append(arr.gshape[1])
                    ntiles += 1
                col_per_proc_list = [ntiles] * arr.comm.size if arr.split == 0 else divs_per_proc
                row_per_proc_list = [ntiles] * arr.comm.size if arr.split == 1 else divs_per_proc

                target_map = lshape_map.clone()
                target_map[..., arr.split] = redist_v_sp
                arr.redistribute_(lshape_map, target_map)

                divs = torch.tensor(divs, dtype=torch.int32, device=torch_dev)
                div_inds = divs.cumsum(dim=0)
                fr_tile_ge = torch.where(div_inds >= mgshape)[0][0]
                if mgshape == arr.gshape[0]:
                    last_diag_pr = arr.comm.size - 1
                elif mgshape == div_inds[fr_tile_ge]:
                    last_diag_pr = torch.floor_divide(fr_tile_ge, tiles_per_proc).item()
                elif fr_tile_ge == 0:  # -> first tile above is 0, thus the last diag pr is also 0
                    last_diag_pr = 0
                else:
                    last_diag_pr = torch.where(
                        fr_tile_ge
                        < torch.tensor(divs_per_proc, dtype=torch.int, device=torch_dev).cumsum(
                            dim=0
                        )
                    )[0][0].item()

            tile_map = self.__create_tile_map(
                row_inds, col_inds, row_per_proc_list, col_per_proc_list, arr
            )

            if arr.split == 1:
                st = 0
                for pr, cols in enumerate(col_per_proc_list):
                    tile_map[:, st : st + cols, 2] = pr
                    st += cols

            self.__DNDarray = arr
            self.__col_per_proc_list = col_per_proc_list
            self.__last_diag_pr = last_diag_pr
            self.__lshape_map = lshape_map
            self.__row_per_proc_list = row_per_proc_list
            self.__tile_map = tile_map
            self.__row_inds = row_inds
            self.__col_inds = col_inds
        else:
            self.__DNDarray = arr
            self.__col_per_proc_list = None
            self.__last_diag_pr = None
            self.__lshape_map = lshape_map
            self.__row_per_proc_list = None
            self.__tile_map = None
            self.__row_inds = None
            self.__col_inds = None

    @staticmethod
    def __create_tile_map(rows, cols, rows_per, cols_per, arr):
        # create the tile map from the given rows, columns, rows/process, cols/process,
        #   and the base array
        tile_map = torch.zeros(
            [len(rows), len(cols), 3], dtype=torch.int, device=arr.device.torch_device
        )

        for num, c in enumerate(cols):  # set columns
            tile_map[:, num, 1] = c
        for num, r in enumerate(rows):  # set rows
            tile_map[num, :, 0] = r

        for i in range(arr.comm.size):
            st = sum(rows_per[:i])
            sp = st + rows_per[i]
            tile_map[..., 2][st:sp] = i
        # to adjust if the last process has more tiles
        i = arr.comm.size - 1
        tile_map[..., 2][sum(rows_per[:i]) :] = i

        if arr.split == 1:
            st = 0
            for pr, cols in enumerate(cols_per):
                tile_map[:, st : st + cols, 2] = pr
                st += cols
        return tile_map

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

        key = list(key)
        if isinstance(key[0], int):
            st0 = row_inds[key[0]] - row_start
            sp0 = row_inds[key[0] + 1] - row_start
        elif isinstance(key[0], slice):
            start = row_inds[key[0].start] if key[0].start is not None else 0
            # print(key, row_inds, self.tile_rows_per_process)
            stop = row_inds[key[0].stop] if key[0].stop is not None else row_inds[-1]
            st0, sp0 = start - row_start, stop - row_start
        else:
            raise TypeError("key[0] must be int or slice, currently {}".format(type(key[0])))
        if isinstance(key[1], int):
            st1 = col_inds[key[1]] - col_start
            sp1 = col_inds[key[1] + 1] - col_start
        elif isinstance(key[1], slice):
            start = col_inds[key[1].start] if key[1].start is not None else 0
            stop = col_inds[key[1].stop] if key[1].stop is not None else col_inds[-1]
            st1, sp1 = start - col_start, stop - col_start
        else:
            raise TypeError("key[1] must be int or slice, currently {}".format(type(key[1])))
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
        self.__getitem__(tuple(key)).__setitem__(slice(None), value)

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
            # print(prev_rows, self.__row_per_proc_list)
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
        # print(key)
        return tuple(key)

    def match_tiles(self, tiles_to_match):
        """
        Match the tiles of the Q matrix associated with R for QR factorization.
        tiles_to_match must be the tiles for the R matrix. Q must be split == 0.
        NOTE: this will redistribute Q. After redistribution, the split axis will equal that of R
            up to the point that there are elements of Q to distribute

        Parameters
        ----------
        tiles_to_match : SquareDiagTiles
            the tiles which should be matched by the current tiling scheme
        """
        if not isinstance(tiles_to_match, SquareDiagTiles):
            raise TypeError(
                "tiles_to_match must be a SquareDiagTiles object, currently: {}".format(
                    type(tiles_to_match)
                )
            )
        if self.arr.split != 0:
            raise ValueError("self.arr.split must be 0, currently: {}".format(self.arr.split))
        self.__match_redist(tiles_to_match)

        match_dnd = tiles_to_match.arr
        mat_shape_type = match_dnd.matrix_shape_classifier()

        if mat_shape_type == "square" and match_dnd.gshape[0] > match_dnd.gshape[1]:
            # can only have split 1 or split 0
            # match lshape, swap cols and
            if match_dnd.split == 1:
                self.__row_per_proc_list = tiles_to_match.__col_per_proc_list.copy()
                self.__col_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
            else:
                self.__row_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
                self.__col_per_proc_list = tiles_to_match.__col_per_proc_list.copy()

            if self.arr.split == 0 and sum(self.__row_per_proc_list) < self.__col_per_proc_list[0]:
                self.__row_per_proc_list[-1] += 1

            self.__row_inds = tiles_to_match.__row_inds.copy()
            self.__col_inds = tiles_to_match.__row_inds.copy()
            # for mostly square matrices the diagonal is stretched to the last process

        elif mat_shape_type == "square":
            print("here")
            self.__row_inds = tiles_to_match.__row_inds.copy()
            self.__col_inds = tiles_to_match.__col_inds.copy()
            if self.arr.split != tiles_to_match.arr.split:
                self.__row_per_proc_list = tiles_to_match.__col_per_proc_list.copy()
                self.__col_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
            else:
                self.__row_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
                self.__col_per_proc_list = tiles_to_match.__col_per_proc_list.copy()

        elif mat_shape_type == "TS":
            if match_dnd.split == 0:
                self.__row_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
            else:
                hld = tiles_to_match.__col_per_proc_list.copy()
                hld[-1] += 1
                self.__row_per_proc_list = hld
            self.__row_inds = tiles_to_match.__row_inds.copy()
            self.__col_inds = tiles_to_match.__row_inds.copy()
            self.__col_per_proc_list = [len(self.__col_inds)] * self.arr.comm.size

        else:  # mat_shape_type == "SF"
            self.__row_inds = tiles_to_match.__row_inds.copy()
            self.__col_inds = tiles_to_match.__row_inds.copy()
            if match_dnd.split == 0:
                self.__row_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
            elif match_dnd.split == 1:
                hld = tiles_to_match.__col_per_proc_list.copy()
                for i in range(tiles_to_match.last_diagonal_process + 1, self.arr.comm.size, 1):
                    hld[i] = 0
                # need to adjust the last diag process,
                ldp = tiles_to_match.last_diagonal_process
                if ldp > 0 and hld[ldp] > hld[ldp - 1]:
                    hld[ldp] = hld[ldp - 1]

                if ldp == 0:
                    hld[0] = len(self.__row_inds)
                self.__row_per_proc_list = hld

            self.__col_per_proc_list = [len(self.__col_inds)] * self.arr.comm.size

        self.__last_diag_pr = tiles_to_match.last_diagonal_process

        self.__tile_map = self.__create_tile_map(
            self.__row_inds,
            self.__col_inds,
            self.__row_per_proc_list,
            self.__col_per_proc_list,
            self.arr,
        )

    def __match_redist(self, tiles_to_match):
        # helper function to match the split axis of selt.arr to that of tiles_to_match.arr
        base_dnd = self.arr
        match_dnd = tiles_to_match.arr
        mat_shape_type = match_dnd.matrix_shape_classifier()

        target_map = self.lshape_map.clone()
        hold = tiles_to_match.lshape_map[..., match_dnd.split].clone()
        if mat_shape_type == "square":
            if match_dnd.split == 1:
                target_map[..., 1] = target_map[..., 0]
                if hold.sum() != base_dnd.gshape[0]:
                    hold[-1] += base_dnd.gshape[0] - hold.sum()
            else:
                if hold.sum() < base_dnd.gshape[base_dnd.split]:
                    hold[-1] += base_dnd.gshape[base_dnd.split] - hold.sum()
        elif mat_shape_type == "TS":
            if hold.sum() < base_dnd.gshape[0]:
                hold[-1] += base_dnd.gshape[0] - hold.sum()
            if match_dnd.split == 1:
                if hold.sum() < base_dnd.gshape[0]:
                    hold[-1] += base_dnd.gshape[0] - hold.sum()
        else:  # mat_shape_type == "SF"
            if match_dnd.split == 0:
                if hold.sum() < base_dnd.gshape[base_dnd.split]:
                    hold[-1] += base_dnd.gshape[base_dnd.split] - hold.sum()
            if match_dnd.split == 1:
                if hold.sum() > base_dnd.gshape[0]:
                    hold[-1] -= hold.sum() - base_dnd.gshape[0]
                for i in range(base_dnd.comm.size - 1, 1, -1):
                    if all(hold >= 0):
                        break
                    if hold[i] < 0:
                        hold[i - 1] += hold[i]
                        hold[i] = 0
                if tiles_to_match.last_diagonal_process == 0 and any(hold < 0):
                    hold[0] = match_dnd.gshape[0]
                    hold[1:] *= 0

        target_map[..., 0 if self.arr.split == 0 else 1] = hold
        self.arr.redistribute_(lshape_map=self.lshape_map, target_map=target_map)

    def match_tiles_qr_lq(self, other_tiles):
        """
        The function is specifically for matching the tiles of the input matrix to its transpose for
        use in the block diagonalization functions. This will not match the tiles exactly, it will make
        adjustments based on the split axis of self.arr as well as to make the diagonal tiles equal
        sizes for the tile row below the diagonal.

        Parameters
        ----------
        other_tiles : SquareDiagTiles
            tiles or the original array
        """
        self.__match_redist(other_tiles)

        col_inds = other_tiles.__row_inds.copy()
        cols_per = other_tiles.__row_per_proc_list.copy()
        row_inds = other_tiles.__col_inds.copy()
        rows_per = other_tiles.__col_per_proc_list.copy()
        ldp = other_tiles.__last_diag_pr
        if ldp < self.arr.comm.size - 1 and other_tiles.arr.split == 1:
            # this loop is to adjust the the tile sizes on the last diagonal process
            #   since the target diagonal is shifted one tile row down the logic is different.
            #   thus the last diagonal tile for other_tiles is not the last diagonal tile for self.
            #   to make it work correctly, the current last diag tile must equal the normal tile size
            #   and the remainder is shifted to the next process
            row_diff = torch.tensor(
                [
                    row_inds[i + 1] - row_inds[i]
                    for i in range(sum(rows_per[:ldp]), sum(rows_per[:ldp]) + rows_per[ldp])
                ],
                dtype=torch.int,
                device=self.arr.device.torch_device,
            )
            row_diff_where = torch.where(row_diff > row_diff[0])[0]
            if row_diff_where.numel() > 0 and row_diff_where[0] < rows_per[ldp] - 1:
                ind = row_diff_where[0] + sum(rows_per[:ldp]) + 1
                diff = row_diff[row_diff_where[0]] - row_inds[1]
                row_inds[ind] -= diff.item()
        # if (
        #     self.arr.gshape[0] > self.arr.gshape[1] + 2
        #     and self.arr.shape[0] > row_inds[1] + row_inds[-1]
        # ):
        #     # add a row before the last element that is equal
        #     row_inds.append(row_inds[1] + row_inds[-1])
        #     if self.arr.split == 1:
        #         rows_per = [r + 1 for r in rows_per]
        #     else:
        #         rows_per[-1] += 1
        # elif (
        #     self.arr.gshape[1] > self.arr.gshape[0] + 2
        #     and self.arr.shape[1] > col_inds[1] + col_inds[-1]
        # ):
        #     # add a row before the last element that is equal
        #     col_inds.append(col_inds[1] + col_inds[-1])
        #     if self.arr.split == 0:
        #         cols_per = [r + 1 for r in cols_per]
        #     else:
        #         cols_per[-1] += 1

        # todo: if the last row/col is larger than the ones before (>1) need to add more rows
        last_diff = min(self.arr.gshape) - row_inds[-1]
        if last_diff > row_inds[1] + 1:
            # print('here')
            # if len(col_inds) > len(row_inds):
            #     row_inds = col_inds.copy()
            # elif len(row_inds) > len(col_inds):
            #     col_inds = row_inds.copy()
            tiles_to_add = last_diff // row_inds[1]
            # print(last_diff, tiles_to_add)
            split_per = rows_per if other_tiles.arr.split == 1 else cols_per
            # print("\nstart", split_per, col_inds)
            for i in range(tiles_to_add):
                if (
                    row_inds[-1] + row_inds[1] >= min(self.arr.gshape) - 1
                    or col_inds[-1] + col_inds[1] >= min(self.arr.gshape) - 1
                ):
                    # print('here2', row_inds[-1] + row_inds[1])
                    tiles_to_add -= 1
                    break
                row_inds.append(row_inds[-1] + row_inds[1])
                col_inds.append(col_inds[-1] + col_inds[1])
                split_per[-1] += 1
            # print("end", col_inds, row_inds, split_per, '\n')
            ttl_tiles = sum(split_per)
            if self.arr.split == 1:
                rows_per = [ttl_tiles for _ in rows_per]
                cols_per = split_per
                # rows_per = [r + tiles_to_add for r in rows_per]
                # cols_per[-1] += tiles_to_add
            else:
                cols_per = [ttl_tiles for _ in rows_per]
                rows_per = split_per
                # cols_per = [r + tiles_to_add for r in cols_per]
                # rows_per[-1] += tiles_to_add

            # todo: add a diagonal end column/row if the difference is right
            # gshape_diff = self.arr.gshape[0] - self.arr.gshape[1]
            # print(self.arr.gshape[0], self.arr.gshape[1], gshape_diff)
            # todo: this doesnt work because the it means that the last block can be 3x2,
            #       which is a problem, need a fix for this...
            # if gshape_diff >= 2:  # rows larger than columns
            #     row_inds.append(self.arr.gshape[1])
            #     # row_inds.append(row_inds[-1] + row_inds[1])
            #     if self.arr.split == 0:
            #         rows_per[-1] += 1
            #     else:
            #         rows_per = [r + 1 for r in rows_per]
            # elif gshape_diff <= -2:  # cols larger than rows
            #     col_inds.append(self.arr.gshape[0])
            #     # col_inds.append(col_inds[-1] + col_inds[1])
            #     if self.arr.split == 1:
            #         cols_per[-1] += 1
            #     else:
            #         cols_per = [c + 1 for c in cols_per]

            # print('\t', col_inds)
            # update the other_tiles elements and make the new tile map for it
            other_tiles.__col_inds = row_inds
            other_tiles.__row_inds = col_inds.copy()
            other_tiles.__col_per_proc_list = rows_per
            other_tiles.__row_per_proc_list = cols_per
            other_tiles.__tile_map = self.__create_tile_map(
                col_inds, row_inds, cols_per, rows_per, other_tiles.arr
            )
            # need to adjust the
            # col_inds = [r - row_inds[1] for r in row_inds][1:len(col_inds) + 1]

            # print(cols_per, col_inds)
        self.__col_inds = col_inds
        self.__col_per_proc_list = cols_per
        self.__last_diag_pr = ldp
        self.__row_inds = row_inds
        self.__row_per_proc_list = rows_per
        self.__tile_map = self.__create_tile_map(row_inds, col_inds, rows_per, cols_per, self.arr)

    def change_row_and_column_index(self, row, column, position_from_end):
        """
        Add a row to the tiles

        Parameters
        ----------
        index : int
            row index to add

        Returns
        -------

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

    # def add_column_to_end(self, index):

    def set_arr(self, arr):
        """
        Set the DNDarray of self to be arr

        Parameters
        ----------
        arr : DNDarray
        """
        if not isinstance(arr, dndarray.DNDarray):
            raise TypeError("arr must be a DNDarray, currently is {}".format(type(arr)))
        self.__DNDarray = arr

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
