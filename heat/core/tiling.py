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
        tile_dims = torch.zeros((arr.numdims, arr.comm.size), device=arr.device.torch_device)
        if arr.split is not None:
            tile_dims[arr.split] = lshape_map[..., arr.split]
        w_size = arr.comm.size
        for ax in range(arr.numdims):
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
            [tile_dims[x].numel() for x in range(arr.numdims)],
            dtype=torch.int64,
            device=arr.device.torch_device,
        )
        if split is None:
            tile_locations += arr.comm.rank
            return tile_locations
        arb_slice = [slice(None)] * arr.numdims
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
        arb_slices = [None] * arr.numdims
        # print(self.tile_locations[key])
        end_rank = (
            max(self.tile_locations[key].unique())
            if self.tile_locations[key].unique().numel() > 1
            else self.tile_locations[key]
        )

        if isinstance(key, int):
            key = [key]
        if len(key) < arr.numdims or key[-1] is None:
            lkey = list(key)
            lkey.extend([slice(0, None)] * (arr.numdims - len(key)))
            key = lkey
        for d in range(arr.numdims):
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
    """Generate the tile map and the other objects which may be useful.
    The tiles generated here are based of square tiles along the diagonal. The size of these
    tiles along the diagonal dictate the divisions across all processes. If
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
    tiles_per_proc : int, optional
        Default = 2
        the number of divisions per process,

    Properties
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

    def __init__(self, arr, tiles_per_proc=2):
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
        # todo: abstract this
        comp = arr.gshape[0] / arr.gshape[1]
        if 2.0 / arr.comm.size <= comp <= arr.comm.size / 2.0:
            # if the diagonal crosses at least have the processes,
            # todo: tune this, might be faster to broaden this ratio
            mat_shape_type = "square"
        elif comp > arr.comm.size / 2.0:
            mat_shape_type = "TS"
        else:
            mat_shape_type = "SF"

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

            col_inds = [0] + torch.tensor([tile_shape] * ntiles).cumsum(0).tolist()
            if rem > 0:
                col_inds[-1] += rem
            row_per_proc_list = [0] * arr.comm.size
            row_inds = col_inds.copy()
            col_inds = col_inds[:-1]
            col_per_proc_list = [len(col_inds)] * arr.comm.size
            # print('c', col_per_proc_list)

            redist_vec = lshape_map[..., arr.split].clone()
            for i in range(st_ldp):
                redist_vec[i] = tile_shape * tiles_per_proc
                # remaining_tiles -= redist_vec[i]
                diff = redist_vec[i] - lshape_map[..., arr.split][i]
                if diff < 0:
                    redist_vec[i + 1] -= diff
            if redist_vec[st_ldp] > tile_shape:
                # include the process with however many shape will fit
                diag_rem_el = mgshape - sum(redist_vec[:st_ldp])
                proc_space_after_diag = redist_vec[st_ldp] - diag_rem_el
                if proc_space_after_diag < 0:
                    raise ValueError("wtf? proc space after diag: {}".format(proc_space_after_diag))

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
                else:
                    # more space after the diagonal (more than 1 element) -> proc space > 1
                    divs_per_proc[st_ldp] += 1
                    row_inds.append(mgshape + proc_space_after_diag.item())

            target_map = lshape_map.clone()
            target_map[..., arr.split] = redist_vec
            arr.redistribute_(lshape_map, target_map)
            # next, chunk each process
            for pr in range(arr.comm.size):
                if pr <= last_diag_pr:
                    row_per_proc_list[pr] = divs_per_proc[pr]
                else:
                    base_sz = redist_vec[pr] // tiles_per_proc
                    rem = redist_vec[pr] % tiles_per_proc
                    lcl_inds = torch.tensor([base_sz] * tiles_per_proc)
                    for r in range(rem):
                        lcl_inds[r] += 1
                    lcl_inds = [i.item() + row_inds[-1] for i in lcl_inds.cumsum(0)]
                    row_inds.extend(lcl_inds)
                    row_per_proc_list[pr] = len(lcl_inds)
            row_inds = row_inds[:-1]
            if mat_shape_type == "SF":
                hld = row_inds
                row_inds = col_inds
                col_inds = hld
                hld = col_per_proc_list
                col_per_proc_list = row_per_proc_list
                row_per_proc_list = hld
                # hld
        else:
            # print(mat_shape_type)
            # this covers the following cases:
            #       Square -> both splits
            #       TS -> split == 1  # diag covers whole size
            #       SF -> split == 0  # diag covers whole size
            divs_per_proc = [tiles_per_proc] * arr.comm.size
            mgshape = min(arr.gshape)
            ntiles = arr.comm.size * tiles_per_proc
            tile_shape, rem = mgshape // ntiles, mgshape % ntiles
            if tile_shape == 1:  # this can be raised as these matrices are mostly square
                raise ValueError(
                    "Dataset too small for tiles to be useful, resplit to None if possible"
                )
            shape_rem = (
                arr.gshape[0] - arr.gshape[1] if arr.split == 0 else arr.gshape[1] - arr.gshape[0]
            )
            divs = [tile_shape] * ntiles
            redist_v_sp, c = [], 0
            for i in range(arr.comm.size):
                st = c
                ed = divs_per_proc[i] + c
                redist_v_sp.append(sum(divs[st:ed]))
                c = ed
            redist_v_sp = torch.tensor(redist_v_sp)
            # original assumption is that there are tiles_per_proc tiles on each process with
            #       an maximum of ((m - n) % sz) * n * tiles_per_proc  additionally on the last process
            #       (for m x n)
            # print(redist_v_sp, rem)
            # todo: if determine distribution -> evenly divide it between the last processes
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
                # divs_per_proc[-1] += 1
                # print('r', row_inds, divs_per_proc)
                # todo: does an extra tile need to be added??
                #       -> only if tiles/proc > 1 and last_diag_pr != size
                # if arr.gshape[0] == mgshape:  # m < n
                #     pass
                #     # col_inds.append(div_inds1[-1].item())
                # elif arr.gshape[0] > arr.gshape[1]:  # m > n
                #     # row_inds.append(div_inds1[-1].item())
                #     divs_per_proc[-1] += 1

            if mat_shape_type == "TS" and arr.split == 1:
                # need to add a division after the diagonal here
                row_inds.append(arr.gshape[1])
                ntiles += 1
            col_per_proc_list = [ntiles] * arr.comm.size if arr.split == 0 else divs_per_proc
            row_per_proc_list = [ntiles] * arr.comm.size if arr.split == 1 else divs_per_proc

            target_map = lshape_map.clone()
            target_map[..., arr.split] = redist_v_sp
            arr.redistribute_(lshape_map, target_map)

            divs = torch.tensor(divs, dtype=torch.int32, device=arr.device.torch_device)
            div_inds = divs.cumsum(dim=0)
            fr_tile_ge = torch.where(div_inds >= mgshape)[0][0]
            if mgshape == arr.gshape[0]:
                last_diag_pr = arr.comm.size - 1
            elif mgshape == div_inds[fr_tile_ge]:
                last_diag_pr = torch.floor_divide(fr_tile_ge, tiles_per_proc).item()
            elif fr_tile_ge == 0:  # -> first tile above is 0, thus the last diag pr is also 0
                last_diag_pr = 0
            else:
                last_diag_pr = torch.where(fr_tile_ge < torch.tensor(divs_per_proc).cumsum(dim=0))[
                    0
                ][0].item()

        tile_map = torch.zeros(
            [len(row_inds), len(col_inds), 3], dtype=torch.int, device=arr._DNDarray__array.device
        )

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

        self.__DNDarray = arr
        self.__col_per_proc_list = col_per_proc_list
        self.__lshape_map = lshape_map
        self.__last_diag_pr = last_diag_pr
        self.__row_per_proc_list = row_per_proc_list
        self.__tile_map = tile_map
        self.__row_inds = row_inds
        self.__col_inds = col_inds

    @property
    def arr(self):
        """
        Returns
        -------
        DNDarray : the DNDarray for which the tiles are defined on
        """
        return self.__DNDarray

    def set_arr(self, arr):
        self.__DNDarray = arr

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
            stop = row_inds[key[0].stop] if key[0].stop is not None else row_inds[-1]
            st0, sp0 = start - row_start, stop - row_start
        if isinstance(key[1], int):
            st1 = col_inds[key[1]] - col_start
            sp1 = col_inds[key[1] + 1] - col_start
        elif isinstance(key[1], slice):
            start = col_inds[key[1].start] if key[1].start is not None else 0
            # print(len(col_inds), col_inds, key)
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
        if not isinstance(key, (int, tuple, slice)):
            raise TypeError(
                "key must be an int, tuple, or slice, is currently {}".format(type(key))
            )
        involved_procs = tile_map[key][..., 2].unique()
        if involved_procs.nelement() == 1 and involved_procs == arr.comm.rank:
            # print('k', key)
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
        # print(key)
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
        # print(key)
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
        comp = match_dnd.gshape[0] / match_dnd.gshape[1]
        if 2.0 / match_dnd.comm.size <= comp <= match_dnd.comm.size / 2.0:
            # if the diagonal crosses at least have the processes,
            # todo: tune this, might be faster to broaden this ratio
            mat_shape_type = "square"
        elif comp > match_dnd.comm.size / 2.0:
            mat_shape_type = "TS"
        else:
            mat_shape_type = "SF"
        # this map will take the same tile row and column sizes up to the last diagonal row/column
        # the last row/column is determined by the number of rows/columns on the non-split dimension

        # todo: cases: mostly square (m > n), mostly square (m < n), TS sp0, TS sp1, SF sp0, SF sp1
        # todo: things to set: row inds, col inds, rows/proc, cols/proc, last diagonal process
        if mat_shape_type == "square":
            # todo: combine these 2 cases, lots of repetition here
            if match_dnd.gshape[0] > match_dnd.gshape[1] and match_dnd.split == 1:
                # match lshape
                target_map = tiles_to_match.lshape_map.clone()
                hold = target_map[..., 1].clone()
                target_map[..., 1] = target_map[..., 0]
                if hold.sum() < base_dnd.gshape[0]:
                    hold[-1] += base_dnd.gshape[0] - hold.sum()
                target_map[..., 0] = hold
                base_dnd.redistribute_(self.lshape_map, target_map)
                self.__row_inds = tiles_to_match.__row_inds
                self.__col_inds = tiles_to_match.__col_inds
                self.__row_per_proc_list = tiles_to_match.__col_per_proc_list
                self.__col_per_proc_list = tiles_to_match.__row_per_proc_list
                self.__last_diag_pr = tiles_to_match.last_diagonal_process
            if match_dnd.gshape[0] > match_dnd.gshape[1] and match_dnd.split == 0:
                # match lshape
                target_map = self.lshape_map.clone()
                hold = tiles_to_match.lshape_map[..., 0].clone()
                if hold.sum() < base_dnd.gshape[0]:
                    hold[-1] += base_dnd.gshape[0] - hold.sum()
                target_map[..., 0] = hold
                base_dnd.redistribute_(self.lshape_map, target_map)
                self.__row_inds = tiles_to_match.__row_inds
                self.__col_inds = tiles_to_match.__col_inds
                self.__row_per_proc_list = tiles_to_match.__row_per_proc_list
                self.__col_per_proc_list = tiles_to_match.__col_per_proc_list
                self.__last_diag_pr = tiles_to_match.last_diagonal_process
            # for mostly square matrices the diagonal is stretched to the last process

        elif mat_shape_type == "TS":
            if match_dnd.split == 0:
                target_map = self.lshape_map.clone()
                hold = tiles_to_match.lshape_map[..., 0].clone()
                if hold.sum() < base_dnd.gshape[0]:
                    hold[-1] += base_dnd.gshape[0] - hold.sum()
                target_map[..., 0] = hold
                base_dnd.redistribute_(self.lshape_map, target_map)
                self.__row_inds = tiles_to_match.__row_inds
                self.__col_inds = tiles_to_match.__row_inds
                self.__row_per_proc_list = tiles_to_match.__row_per_proc_list
                self.__col_per_proc_list = [len(self.__col_inds)] * base_dnd.comm.size
                self.__last_diag_pr = tiles_to_match.last_diagonal_process
            elif match_dnd.split == 1:
                target_map = self.lshape_map.clone()
                hold = tiles_to_match.lshape_map[..., 1].clone()
                if hold.sum() < base_dnd.gshape[0]:
                    hold[-1] += base_dnd.gshape[0] - hold.sum()
                target_map[..., 0] = hold
                base_dnd.redistribute_(self.lshape_map, target_map)
                self.__row_inds = tiles_to_match.__row_inds  # + [match_dnd.gshape[1]]
                self.__col_inds = tiles_to_match.__row_inds  # + [match_dnd.gshape[1]]
                hld = tiles_to_match.__col_per_proc_list.copy()
                hld[-1] += 1
                self.__row_per_proc_list = hld
                self.__col_per_proc_list = [len(self.__col_inds)] * base_dnd.comm.size
                self.__last_diag_pr = tiles_to_match.last_diagonal_process

        else:  # mat_shape_type == "SF"
            if match_dnd.split == 0:
                target_map = self.lshape_map.clone()
                hold = tiles_to_match.lshape_map[..., 0].clone()
                if hold.sum() < base_dnd.gshape[0]:
                    hold[-1] += base_dnd.gshape[0] - hold.sum()
                target_map[..., 0] = hold
                base_dnd.redistribute_(self.lshape_map, target_map)
                self.__row_inds = tiles_to_match.__row_inds
                self.__col_inds = tiles_to_match.__row_inds
                self.__row_per_proc_list = tiles_to_match.__row_per_proc_list
                self.__col_per_proc_list = [len(self.__col_inds)] * base_dnd.comm.size
                self.__last_diag_pr = tiles_to_match.last_diagonal_process
            elif match_dnd.split == 1:
                target_map = self.lshape_map.clone()
                hold = tiles_to_match.lshape_map[..., 1].clone()
                if hold.sum() > base_dnd.gshape[0]:
                    hold[-1] -= hold.sum() - base_dnd.gshape[0]
                for i in range(base_dnd.comm.size - 1, 1, -1):
                    if all(hold >= 0):
                        break
                    if hold[i] < 0:
                        hold[i - 1] += hold[i]
                        hold[i] = 0
                target_map[..., 0] = hold
                # print(target_map)
                base_dnd.redistribute_(self.lshape_map, target_map)
                self.__row_inds = tiles_to_match.__row_inds  # + [match_dnd.gshape[1]]
                self.__col_inds = tiles_to_match.__row_inds  # + [match_dnd.gshape[1]]
                hld = tiles_to_match.__col_per_proc_list.copy()
                for i in range(tiles_to_match.last_diagonal_process + 1, base_dnd.comm.size):
                    hld[i] = 0
                # need to adjust the last diag process,
                ldp = tiles_to_match.last_diagonal_process
                if hld[ldp] > hld[ldp - 1]:
                    hld[ldp] = hld[ldp - 1]
                self.__row_per_proc_list = hld
                self.__col_per_proc_list = [len(self.__col_inds)] * base_dnd.comm.size
                self.__last_diag_pr = tiles_to_match.last_diagonal_process

        tile_map = torch.zeros(
            [len(self.__row_inds), len(self.__col_inds), 3],
            dtype=torch.int,
            device=base_dnd._DNDarray__array.device,
        )

        for num, c in enumerate(self.__col_inds):  # set columns
            tile_map[:, num, 1] = c
        for num, r in enumerate(self.__row_inds):  # set rows
            tile_map[num, :, 0] = r

        for i in range(base_dnd.comm.size):
            st = sum(self.__row_per_proc_list[:i])
            sp = st + self.__row_per_proc_list[i]
            tile_map[..., 2][st:sp] = i
        # to adjust if the last process has more tiles
        i = base_dnd.comm.size - 1
        tile_map[..., 2][sum(self.__row_per_proc_list[:i]) :] = i

        if base_dnd.split == 1:
            st = 0
            for pr, cols in enumerate(self.__col_per_proc_list):
                tile_map[:, st : st + cols, 2] = pr
                st += cols
        self.__tile_map = tile_map
        # elif :

        # if base_dnd.split == match_dnd.split == 0 and base_dnd.shape[0] == match_dnd.shape[0]:
        #     # this implies that the gshape[0]'s are equal
        #     # rows are the exact same, and the cols are also equal to the rows (square matrix)
        #     base_dnd.redistribute_(lshape_map=self.lshape_map, target_map=tiles_to_match.lshape_map)
        #
        #     self.__row_per_proc_list = tiles_to_match.__row_per_proc_list.copy()
        #     self.__col_per_proc_list = [tiles_to_match.tile_rows] * len(self.__row_per_proc_list)
        #     self.__row_inds = (
        #         tiles_to_match.__row_inds.copy()
        #         if base_dnd.gshape[0] >= base_dnd.gshape[1]
        #         else tiles_to_match.__col_inds.copy()
        #     )
        #     self.__col_inds = (
        #         tiles_to_match.__row_inds.copy()
        #         if base_dnd.gshape[0] >= base_dnd.gshape[1]
        #         else tiles_to_match.__col_inds.copy()
        #     )
        #     # todo: problem is in here somewhere, some index isnt being set correctly, which one?
        #
        #     self.__tile_map = torch.zeros(
        #         (self.tile_rows, self.tile_columns, 3),
        #         dtype=torch.int,
        #         device=match_dnd._DNDarray__array.device,
        #     )
        #     for i in range(self.tile_rows):
        #         self.__tile_map[..., 0][i] = self.__row_inds[i]
        #     for i in range(self.tile_columns):
        #         self.__tile_map[..., 1][:, i] = self.__col_inds[i]
        #     for i in range(self.arr.comm.size - 1):
        #         st = sum(self.__row_per_proc_list[:i])
        #         sp = st + self.__row_per_proc_list[i]
        #         self.__tile_map[..., 2][st:sp] = i
        #     # to adjust if the last process has more tiles
        #     i = self.arr.comm.size - 1
        #     self.__tile_map[..., 2][sum(self.__row_per_proc_list[:i]) :] = i
        # elif base_dnd.split == 0 and match_dnd.split == 1:
        #     # rows determine the q sizes -> cols = rows
        #     # col inds, row inds, rows per, cols per, last diag pr
        #     # if base_dnd.gshape[0] <= base_dnd.gshape[1]:
        #     #     self.__col_inds = tiles_to_match.row_indices
        #     #     self.__row_inds = tiles_to_match.col_indices
        #     # self.__col_inds = (
        #     #     tiles_to_match.__row_inds.copy()
        #     #     if base_dnd.gshape[0] <= base_dnd.gshape[1]
        #     #     else tiles_to_match.__col_inds.copy()
        #     # )
        #     #
        #     # self.__row_inds = (
        #     #     tiles_to_match.__row_inds.copy()
        #     #     if base_dnd.gshape[0] <= base_dnd.gshape[1]
        #     #     else tiles_to_match.__col_inds.copy()
        #     # )
        #     # print('h', self.__col_inds, tiles_to_match.col_indices, tiles_to_match.lshape_map)
        #     self.__row_inds = [r for r in tiles_to_match.__row_inds if r < base_dnd.gshape[0]]
        #     self.__col_inds = [r for r in tiles_to_match.__col_inds if r < base_dnd.gshape[1]]
        #     if match_dnd.gshape[0] <= match_dnd.gshape[1]:
        #         rows_per = tiles_to_match.__row_per_proc_list.copy()
        #     else:
        #         rows_per = tiles_to_match.__col_per_proc_list.copy()
        #     if sum(rows_per) < tiles_to_match.tile_rows_per_process[0]:
        #         rows_per[-1] += 1
        #     if len(self.__row_inds) == base_dnd.comm.size:
        #         rows_per = [1 for _ in rows_per]
        #
        #     # print(tiles_to_match.lshape_map)
        #     # rows_per = [x for x in self.__col_inds if x < base_dnd.shape[0]]
        #
        #     targe_map = self.lshape_map.clone()
        #     target_0 = torch.zeros_like(tiles_to_match.lshape_map[..., 1])
        #     c = 0
        #     # print(rows_per)
        #     i = 0
        #     for i in range(base_dnd.comm.size):
        #         try:
        #             end_ind = self.__row_inds[rows_per[i]]
        #             # print(i, rows_per[i], end_ind)
        #             target_0[i] = end_ind - c
        #         # if end_ind < base_dnd.gshape[0]:
        #         except IndexError:
        #             target_0[i] = base_dnd.gshape[0] - c
        #             rows_per[i] = i
        #             rows_per[i + 1:] *= 0
        #             # cols_per =
        #             # print(rows_per, i)
        #             break
        #         c = end_ind
        #     cols_per = [sum(rows_per)] * len(rows_per)
        #     rows_per.extend([0] * (base_dnd.comm.size - i - 1))
        #     cols_per.extend([0] * (base_dnd.comm.size - i - 1))
        #
        #     # print('row', self.row_indices)
        #     targe_map[..., 0] = target_0
        #     # print(base_dnd.gshape, match_dnd.gshape, target_0)
        #     # if match_dnd.gshape[0] != match_dnd.gshape[1]:
        #     #     target_0[-1] -= match_dnd.gshape[1] - match_dnd.gshape[0]
        #
        #     base_dnd.redistribute_(lshape_map=self.lshape_map, target_map=targe_map)
        #
        #     # if base_dnd.gshape[0] > base_dnd.gshape[1]:
        #     #     row_inds.append(row_inds[-1] + row_inds[1])
        #     #     rows_per[-1] += 1
        #
        #     # self.__row_inds = row_inds
        #
        #     self.__tile_map = torch.zeros(
        #         (self.tile_rows, self.tile_columns, 3),
        #         dtype=torch.int,
        #         device=tiles_to_match.arr._DNDarray__array.device,
        #     )
        #
        #     for i in range(len(self.__row_inds)):
        #         self.__tile_map[..., 0][i] = self.__row_inds[i]
        #     for i in range(len(self.__col_inds)):
        #         self.__tile_map[..., 1][:, i] = self.__col_inds[i]
        #     for i in range(self.arr.comm.size):
        #         st = sum(self.__row_per_proc_list[:i])
        #         sp = st + self.__row_per_proc_list[i]
        #         self.__tile_map[..., 2][st:sp] = i
        #     # to adjust if the last process has more tiles
        #     i = self.arr.comm.size - 1
        #
        #     self.__tile_map[..., 2][sum(self.__row_per_proc_list[:i]) :] = i
        #     self.__col_per_proc_list = cols_per
        #     # print(self.__col_inds, self.tile_columns, self.__col_per_proc_list)
        #     self.__row_per_proc_list = rows_per
        #     self.__last_diag_pr = tiles_to_match.__last_diag_pr
        # else:
        #     raise NotImplementedError(
        #         "splits not implements, {}, {}".format(self.arr.split, tiles_to_match.arr.split)
        #     )

    def match_tiles_qr_lq(self, other_tiles):
        base_dnd = self.arr
        other_dnd = other_tiles.arr
        # match lshape redistribute if needed
        target_map = torch.zeros_like(self.lshape_map)
        target_map[..., 0] = other_tiles.lshape_map[..., 1]
        target_map[..., 1] = other_tiles.lshape_map[..., 0]
        base_dnd.redistribute_(lshape_map=self.lshape_map, target_map=target_map)
        cols_per = other_tiles.__row_per_proc_list.copy()
        rows_per = other_tiles.__col_per_proc_list.copy()

        if base_dnd.split == 1 and other_dnd.split == 0:
            # the lshapes are used for the tiles because we need to have 1 tile/process, elsewise the tiles
            #   do not line up. more sophisticated redistribution is required for that case.
            # todo: if multiple tiles desired -> need the hard splits to have the number of tiles as a divisor actually
            #  this could be an easy way to redo the tiling class to reduce the difficulties there
            row_inds = other_tiles.__col_inds.copy()
            col_inds = other_tiles.__row_inds.copy()
            col_inds = [c for c in col_inds if c < base_dnd.gshape[1]]
            row_inds = [r for r in row_inds if r < base_dnd.gshape[0]]

            # print(base_dnd.gshape, other_dnd.shape)
            # # todo: need to 1 N rows after the end of the diagonal, or just one at the gshape?
            if base_dnd.shape[0] > base_dnd.shape[1] + 1:
                # print(row_inds, col_inds, self.__col_inds, self.__row_inds)
                row_inds.append(base_dnd.gshape[1] - 2)
                rows_per = [r + 1 for r in rows_per]
                # print(rows_per)

            # if base_dnd.gshape[0] >= base_dnd.gshape[1] + row_inds[1]:
            #     offset = row_inds[1]
            #     row_inds = [r + offset for r in other_tiles.row_indices]
            #     row_inds = [0] + row_inds

            if len(row_inds) > 1:
                # comp_size =
                last_diag_pr = torch.where(
                    torch.cumsum(self.lshape_map[..., 1], dim=0) >= base_dnd.gshape[0] - row_inds[1]
                )[0]
                last_diag_pr = (
                    last_diag_pr[0] if last_diag_pr.numel() > 0 else base_dnd.comm.size - 1
                )
            else:
                last_diag_pr = 0

            # todo: abstract the tile map setting
            tile_map = torch.zeros(
                (len(row_inds), len(col_inds), 3),
                dtype=other_tiles.tile_map.dtype,
                device=other_tiles.tile_map.device,
            )
            for i in range(len(row_inds)):
                tile_map[..., 0][i] = row_inds[i]
            for i in range(len(col_inds)):
                tile_map[..., 1][:, i] = col_inds[i]
            st = 0
            for pr, cols in enumerate(cols_per):
                tile_map[:, st : st + cols, 2] = pr
                st += cols
            # print(tile_map)

        elif base_dnd.split == 0 and other_dnd.split == 1:
            # only working for 1 tile
            # need to adjust the col_inds here
            # the cols should start at 0,0 then the next one should be plus the size of the first
            # after that it should be
            row_inds = other_tiles.__col_inds.copy()
            col_inds = other_tiles.__row_inds.copy()
            col_inds = [c for c in col_inds if c < base_dnd.gshape[1]]
            row_inds = [r for r in row_inds if r < base_dnd.gshape[0]]

            if len(col_inds) > 1:
                last_diag_pr = torch.where(
                    torch.cumsum(self.lshape_map[..., 0], dim=0)
                    >= min(base_dnd.gshape) + col_inds[1]
                )[0]
                last_diag_pr = (
                    last_diag_pr[0] if last_diag_pr.numel() > 0 else base_dnd.comm.size - 1
                )
            else:
                last_diag_pr = 0

            tile_map = torch.zeros(
                (len(row_inds), len(col_inds), 3),
                dtype=other_tiles.tile_map.dtype,
                device=other_tiles.tile_map.device,
            )

            for i in range(len(row_inds)):
                tile_map[..., 0][i] = row_inds[i]
            for i in range(len(col_inds)):
                tile_map[..., 1][:, i] = col_inds[i]
            for i in range(self.arr.comm.size - 1):
                st = sum(rows_per[:i])
                sp = st + rows_per[i]
                tile_map[..., 2][st:sp] = i
            i = self.arr.comm.size - 1
            tile_map[..., 2][sum(rows_per[:i]) :] = i

        else:
            raise NotImplementedError("Both DNDarrays must have different splits")
        self.__col_inds = list(col_inds)
        self.__col_per_proc_list = cols_per
        self.__last_diag_pr = last_diag_pr
        self.__row_per_proc_list = rows_per
        self.__tile_map = tile_map
        self.__row_inds = list(row_inds)

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
