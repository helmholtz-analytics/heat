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
        # super(DNDarray, self).__init__()
        # lshape_map -> rank (int), lshape (tuple of the local lshape, self.lshape)
        # block_map -> rank, tile row, tile column, tile size
        # if not isinstance(self, DNDarray):
        #     raise TypeError('self must be a DNDarray, is currently a {}'.format(type(self)))

        lshape_map = torch.zeros((arr.comm.size, len(arr.gshape)), dtype=int)
        lshape_map[arr.comm.rank, :] = torch.Tensor(arr.lshape)
        arr.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)
        # =================================================================================================
        self.__lshape_map = lshape_map
        # =================================================================================================

        # chunk map
        # is the diagonal crossed by a division between processes/where
        last_diag_pr = torch.where(lshape_map[..., arr.split].cumsum(dim=0) >= min(arr.gshape))[0][0]
        # adjust for small blocks on the last diag pr:
        rem_cols_last_pr = min(arr.gshape) - lshape_map[..., arr.split].cumsum(dim=0)[last_diag_pr - 1]  # end of the process before the split
        last_tile_cols = tile_rows
        while rem_cols_last_pr / last_tile_cols < 10:
            # if there cannot be tiles formed which are at list ten items large then need to reduce the number of tiles
            last_tile_cols -= 1
            if last_tile_cols == 1:
                break

        tile_columns = tile_rows * last_diag_pr + last_tile_cols

        tiles_per_process = [arr.comm.size, tile_rows, int(tile_columns), 2]
        # units: process, # of rows per process, number of total tile rows (also the number of columns), tile indices
        domain_tile_shapes = torch.zeros(tiles_per_process, dtype=torch.int)

        diag_crossings = lshape_map[..., arr.split].cumsum(dim=0)[:last_diag_pr + 1]
        diag_crossings[-1] = diag_crossings[-1] if diag_crossings[-1] <= min(arr.gshape) else min(arr.gshape)

        diag_crossings = torch.cat((torch.tensor([0]), diag_crossings), dim=0)
        for col in range(tile_columns):
            _, lshape, _ = arr.comm.chunk([diag_crossings[col // tile_rows + 1] - diag_crossings[col // tile_rows]], 0,
                                          rank=int(col % tile_rows), w_size=tile_rows if col // tile_rows != last_diag_pr else last_tile_cols)
            domain_tile_shapes[col // tile_rows, col % tile_rows, :(col + 1), 0] = lshape[0]
            domain_tile_shapes[:, :, col, 1] = lshape[0]

        for pr in range(arr.comm.size):
            # test if the data accounted for in the first dimension is == to the 0th dim
            unq = domain_tile_shapes[pr, :tile_rows, 0, 0]
            if unq.sum() != lshape_map[pr, 0].sum():
                if unq[0] == 0:  # this is the case that there *are not* values in the 0th dim here
                    for row in range(tile_rows):
                        _, lshape, _ = arr.comm.chunk(lshape_map[pr], 0, rank=int(row), w_size=tile_rows)
                        domain_tile_shapes[pr, row, :, 0] = lshape[0]
                else:  # need to adjust the last tile to be rectangular (just the difference from the bottom tile to the end of the process
                    # print(domain_tile_shapes[pr, :-1, 0, 0])
                    domain_tile_shapes[pr, tile_rows - last_tile_cols - 1, :, 0] = lshape_map[pr, 0] - domain_tile_shapes[pr, :-1, 0, 0].sum()

        unq = domain_tile_shapes[-1, 0, :, 1]
        if unq.sum() < arr.gshape[1]:
            domain_tile_shapes[-1, tile_rows - last_tile_cols - 1, -1, 1] = arr.gshape[1] - unq.sum() + \
                                                                            domain_tile_shapes[-1, tile_rows - last_tile_cols - 1, -1, 1]
        num_local_row_tiles = [tile_rows] * arr.comm.size
        if torch.all(domain_tile_shapes[-1, -1, :, 0] == 0):
            num_local_row_tiles[-1] -= 1

        # =================================================================================================
        self.__chunk_map = domain_tile_shapes
        # =================================================================================================
        self.__DNDarray = arr

    @property
    def tile_map(self):
        return self.__chunk_map

    @property
    def lsahpe_map(self):
        return self.__lshape_map

    def __getitem__(self, item):
        # default getitem will return the
        pass
    # need to get:
    # tile data
    # tile start
    # tile end
    # tile size
    #
