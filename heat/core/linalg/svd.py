import torch

from . import qr

from .. import factories
from .. import tiling

__all__ = ["block_diagoalize", "svd"]


def block_diagoalize(arr, tiles_per_proc=2, overwrite_arr=False):
    # no copies!
    # steps to get ready for loop:
    # 1. tile arr if needed
    # 2. get transpose of arr
    # 3. tile arr_t
    # 4. match tiles to arr
    # ----------------------------------------------------------------------------------------------
    # 1. tile arr if needed
    if not overwrite_arr:
        arr = arr.copy()
    if arr.tiles is None:
        arr.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)

    # 2. get transpose of arr
    arr_t = arr.T
    # 3. tile arr_t
    arr_t.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    # 4. match tiles to arr
    arr_t.tiles.match_tiles(arr.tiles)

    # todo: change the split dynamically -----------------------------------------------------------
    q0 = factories.eye(
        (arr.gshape[0], arr.gshape[0]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q0.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q0.tiles.match_tiles(arr.tiles)

    q1 = factories.eye(
        (arr.gshape[1], arr.gshape[1]), split=0, dtype=arr.dtype, comm=arr.comm, device=arr.device
    )
    q1.create_square_diag_tiles(tiles_per_proc=tiles_per_proc)
    q1.tiles.match_tiles(arr.tiles)
    # ----------------------------------------------------------------------------------------------
    tile_columns = arr.tiles.tile_columns
    # tile_rows_proc = arr.tiles.tile_rows_per_process
    #
    # torch_device = arr._DNDarray__array.device
    # rank = arr.comm.rank
    # proc_tile_start = torch.cumsum(
    #     torch.tensor(arr.tiles.tile_rows_per_process, device=torch_device), dim=0
    # )
    # q0_dict = {}
    # q0_dict_waits = {}

    # looping over number of tile columns - 1 (col)
    # 1. do QR on arr for column=col (standard QR as written)
    # 2. do LQ on arr_t for column=col+1 (standard QR again, the transpose makes it LQ)
    #       both of these steps overwrite arr (or an initial copy of it, optional)
    # 3. do QR on the last column if m >= n
    for col in range(tile_columns - 1):
        # 1. do QR on arr for column=col (standard QR as written) (this assumes split == 0)
        # 2. do full QR on that column for LQ on arr_t
        pass


def svd(arr):
    pass
