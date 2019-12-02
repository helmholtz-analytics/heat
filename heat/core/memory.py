import numpy as np
import torch
from . import dndarray

__all__ = ["copy", "sanitize_memory_layout"]


def copy(a):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : ht.DNDarray
        Input data to be copied.

    Returns
    -------
    copied : ht.DNDarray
        A copy of the original
    """
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("input needs to be a tensor")
    return dndarray.DNDarray(
        a._DNDarray__array.clone(), a.shape, a.dtype, a.split, a.device, a.comm
    )


def sanitize_memory_layout(x, order="C"):
    """
    Return the given object with memory layout as defined below.

    Parameters
    -----------

    x: torch.tensor
        Input data

    order: str, optional.
        Default is 'C' as in C-like (row-major) memory layout. The array is stored first dimension first (rows first if ndim=2).
        Alternative is 'F', as in Fortran-like (column-major) memory layout. The array is stored last dimension first (columns first if ndim=2).
    """
    if x.ndim < 2:
        # do nothing
        return x
    dims = list(range(x.ndim))
    stride = list(x.stride())
    row_major = all(np.diff(stride) <= 0)
    column_major = all(np.diff(stride) >= 0)
    if (order == "C" and row_major) or (order == "F" and column_major):
        # do nothing
        return x
    if (order == "C" and column_major) or (order == "F" and row_major):
        dims = tuple(reversed(dims))
        y = torch.empty_like(x)
        permutation = x.permute(dims).contiguous()
        y = y.set_(
            permutation.storage(),
            x.storage_offset(),
            x.shape,
            tuple(reversed(permutation.stride())),
        )
    if order == "K":
        raise NotImplementedError(
            "Internal usage of torch.clone() means losing original memory layout for now. \n Please specify order='C' for row-major, order='F' for column-major layout."
        )
    return y
