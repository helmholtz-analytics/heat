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
    Return the given object with memory layout as defined below. The default memory distribution is assumed.

    Parameters
    -----------

    x: torch.tensor
        Input data

    order: str, optional.
        Default is 'C' as in C-like (row-major) memory layout. The array is stored first dimension first (rows first if ndim=2).
        Alternative is 'F', as in Fortran-like (column-major) memory layout. The array is stored last dimension first (columns first if ndim=2).
    """
    if order == "K":
        raise NotImplementedError(
            "Internal usage of torch.clone() means losing original memory layout for now. \n Please specify order='C' for row-major, order='F' for column-major layout."
        )
    if x.ndim < 2 or x.numel() == 0:
        # do nothing
        return x
    dims = list(range(x.ndim))
    stride = torch.tensor(x.stride())
    # since strides can get a bit wonky with operations like transpose
    #   we should assume that the tensors are row major or are distributed the default way
    sdiff = stride[1:] - stride[:-1]
    column_major = all(sdiff >= 0)
    row_major = True if not column_major else False
    if (order == "C" and row_major) or (order == "F" and column_major):
        # do nothing
        return x
    elif (order == "C" and column_major) or (order == "F" and row_major):
        dims = tuple(reversed(dims))
        y = torch.empty_like(x)
        permutation = x.permute(dims).contiguous()
        y = y.set_(
            permutation.storage(),
            x.storage_offset(),
            x.shape,
            tuple(reversed(permutation.stride())),
        )
        return y
    else:
        raise ValueError(
            "combination of order and layout not permitted, order: {} column major: {} row major: {}".format(
                order, column_major, row_major
            )
        )
