"""
This module changes the internal memory of an array.
"""

import torch

from . import sanitation
from .dndarray import DNDarray

__all__ = ["copy", "sanitize_memory_layout"]


def copy(x: DNDarray) -> DNDarray:
    """
    Return a deep copy of the given object.

    Parameters
    ----------
    x : DNDarray
        Input array to be copied.

    Examples
    --------
    >>> a = ht.array([1,2,3])
    >>> b = ht.copy(a)
    >>> b
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a[0] = 4
    >>> a
    DNDarray([4, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> b
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(x)
    return DNDarray(x.larray.clone(), x.shape, x.dtype, x.split, x.device, x.comm, x.balanced)


DNDarray.copy = lambda self: copy(self)
DNDarray.copy.__doc__ = copy.__doc__


def sanitize_memory_layout(x: torch.Tensor, order: str = "C") -> torch.Tensor:
    """
    Return the given object with memory layout as defined below. The default memory distribution is assumed.

    Parameters
    -----------
    x: torch.Tensor
        Input data
    order: str, optional.
        Default is ``'C'`` as in C-like (row-major) memory layout. The array is stored first dimension first (rows first if ``ndim=2``).
        Alternative is ``'F'``, as in Fortran-like (column-major) memory layout. The array is stored last dimension first (columns first if ``ndim=2``).
    """
    if order == "K":
        raise NotImplementedError(
            "Internal usage of torch.clone() means losing original memory layout for now. \n Please specify order='C' for row-major, order='F' for column-major layout."
        )
    if x.ndim < 2 or x.numel() == 0:
        # do nothing
        return x
    stride = torch.tensor(x.stride())
    # since strides can get a bit wonky with operations like transpose
    #   we should assume that the tensors are row major or are distributed the default way
    column_major = (stride[1:] - stride[:-1] >= 0).all()
    if (order == "C" and not column_major) or (order == "F" and column_major):
        # do nothing
        return x
    if (order == "C" and column_major) or (order == "F" and not column_major):
        dims = tuple(range(x.ndim - 1, -1, -1))
        storage_offset = x.storage_offset()
        shape = x.shape
        x = x.permute(dims).contiguous()
        reversed_stride = tuple(reversed(x.stride()))
        x.set_(x.storage(), storage_offset, shape, reversed_stride)
        return x

    raise ValueError(
        "combination of order and layout not permitted, order: {} column major: {} row major: {}".format(
            order, column_major, not column_major
        )
    )
