import numpy as np
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
    x: torch.tensor
    order: str, optional. can be "K"?, "A"?, "C", "F". #TODO NotImplementedError for "A" 
   
    """
    dims = list(range(x.ndim))
    shape = x.shape
    # assess current layout
    row_major = bool(x.stride()[i] > x.stride()[i + 1] for i in dims)
    column_major = False if row_major else bool(x.stride()[i] < x.stride()[i + 1] for i in dims)
    if not row_major and not column_major:
        raise NotImplementedError(
            "Expecting row/major or column-major memory layout, not implemented for alternative layouts."
        )
    if order == "K":
        raise NotImplementedError(
            "Internal usage of torch.clone() means losing original memory layout for now. \n Please specify order='C' for row-major, order='F' for column-major layout."
        )
    if order == "C" and not row_major:
        new_stride = tuple(np.prod(shape[i + 1 :]) for i in dims[:-1]) + (1,)
    elif order == "F" and not column_major:
        new_stride = (1,) + tuple(np.prod(shape[-x.ndim : -x.ndim + i]) for i in dims[1:])
    dims[0], dims[-1] = dims[-1], dims[0]
    permutation = tuple(dims)
    x = x.permute(permutation).contiguous()
    x = x.set_(x.storage(), x.storage_offset(), shape, new_stride)
    return x
