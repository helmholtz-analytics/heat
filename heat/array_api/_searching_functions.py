from __future__ import annotations

from ._array_object import Array
from ._dtypes import _result_type, _numeric_dtypes

from typing import Optional, Tuple

import heat as ht


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Returns the indices of the maximum values along a specified axis. When the
    maximum value occurs multiple times, only the indices corresponding to the
    first occurrence are returned.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    axis : Optional[int]
        Axis along which to search. If ``None``, the function returns the index of
        the maximum value of the flattened array. Default: ``None``.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes (dimensions)
        are not be included in the result. Default: ``False``.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in argmax")
    if axis is None:
        if keepdims:
            output_shape = tuple(1 for _ in range(x.ndim))
        else:
            output_shape = ()
    else:
        if axis < 0:
            axis += x.ndim
        if keepdims:
            output_shape = tuple(dim if i != axis else 1 for i, dim in enumerate(x.shape))
        else:
            output_shape = tuple(dim for i, dim in enumerate(x.shape) if i != axis)
    res = ht.argmax(x._array, axis=axis, keepdim=True).reshape(output_shape)
    return Array._new(res)


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Returns the indices of the minimum values along a specified axis. When the
    minimum value occurs multiple times, only the indices corresponding to the
    first occurrence are returned.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    axis : Optional[int]
        Axis along which to search. If ``None``, the function returns the index of
        the minimum value of the flattened array. Default: ``None``.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes (dimensions)
        are not be included in the result. Default: ``False``.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in argmin")
    if axis is None:
        if keepdims:
            output_shape = tuple(1 for _ in range(x.ndim))
        else:
            output_shape = ()
    else:
        if axis < 0:
            axis += x.ndim
        if keepdims:
            output_shape = tuple(dim if i != axis else 1 for i, dim in enumerate(x.shape))
        else:
            output_shape = tuple(dim for i, dim in enumerate(x.shape) if i != axis)
    res = ht.argmin(x._array, axis=axis, keepdim=True).reshape(output_shape)
    return Array._new(res)
