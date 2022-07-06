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


def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """
    Returns the indices of the array elements which are non-zero.

    Parameters
    ----------
    x : Array
        Input array. Must have a positive rank.
    """
    # Fixed in PR #937, waiting for merge
    return ht.nonzero(x._array)


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    """
    Returns elements chosen from ``x1`` or ``x2`` depending on ``condition``.

    Parameters
    ----------
    condition : Array
        When ``True``, yield ``x1_i``; otherwise, yield ``x2_i``. Must be
        compatible with ``x1`` and ``x2``.
    x1 : Array
        First input array. Must be compatible with ``condition`` and ``x2``.
    x2 : Array
        Second input array. Must be compatible with ``condition`` and ``x1``.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.where(condition._array, x1._array, x2._array))
