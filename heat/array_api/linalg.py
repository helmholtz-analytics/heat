"""
Linear Algebra Extension for the Array API standard.
"""
from __future__ import annotations

from ._dtypes import _numeric_dtypes, _result_type
from ._array_object import Array

import heat as ht


def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Computes the matrix product.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type and at least one dimension.
    x2 : Array
        Second input array. Must have a numeric data type and at least one dimension.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in matmul")
    return Array._new(ht.matmul(x1._array, x2._array))


def matrix_transpose(x: Array, /) -> Array:
    """
    Transposes a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x : Array
        Input array having shape ``(..., M, N)`` and whose innermost two
        dimensions form ``MxN`` matrices.
    """
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return Array._new(ht.swapaxes(x._array, -1, -2))
    # axes = list(range(x.ndim))
    # axes[-1], axes[-2] = axes[-2], axes[-1]
    # return Array._new(ht.transpose(x._array, axes))


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    """
    Computes the (vector) dot product of two arrays.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a numeric data type.
    axis : int
        Axis over which to compute the dot product. Must be an integer on the interval
        ``[-N, N)``, where ``N`` is the rank (number of dimensions) of the shape
        determined according to Broadcasting. If specified as a negative integer, the
        function determines the axis along which to compute the dot product by counting
        backward from the last dimension (where ``-1`` refers to the last dimension).
        By default, the function computes the dot product over the last axis.
        Default: ``-1``.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in vecdot")
    res = ht.vecdot(x1._array, x2._array, axis=axis)
    return Array._new(res.astype(_result_type(x1.dtype, x2.dtype)))
