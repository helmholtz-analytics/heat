"""
Linear Algebra Extension for the Array API standard.
"""
from __future__ import annotations

from ._dtypes import _numeric_dtypes, _result_type
from ._array_object import Array

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Sequence, Tuple, Union

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


def tensordot(
    x1: Array, x2: Array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2
) -> Array:
    """
    Return a tensor contraction of ``x1`` and ``x2`` over specific axes.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must have a numeric data type. Corresponding contracted axes of ``x1``
        and ``x2`` must be equal.
    axes : Union[int, Tuple[Sequence[int], Sequence[int]]]
        Number of axes (dimensions) to contract or explicit sequences of axes (dimensions) for
        ``x1`` and ``x2``, respectively. If ``axes`` is an ``int`` equal to ``N``, then contraction is
        performed over the last ``N`` axes of ``x1`` and the first ``N`` axes of ``x2`` in order.
        The size of each corresponding axis (dimension) must match. Must be nonnegative.
        If ``axes`` is a tuple of two sequences ``(x1_axes, x2_axes)``, the first sequence must apply
        to ``x1`` and the second sequence to ``x2``. Both sequences must have the same length.
        Each axis (dimension) ``x1_axes[i]`` for ``x1`` must have the same size as the respective axis
        (dimension) ``x2_axes[i]`` for ``x2``. Each sequence must consist of unique (nonnegative)
        integers that specify valid axes for each respective array.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in tensordot")
    return Array._new(ht.tensordot(x1._array, x2._array, axes=axes))


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
