"""
Linear Algebra Extension for the Array API standard.
"""
from __future__ import annotations

from ._dtypes import _numeric_dtypes
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
