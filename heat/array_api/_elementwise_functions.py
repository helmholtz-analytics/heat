from __future__ import annotations

import heat as ht

from ._array_object import Array
from ._dtypes import _numeric_dtypes, _result_type


def equal(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of `x1_i == x2_i` for each element `x1_i` of
    the input array `x1` with the respective element `x2_i` of the input
    array `x2`.
    """
    # Call result type here just to raise on disallowed type combinations
    # _result_type(x1.dtype, x2.dtype)
    # x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.eq(x1._array, x2._array))


def isfinite(x: Array, /) -> Array:
    """
    Tests each element `x_i` of the input array `x` to determine if finite
    (i.e., not `NaN` and not equal to positive or negative infinity).
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isfinite")
    return Array._new(ht.isfinite(x._array))


def isinf(x: Array, /) -> Array:
    """
    Tests each element `x_i` of the input array `x` to determine if equal
    to positive or negative infinity.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isinf")
    return Array._new(ht.isinf(x._array))


def isnan(x: Array, /) -> Array:
    """
    Tests each element `x_i` of the input array `x` to determine whether
    the element is `NaN`.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isnan")
    return Array._new(ht.isnan(x._array))
