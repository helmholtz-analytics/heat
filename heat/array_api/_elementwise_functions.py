from __future__ import annotations

from ._dtypes import (
    _numeric_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _result_type,
)
from ._array_object import Array

import heat as ht


def abs(x: Array, /) -> Array:
    """
    Calculates the absolute value for each element ``x_i`` of the input array ``x``
    (i.e., the element-wise result has the same magnitude as the respective
    element in ``x`` but has positive sign).

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in abs")
    return Array._new(ht.abs(x._array, dtype=x.dtype))


def add(x1: Array, x2: Array, /) -> Array:
    """
    Calculates the sum for each element ``x1_i`` of the input array ``x1`` with
    the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.add(x1._array, x2._array))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    """
    Computes the bitwise AND of the underlying binary representation of each
    element ``x1_i`` of the input array ``x1`` with the respective element
    ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have an integer or boolean data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have an integer
        or boolean data type.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.bitwise_and(x1._array, x2._array))


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    """
    Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the
    left by appending ``x2_i`` (i.e., the respective element in the input array
    ``x2``) zeros to the right of ``x1_i``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have an integer data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have an integer
        data type. Each element must be greater than or equal to ``0``.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_left_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    # Note: bitwise_left_shift is only defined for x2 nonnegative.
    if ht.any(x2._array < 0):
        raise ValueError("bitwise_left_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(ht.left_shift(x1._array, x2._array))


def bitwise_invert(x: Array, /) -> Array:
    """
    Inverts (flips) each bit for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have an integer or boolean data type.
    """
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_invert")
    return Array._new(ht.invert(x._array))


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    """
    Computes the bitwise OR of the underlying binary representation of each
    element ``x1_i`` of the input array ``x1`` with the respective element
    ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have an integer or boolean data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have an integer
        or boolean data type.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.bitwise_or(x1._array, x2._array))


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    """
    Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the
    right according to the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have an integer data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have an integer
        data type. Each element must be greater than or equal to ``0``.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_right_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    # Note: bitwise_right_shift is only defined for x2 nonnegative.
    if ht.any(x2._array < 0):
        raise ValueError("bitwise_right_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(ht.right_shift(x1._array, x2._array))


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    """
    Computes the bitwise XOR of the underlying binary representation of each element
    ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the
    input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have an integer or boolean data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have an integer
        or boolean data type.
    """
    if x1.dtype not in _integer_or_boolean_dtypes or x2.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.bitwise_xor(x1._array, x2._array))


def equal(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of ``x1_i == x2_i`` for each element ``x1_i`` of
    the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array.
    x2 : Array
        Second input array. Must be compatible with ``x1``.
    """
    return Array._new(ht.eq(x1._array, x2._array))


def isfinite(x: Array, /) -> Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine if finite
    (i.e., not ``NaN`` and not equal to positive or negative infinity).

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isfinite")
    return Array._new(ht.isfinite(x._array))


def isinf(x: Array, /) -> Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine if equal
    to positive or negative infinity.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isinf")
    return Array._new(ht.isinf(x._array))


def isnan(x: Array, /) -> Array:
    """
    Tests each element ``x_i`` of the input array ``x`` to determine whether
    the element is ``NaN``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isnan")
    return Array._new(ht.isnan(x._array))


def less(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of ``x1_i < x2_i`` for each element ``x1_i`` of
    the input array ``x1`` with the respective element ``x2_i`` of the input
    array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in less")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.less(x1._array, x2._array))
