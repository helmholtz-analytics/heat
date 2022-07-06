from __future__ import annotations

from ._dtypes import (
    _numeric_dtypes,
    _boolean_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _floating_dtypes,
    _result_type,
)
from ._array_object import Array
from ._data_type_functions import astype

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


def acos(x: Array, /) -> Array:
    """
    Calculates an approximation of the principal value of the inverse cosine,
    having domain ``[-1, +1]`` and codomain ``[+0, +π]``, for each element ``x_i``
    of the input array ``x``. Each element-wise result is expressed in radians.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acos")
    return Array._new(ht.acos(x._array))


def acosh(x: Array, /) -> Array:
    """
    Calculates an approximation to the inverse hyperbolic cosine, having domain
    ``[+1, +infinity]`` and codomain ``[+0, +infinity]``, for each element ``x_i``
    of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array whose elements each represent the area of a hyperbolic sector.
        Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acosh")
    return Array._new(ht.acosh(x._array))


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


def asin(x: Array, /) -> Array:
    """
    Calculates an approximation of the principal value of the inverse sine, having
    domain ``[-1, +1]`` and codomain ``[-π/2, +π/2]`` for each element ``x_i`` of
    the input array ``x``. Each element-wise result is expressed in radians.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asin")
    return Array._new(ht.asin(x._array))


def asinh(x: Array, /) -> Array:
    """
    Calculates an approximation to the inverse hyperbolic sine, having domain
    ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``, for each
    element ``x_i`` in the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array whose elements each represent the area of a hyperbolic sector.
        Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asinh")
    return Array._new(ht.asinh(x._array))


def atan(x: Array, /) -> Array:
    """
    Calculates an implementation-dependent approximation of the principal value of
    the inverse tangent, having domain ``[-infinity, +infinity]`` and codomain
    ``[-π/2, +π/2]``, for each element ``x_i`` of the input array ``x``.
    Each element-wise result is expressed in radians.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan")
    return Array._new(ht.atan(x._array))


def atan2(x1: Array, x2: Array, /) -> Array:
    """
    Calculates an approximation of the inverse tangent of the quotient ``x1/x2``,
    having domain ``[-infinity, +infinity] x [-infinity, +infinity]`` (where the
    ``x`` notation denotes the set of ordered pairs of elements ``(x1_i, x2_i)``)
    and codomain ``[-π, +π]``, for each pair of elements ``(x1_i, x2_i)`` of the
    input arrays ``x1`` and ``x2``, respectively. Each element-wise result is
    expressed in radians.

    The mathematical signs of ``x1_i`` and ``x2_i`` determine the quadrant of each
    element-wise result. The quadrant (i.e., branch) is chosen such that each
    element-wise result is the signed angle in radians between the ray ending at the
    origin and passing through the point ``(1,0)`` and the ray ending at the origin
    and passing through the point ``(x2_i, x1_i)``.

    Parameters
    ----------
    x1 : Array
        Input array corresponding to the y-coordinates. Must have a floating-point
        data type.
    x2 : Array
        Input array corresponding to the x-coordinates. Must be compatible with ``x1``
        and have a floating-point data type.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan2")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.atan2(x1._array, x2._array))


def atanh(x: Array, /) -> Array:
    """
    Calculates an approximation to the inverse hyperbolic tangent, having domain
    ``[-1, +1]`` and codomain ``[-infinity, +infinity]``, for each element ``x_i``
    of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array whose elements each represent the area of a hyperbolic sector.
        Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atanh")
    return Array._new(ht.atanh(x._array))


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


def ceil(x: Array, /) -> Array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the smallest (i.e., closest
    to ``-infinity``) integer-valued number that is not less than ``x_i``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in ceil")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of ceil is the same as the input
        return x
    return Array._new(ht.ceil(x._array))


def cos(x: Array, /) -> Array:
    """
    Calculates an approximation to the cosine, having domain ``(-infinity, +infinity)``
    and codomain ``[-1, +1]``, for each element ``x_i`` of the input array ``x``.
    Each element ``x_i`` is assumed to be expressed in radians.

    Parameters
    ----------
    x : Array
        Input array whose elements are each expressed in radians. Must have a
        floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cos")
    return Array._new(ht.cos(x._array))


def cosh(x: Array, /) -> Array:
    """
    Calculates an approximation to the hyperbolic cosine, having domain
    ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``, for each
    element ``x_i`` in the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array whose elements each represent a hyperbolic angle. Must have
        a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cosh")
    return Array._new(ht.cosh(x._array))


def divide(x1: Array, x2: Array, /) -> Array:
    """
    Calculates the division for each element ``x1_i`` of the input array ``x1``
    with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        Dividend input array. Must have a numeric data type.
    x2 : Array
        Divisor input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.divide(x1._array, x2._array))


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
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.eq(x1._array, x2._array))


def exp(x: Array, /) -> Array:
    """
    Calculates an approximation to the exponential function, having domain
    ``[-infinity, +infinity]`` and codomain ``[+0, +infinity]``, for each element
    ``x_i`` of the input array ``x`` (``e`` raised to the power of ``x_i``, where
    ``e`` is the base of the natural logarithm).

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return Array._new(ht.exp(x._array))


def expm1(x: Array, /) -> Array:
    """
    Calculates an approximation to ``exp(x)-1``, having domain
    ``[-infinity, +infinity]`` and codomain ``[-1, +infinity]``, for each element
    ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in expm1")
    return Array._new(ht.expm1(x._array))


def floor(x: Array, /) -> Array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the greatest
    (i.e., closest to ``+infinity``) integer-valued number that is not greater
    than ``x_i``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of floor is the same as the input
        return x
    return Array._new(ht.floor(x._array))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    """
    Rounds the result of dividing each element ``x1_i`` of the input array ``x1``
    by the respective element ``x2_i`` of the input array ``x2`` to the greatest
    (i.e., closest to ``+infinity``) integer-value number that is not greater than
    the division result.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor_divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.floor_divide(x1._array, x2._array))


def greater(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of ``x1_i > x2_i`` for each element ``x1_i`` of
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
        raise TypeError("Only numeric dtypes are allowed in greater")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.greater(x1._array, x2._array))


def greater_equal(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of ``x1_i >= x2_i`` for each element ``x1_i`` of
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
        raise TypeError("Only numeric dtypes are allowed in greater_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.greater_equal(x1._array, x2._array))


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


def less_equal(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of ``x1_i <= x2_i`` for each element ``x1_i``
    of the input array ``x1`` with the respective element ``x2_i`` of the
    input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in less_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.less_equal(x1._array, x2._array))


def log(x: Array, /) -> Array:
    """
    Calculates an  approximation to the natural (base ``e``) logarithm, having domain
    ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element
    ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(ht.log(x._array))


def log1p(x: Array, /) -> Array:
    """
    Calculates an approximation to ``log(1+x)``, where ``log`` refers to the natural
    (base ``e``) logarithm, having domain ``[-1, +infinity]`` and codomain
    ``[-infinity, +infinity]``, for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log1p")
    return Array._new(ht.log1p(x._array))


def log2(x: Array, /) -> Array:
    """
    Calculates an approximation to the base ``2`` logarithm, having domain
    ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element
    ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log2")
    return Array._new(ht.log2(x._array))


def log10(x: Array, /) -> Array:
    """
    Calculates an approximation to the base ``10`` logarithm, having domain
    ``[0, +infinity]`` and codomain ``[-infinity, +infinity]``, for each element
    ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log10")
    return Array._new(ht.log10(x._array))


def logaddexp(x1: Array, x2: Array) -> Array:
    """
    Calculates the logarithm of the sum of exponentiations ``log(exp(x1) + exp(x2))``
    for each element ``x1_i`` of the input array ``x1`` with the respective element
    ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a floating-point data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a floating-point
        data type.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in logaddexp")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.logaddexp(x1._array, x2._array))


def logical_and(x1: Array, x2: Array, /) -> Array:
    """
    Computes the logical AND for each element ``x1_i`` of the input array ``x1``
    with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a boolean data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a boolean
        data type.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.logical_and(x1._array, x2._array))


def logical_not(x: Array, /) -> Array:
    """
    Computes the logical NOT for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a boolean data type.
    """
    if x.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_not")
    return Array._new(ht.logical_not(x._array))


def logical_or(x1: Array, x2: Array, /) -> Array:
    """
    Computes the logical OR for each element ``x1_i`` of the input array ``x1``
    with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a boolean data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a boolean
        data type.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.logical_or(x1._array, x2._array))


def logical_xor(x1: Array, x2: Array, /) -> Array:
    """
    Computes the logical XOR for each element ``x1_i`` of the input array ``x1``
    with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a boolean data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a boolean
        data type.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.logical_xor(x1._array, x2._array))


def multiply(x1: Array, x2: Array, /) -> Array:
    """
    Calculates the product for each element ``x1_i`` of the input array ``x1``
    with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in multiply")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.multiply(x1._array, x2._array))


def negative(x: Array, /) -> Array:
    """
    Computes the numerical negative of each element ``x_i``
    (i.e., ``y_i = -x_i``) of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in negative")
    return Array._new(ht.negative(x._array))


def not_equal(x1: Array, x2: Array, /) -> Array:
    """
    Computes the truth value of ``x1_i != x2_i`` for each element ``x1_i``
    of the input array ``x1`` with the respective element ``x2_i`` of the
    input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array.
    x2 : Array
        Second input array. Must be compatible with ``x1``.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.not_equal(x1._array, x2._array))


def positive(x: Array, /) -> Array:
    """
    Computes the numerical positive of each element ``x_i``
    (i.e., ``y_i = +x_i``) of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in positive")
    return Array._new(ht.positive(x._array))


def pow(x1: Array, x2: Array, /) -> Array:
    """
    Calculates an approximation of exponentiation by raising each element
    ``x1_i`` (the base) of the input array ``x1`` to the power of
    ``x2_i`` (the exponent), where ``x2_i`` is the corresponding element of
    the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array whose elements correspond to the exponentiation base.
        Must have a numeric data type.
    x2 : Array
        Second input array whose elements correspond to the exponentiation exponent.
        Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in pow")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.pow(x1._array, x2._array))


def remainder(x1: Array, x2: Array, /) -> Array:
    """
    Returns the remainder of division for each element ``x1_i`` of the input
    array ``x1`` and the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        Dividend input array. Must have a numeric data type.
    x2 : Array
        Divisor input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in remainder")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.remainder(x1._array, x2._array))


def round(x: Array, /) -> Array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the nearest
    integer-valued number.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in round")
    elif x.dtype in _integer_dtypes:
        return x
    return Array._new(ht.round(x._array))


def sign(x: Array, /) -> Array:
    """
    Returns an indication of the sign of a number for each element ``x_i`` of
    the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return astype(Array._new(ht.sign(x._array)), x.dtype)


def sin(x: Array, /) -> Array:
    """
    Calculates an approximation to the sine, having domain ``(-infinity, +infinity)``
    and codomain ``[-1, +1]``, for each element ``x_i`` of the input array ``x``.
    Each element ``x_i`` is assumed to be expressed in radians.

    Parameters
    ----------
    x : Array
        Input array whose elements are each expressed in radians. Must have a
        floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sin")
    return Array._new(ht.sin(x._array))


def sinh(x: Array, /) -> Array:
    """
    Calculates an approximation to the hyperbolic sine, having domain
    ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``, for
    each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array whose elements each represent a hyperbolic angle. Must have
        a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sinh")
    return Array._new(ht.sinh(x._array))


def square(x: Array, /) -> Array:
    """
    Squares ``(x_i * x_i)`` each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in square")
    return astype(Array._new(ht.square(x._array)), x.dtype)


def sqrt(x: Array, /) -> Array:
    """
    Calculates the square root, having domain ``[0, +infinity]`` and codomain
    ``[0, +infinity]``, for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrt")
    return Array._new(ht.sqrt(x._array))


def subtract(x1: Array, x2: Array, /) -> Array:
    """
    Calculates the difference for each element ``x1_i`` of the input array
    ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1 : Array
        First input array. Must have a numeric data type.
    x2 : Array
        Second input array. Must be compatible with ``x1`` and have a numeric
        data type.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(ht.subtract(x1._array, x2._array))


def tan(x: Array, /) -> Array:
    """
    Calculates an approximation to the tangent, having domain ``(-infinity, +infinity)``
    and codomain ``(-infinity, +infinity)``, for each element ``x_i`` of the
    input array ``x``. Each element ``x_i`` is assumed to be expressed in radians.

    Parameters
    ----------
    x : Array
        Input array whose elements are each expressed in radians. Must have a
        floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tan")
    return Array._new(ht.tan(x._array))


def tanh(x: Array, /) -> Array:
    """
    Calculates an approximation to the hyperbolic tangent, having domain
    ``[-infinity, +infinity]`` and codomain ``[-1, +1]``, for each element ``x_i``
    of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array whose elements each represent a hyperbolic angle. Must have
        a floating-point data type.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tanh")
    return Array._new(ht.tanh(x._array))
