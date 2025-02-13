"""
Arithmetic functions for DNDarrays
"""

from __future__ import annotations

import torch
from typing import Optional, Union, Tuple

from . import factories
from . import manipulations
from . import _operations
from . import sanitation
from . import stride_tricks
from . import types
from . import logical

from .communication import MPI
from .dndarray import DNDarray
from .types import (
    canonical_heat_type,
    heat_type_is_inexact,
    heat_type_is_exact,
    heat_type_of,
    datatype,
    can_cast,
    _complexfloating,
)


__all__ = [
    "add",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "copysign",
    "cumprod",
    "cumproduct",
    "cumsum",
    "diff",
    "div",
    "divide",
    "divmod",
    "floordiv",
    "floor_divide",
    "fmod",
    "gcd",
    "hypot",
    "invert",
    "lcm",
    "left_shift",
    "mod",
    "mul",
    "multiply",
    "nan_to_num",
    "nanprod",
    "nansum",
    "neg",
    "negative",
    "pos",
    "positive",
    "pow",
    "power",
    "prod",
    "remainder",
    "right_shift",
    "sub",
    "subtract",
    "sum",
]


def add(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise addition of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose
    elements are to be added as argument and returns a ``DNDarray`` containing the results of
    element-wise addition of ``t1`` and ``t2``.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the addition
    t2: DNDarray or scalar
        The second operand involved in the addition
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the added value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Examples
    --------
    >>> import heat as ht
    >>> ht.add(1.0, 4.0)
    DNDarray(5., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.add(T1, T2)
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.add(T1, s)
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.add, t1, t2, out, where)


def _add(self, other):
    try:
        return add(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__add__ = _add
DNDarray.__add__.__doc__ = add.__doc__
DNDarray.__radd__ = lambda self, other: _add(other, self)
DNDarray.__radd__.__doc__ = add.__doc__


def add_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise in-place addition of values of two operands.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise adds the
    element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
    i.e. the element(s) of `t1` are overwritten by the results of element-wise addition of `t1` and
    `t2`.
    Can be called as a DNDarray method or with the symbol `+=`.

    Parameters
    ----------
    t1: DNDarray
        The first operand involved in the addition
    t2: DNDarray or scalar
        The second operand involved in the addition

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> T1 += T2
    >>> T1
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[2., 2.],
              [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> T2.add_(s)
    DNDarray([[4., 4.],
              [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[4., 4.],
              [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s
    2.0
    """

    def wrap_add_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.add_(b)

    try:
        return _operations.__binary_op(wrap_add_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__iadd__ = add_
DNDarray.add_ = add_


def bitwise_and(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Compute the bitwise AND of two :class:`~heat.core.dndarray.DNDarray` ``t1`` and ``t2``
    element-wise. Only integer and boolean types are handled. If ``t1.shape!=t2.shape``, they must
    be broadcastable to a common shape (which becomes the shape of the output)

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the added value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.bitwise_and(13, 17)
    DNDarray(1, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(14, 13)
    DNDarray(12, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([14,3]), 13)
    DNDarray([12,  1], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([11,7]), ht.array([4,25]))
    DNDarray([0, 1], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([2,5,255]), ht.array([3,14,16]))
    DNDarray([ 2,  4, 16], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([True, True]), ht.array([False, True]))
    DNDarray([False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if heat_type_is_inexact(dt):
            raise TypeError("Operation is not supported for float types")

    return _operations.__binary_op(torch.bitwise_and, t1, t2, out, where)


def _and(self, other):
    try:
        return bitwise_and(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__and__ = _and
DNDarray.__and__.__doc__ = bitwise_and.__doc__
DNDarray.__rand__ = lambda self, other: _and(other, self)
DNDarray.__rand__.__doc__ = bitwise_and.__doc__


def bitwise_and_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Bitwise AND of two operands computed element-wise and in-place.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
    bitwise AND with the corresponding element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise bitwise AND of `t1` and `t2`.
    Can be called as a DNDarray method or with the symbol `&=`. Only integer and boolean types are
    handled.

    Parameters
    ----------
    t1: DNDarray
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array(13)
    >>> T2 = ht.array(17)
    >>> T1 &= T2
    >>> T1
    DNDarray(1, dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray(17, dtype=ht.int64, device=cpu:0, split=None)
    >>> T3 = ht.array(22)
    >>> T2.bitwise_and_(T3)
    DNDarray(16, dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray(16, dtype=ht.int64, device=cpu:0, split=None)
    >>> T4 = ht.array([14,3])
    >>> s = 29
    >>> T4 &= s
    >>> T4
    DNDarray([12,  1], dtype=ht.int64, device=cpu:0, split=None)
    >>> s
    29
    >>> T5 = ht.array([2,5,255])
    >>> T6 = ht.array([3,14,16])
    >>> T5 &= T6
    >>> T5
    DNDarray([ 2,  4, 16], dtype=ht.int64, device=cpu:0, split=None)
    >>> T7 = ht.array([True, True])
    >>> T8 = ht.array([False, True])
    >>> T7 &= T8
    >>> T7
    DNDarray([False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if heat_type_is_inexact(dt):
            raise TypeError("Operation is not supported for float types.")

    def wrap_bitwise_and_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.bitwise_and_(b)

    try:
        return _operations.__binary_op(wrap_bitwise_and_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__iand__ = bitwise_and_
DNDarray.bitwise_and_ = bitwise_and_


def bitwise_or(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Compute the bit-wise OR of two :class:`~heat.core.dndarray.DNDarray` ``t1`` and ``t2``
    element-wise. Only integer and boolean types are handled. If ``t1.shape!=t2.shape``, they must
    be broadcastable to a common shape (which becomes the shape of the output)

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the added value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.bitwise_or(13, 16)
    DNDarray(29, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(32, 2)
    DNDarray(34, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([33, 4]), 1)
    DNDarray([33,  5], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([33, 4]), ht.array([1, 2]))
    DNDarray([33,  6], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([2, 5, 255]), ht.array([4, 4, 4]))
    DNDarray([  6,   5, 255], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([2, 5, 255, 2147483647], dtype=ht.int32),
                      ht.array([4, 4, 4, 2147483647], dtype=ht.int32))
    DNDarray([         6,          5,        255, 2147483647], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([True, True]), ht.array([False, True]))
    DNDarray([True, True], dtype=ht.bool, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if heat_type_is_inexact(dt):
            raise TypeError("Operation is not supported for float types")

    return _operations.__binary_op(torch.bitwise_or, t1, t2, out, where)


def _or(self, other):
    try:
        return bitwise_or(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__or__ = _or
DNDarray.__or__.__doc__ = bitwise_or.__doc__
DNDarray.__ror__ = lambda self, other: _or(other, self)
DNDarray.__ror__.__doc__ = bitwise_or.__doc__


def bitwise_or_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Bitwise OR of two operands computed element-wise and in-place.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
    bitwise OR with the corresponding element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise bitwise OR of `t1` and `t2`.
    Can be called as a DNDarray method or with the symbol `|=`. Only integer and boolean types are
    handled.

    Parameters
    ----------
    t1: DNDarray
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array(13)
    >>> T2 = ht.array(16)
    >>> T1 |= T2
    >>> T1
    DNDarray(29, dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray(16, dtype=ht.int64, device=cpu:0, split=None)
    >>> T3 = ht.array([33, 4])
    >>> s = 1
    >>> T3.bitwise_or_(s)
    DNDarray([33,  5], dtype=ht.int64, device=cpu:0, split=None)
    >>> T3
    DNDarray([33,  5], dtype=ht.int64, device=cpu:0, split=None)
    >>> s
    1
    >>> T4 = ht.array([2,5,255])
    >>> T5 = ht.array([4, 4, 4])
    >>> T4 |= T5
    >>> T4
    DNDarray([  6,   5, 255], dtype=ht.int64, device=cpu:0, split=None)
    >>> T6 = ht.array([True, True])
    >>> T7 = ht.array([False, True])
    >>> T6 |= T7
    >>> T6
    DNDarray([True, True], dtype=ht.bool, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if heat_type_is_inexact(dt):
            raise TypeError("Operation is not supported for float types.")

    def wrap_bitwise_or_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.bitwise_or_(b)

    try:
        return _operations.__binary_op(wrap_bitwise_or_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__ior__ = bitwise_or_
DNDarray.bitwise_or_ = bitwise_or_


def bitwise_xor(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Compute the bit-wise XOR of two arrays ``t1`` and ``t2`` element-wise.
    Only integer and boolean types are handled. If ``x1.shape!=x2.shape``, they must be
    broadcastable to a common shape (which becomes the shape of the output).

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the added value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.bitwise_xor(13, 17)
    DNDarray(28, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(31, 5)
    DNDarray(26, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(ht.array([31,3]), 5)
    DNDarray([26,  6], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(ht.array([31,3]), ht.array([5,6]))
    DNDarray([26,  5], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(ht.array([True, True]), ht.array([False, True]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if heat_type_is_inexact(dt):
            raise TypeError("Operation is not supported for float types")

    return _operations.__binary_op(torch.bitwise_xor, t1, t2, out, where)


def _xor(self, other):
    try:
        return bitwise_xor(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__xor__ = _xor
DNDarray.__xor__.__doc__ = bitwise_xor.__doc__
DNDarray.__rxor__ = lambda self, other: _xor(other, self)
DNDarray.__rxor__.__doc__ = bitwise_xor.__doc__


def bitwise_xor_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Bitwise XOR of two operands computed element-wise and in-place.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
    bitwise XOR with the corresponding element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise bitwise XOR of `t1` and `t2`.
    Can be called as a DNDarray method or with the symbol `^=`. Only integer and boolean types are
    handled.

    Parameters
    ----------
    t1: DNDarray
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array(13)
    >>> T2 = ht.array(17)
    >>> T1 ^= T2
    >>> T1
    DNDarray(28, dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray(17, dtype=ht.int64, device=cpu:0, split=None)
    >>> T3 = ht.array([31, 3])
    >>> s = 5
    >>> T3.bitwise_xor_(s)
    DNDarray([26,  6], dtype=ht.int64, device=cpu:0, split=None)
    >>> T3
    DNDarray([26,  6], dtype=ht.int64, device=cpu:0, split=None)
    >>> s
    5
    >>> T4 = ht.array([31,3,255])
    >>> T5 = ht.array([5, 6, 4])
    >>> T4 ^= T5
    >>> T4
    DNDarray([ 26,   5, 251], dtype=ht.int64, device=cpu:0, split=None)
    >>> T6 = ht.array([True, True])
    >>> T7 = ht.array([False, True])
    >>> T6 ^= T7
    >>> T6
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if heat_type_is_inexact(dt):
            raise TypeError("Operation is not supported for float types.")

    def wrap_bitwise_xor_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.bitwise_xor_(b)

    try:
        return _operations.__binary_op(wrap_bitwise_xor_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__ixor__ = bitwise_xor_
DNDarray.bitwise_xor_ = bitwise_xor_


def copysign(
    a: DNDarray,
    b: Union[DNDarray, float, int],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Create a new floating-point tensor with the magnitude of 'a' and the sign of 'b', element-wise

    Parameters
    ----------
    a:  DNDarray
        The input array
    b:  DNDarray or Number
        value(s) whose signbit(s) are applied to the magnitudes in 'a'
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.copysign(ht.array([3, 2, -8, -2, 4]), 1)
    DNDarray([3, 2, 8, 2, 4], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.copysign(ht.array([3., 2., -8., -2., 4.]), ht.array([1., -1., 1., -1., 1.]))
    DNDarray([ 3., -2.,  8., -2.,  4.], dtype=ht.float32, device=cpu:0, split=None)
    """
    try:
        res = _operations.__binary_op(torch.copysign, a, b, out, where)
    except RuntimeError:
        # every other possibility is caught by __binary_op
        raise TypeError(f"Not implemented for input type, got {type(a)}, {type(b)}")

    return res


def copysign_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    In-place version of the element-wise operation 'copysign'.
    The magnitudes of the element(s) of 't1' are kept but the sign(s) are adopted from the
    element(s) of 't2'.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t1:    DNDarray
           The input array
           Entries must be of type float.
    t2:    DNDarray or scalar
           value(s) whose signbit(s) are applied to the magnitudes in 't1'

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        At the moment, the operation only works for DNDarrays whose elements are floats and are not
        complex. This is due to the fact that it relies on the PyTorch function 'copysign_', which
        does not work if the entries of 't1' are integers. The case when 't1' contains floats and
        't2' contains integers works in PyTorch but has not been implemented properly in Heat yet.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array([3., 2., -8., -2., 4.])
    >>> s = 2.0
    >>> T1.copysign_(s)
    DNDarray([3., 2., 8., 2., 4.], dtype=ht.float32, device=cpu:0, split=None)
    >>> T1
    DNDarray([3., 2., 8., 2., 4.], dtype=ht.float32, device=cpu:0, split=None)
    >>> s
    2.0
    >>> T2 = ht.array([[1., -1.],[1., -1.]])
    >>> T3 = ht.array([-5., 2.])
    >>> T2.copysign_(T3)
    DNDarray([[-1.,  1.],
              [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[-1.,  1.],
              [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T3
    DNDarray([-5.,  2.], dtype=ht.float32, device=cpu:0, split=None)
    """
    dtypes = dtype1, dtype2 = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if heat_type_is_exact(dt) or dt in _complexfloating:
            raise TypeError(
                "Operation is only supported for inputs whose elements are floats and are not "
                + f"complex. But your inputs have the datatypes {dtype1} and {dtype2}."
            )

    def wrap_copysign_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.copysign_(b)

    try:
        return _operations.__binary_op(wrap_copysign_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.copysign_ = copysign_


def cumprod(a: DNDarray, axis: int, dtype: datatype = None, out=None) -> DNDarray:
    """
    Return the cumulative product of elements along a given axis.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : int
        Axis along which the cumulative product is computed.
    dtype : datatype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If ``dtype`` is not specified, it
        defaults to the datatype of ``a``, unless ``a`` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Examples
    --------
    >>> a = ht.full((3,3), 2)
    >>> ht.cumprod(a, 0)
    DNDarray([[2., 2., 2.],
            [4., 4., 4.],
            [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__cum_op(a, torch.cumprod, MPI.PROD, torch.mul, 1, axis, dtype, out)


# Alias support
cumproduct = cumprod
"""Alias for :py:func:`cumprod`"""


def cumprod_(t: DNDarray, axis: int) -> DNDarray:
    """
    Return the cumulative product of elements along a given axis in-place.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t:      DNDarray
            Input array.
    axis:   int
            Axis along which the cumulative product is computed.

    Examples
    --------
    >>> import heat as ht
    >>> T = ht.full((3,3), 2)
    >>> T.cumprod_(0)
    DNDarray([[2., 2., 2.],
              [4., 4., 4.],
              [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T
    DNDarray([[2., 2., 2.],
              [4., 4., 4.],
              [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T.cumproduct_(1)
    DNDarray([[  2.,   4.,   8.],
              [  4.,  16.,  64.],
              [  8.,  64., 512.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T
    DNDarray([[  2.,   4.,   8.],
              [  4.,  16.,  64.],
              [  8.,  64., 512.]], dtype=ht.float32, device=cpu:0, split=None)
    """

    def wrap_cumprod_(a: torch.Tensor, b: int, out=None, dtype=None) -> torch.Tensor:
        return a.cumprod_(b)

    def wrap_mul_(a: torch.Tensor, b: torch.Tensor, out=None) -> torch.Tensor:
        return a.mul_(b)

    return _operations.__cum_op(t, wrap_cumprod_, MPI.PROD, wrap_mul_, 1, axis, dtype=None, out=t)


DNDarray.cumprod_ = DNDarray.cumproduct_ = cumprod_


def cumsum(a: DNDarray, axis: int, dtype: datatype = None, out=None) -> DNDarray:
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : int
        Axis along which the cumulative sum is computed.
    dtype : datatype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If ``dtype`` is not specified, it defaults
        to the datatype of ``a``, unless ``a`` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Examples
    --------
    >>> a = ht.ones((3,3))
    >>> ht.cumsum(a, 0)
    DNDarray([[1., 1., 1.],
              [2., 2., 2.],
              [3., 3., 3.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__cum_op(a, torch.cumsum, MPI.SUM, torch.add, 0, axis, dtype, out)


def cumsum_(t: DNDarray, axis: int) -> DNDarray:
    """
    Return the cumulative sum of the elements along a given axis in-place.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t:      DNDarray
            Input array.
    axis:   int
            Axis along which the cumulative sum is computed.

    Examples
    --------
    >>> import heat as ht
    >>> T = ht.ones((3,3))
    >>> T.cumsum_(0)
    DNDarray([[1., 1., 1.],
              [2., 2., 2.],
              [3., 3., 3.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T
    DNDarray([[1., 1., 1.],
              [2., 2., 2.],
              [3., 3., 3.]], dtype=ht.float32, device=cpu:0, split=None)
    """

    def wrap_cumsum_(a: torch.Tensor, b: int, out=None, dtype=None) -> torch.Tensor:
        return a.cumsum_(b)

    def wrap_add_(a: torch.Tensor, b: torch.Tensor, out=None) -> torch.Tensor:
        return a.add_(b)

    return _operations.__cum_op(t, wrap_cumsum_, MPI.SUM, wrap_add_, 0, axis, dtype=None, out=t)


DNDarray.cumsum_ = cumsum_


def diff(
    a: DNDarray,
    n: int = 1,
    axis: int = -1,
    prepend: Union[int, float, DNDarray] = None,
    append: Union[int, float, DNDarray] = None,
) -> DNDarray:
    """
    Calculate the n-th discrete difference along the given axis.
    The first difference is given by ``out[i]=a[i+1]-a[i]`` along the given axis, higher differences
    are calculated by using diff recursively. The shape of the output is the same as ``a`` except
    along axis where the dimension is smaller by ``n``. The datatype of the output is the same as
    the datatype of the difference between any two elements of ``a``. The split does not change. The
    output array is balanced.

    Parameters
    -------
    a : DNDarray
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
        ``n=2`` is equivalent to ``diff(diff(a))``
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    prepend : Union[int, float, DNDarray]
        Value to prepend along axis prior to performing the difference.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes. Otherwise the dimension and
        shape must match a except along axis.
    append : Union[int, float, DNDarray]
        Values to append along axis prior to performing the difference.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes. Otherwise the dimension and
        shape must match a except along axis.
    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError(f"diff requires that n be a positive number, got {n}")
    if not isinstance(a, DNDarray):
        raise TypeError("'a' must be a DNDarray")

    axis = stride_tricks.sanitize_axis(a.gshape, axis)

    if prepend is not None or append is not None:
        pend_shape = a.gshape[:axis] + (1,) + a.gshape[axis + 1 :]
        pend = [prepend, append]

        for p, p_el in enumerate(pend):
            if p_el is not None:
                if isinstance(p_el, (int, float)):
                    # TODO: implement broadcast_to
                    p_el = factories.full(
                        pend_shape,
                        p_el,
                        dtype=canonical_heat_type(torch.tensor(p_el).dtype),
                        split=a.split,
                        device=a.device,
                        comm=a.comm,
                    )
                elif isinstance(p_el, DNDarray) and p_el.gshape == pend_shape:
                    pass
                elif not isinstance(p_el, DNDarray):
                    raise TypeError(
                        f"prepend/append should be a scalar or a DNDarray, was {type(p_el)}"
                    )
                elif p_el.gshape != pend_shape:
                    raise ValueError(
                        f"shape mismatch: expected prepend/append to be {pend_shape}, got {p_el.gshape}"
                    )
                if p == 0:
                    # prepend
                    a = manipulations.concatenate((p_el, a), axis=axis)
                else:
                    # append
                    a = manipulations.concatenate((a, p_el), axis=axis)

    if not a.is_distributed():
        ret = a.copy()
        for _ in range(n):
            axis_slice = [slice(None)] * len(ret.shape)
            axis_slice[axis] = slice(1, None, None)
            axis_slice_end = [slice(None)] * len(ret.shape)
            axis_slice_end[axis] = slice(None, -1, None)
            ret = ret[tuple(axis_slice)] - ret[tuple(axis_slice_end)]
        return ret

    size = a.comm.size
    rank = a.comm.rank
    ret = a.copy()
    # work loop, runs n times. using the result at the end of the loop as the starting values for each loop
    for _ in range(n):
        axis_slice = [slice(None)] * len(ret.shape)
        axis_slice[axis] = slice(1, None, None)
        axis_slice_end = [slice(None)] * len(ret.shape)
        axis_slice_end[axis] = slice(None, -1, None)

        # build the slice for the first element on the specified axis
        arb_slice = [slice(None)] * len(a.shape)
        if ret.lshape[axis] > 0:
            arb_slice[axis] = 0
        # send the first element of the array to rank - 1
        if rank > 0:
            snd = ret.comm.Isend(ret.lloc[arb_slice].clone(), dest=rank - 1, tag=rank)

        # standard logic for the diff with the next element
        dif = ret.lloc[axis_slice] - ret.lloc[axis_slice_end]
        # need to slice out to select the proper elements of out
        diff_slice = [slice(x) for x in dif.shape]
        ret.lloc[diff_slice] = dif

        if rank > 0:
            snd.Wait()  # wait for the send to finish
        if rank < size - 1:
            cr_slice = [slice(None)] * len(a.shape)
            # slice of 1 element in the selected axis for the shape creation
            if ret.lshape[axis] > 1:
                cr_slice[axis] = 1
            recv_data = torch.ones(
                ret.lloc[cr_slice].shape, dtype=ret.dtype.torch_type(), device=a.device.torch_device
            )
            rec = ret.comm.Irecv(recv_data, source=rank + 1, tag=rank + 1)
            axis_slice_end = [slice(None)] * len(a.shape)
            # select the last elements in the selected axis
            axis_slice_end[axis] = slice(-1, None)
            rec.Wait()
            # diff logic
            ret.lloc[axis_slice_end] = (
                recv_data.reshape(ret.lloc[axis_slice_end].shape) - ret.lloc[axis_slice_end]
            )

    axis_slice_end = [slice(None, None, None)] * len(a.shape)
    axis_slice_end[axis] = slice(None, -1 * n, None)
    ret = ret[tuple(axis_slice_end)]  # slice off the last element on the array (nonsense data)
    ret.balance_()  # balance the array before returning
    return ret


def div(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise true division of values of operand ``t1`` by values of operands ``t2`` (i.e ``t1/t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided.
    t2: DNDarray or scalar
        The second operand by whose values is divided.
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the divided value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Example
    ---------
    >>> ht.div(2.0, 2.0)
    DNDarray(1., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.div(T1, T2)
    DNDarray([[0.5000, 1.0000],
              [1.5000, 2.0000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.div(s, T1)
    DNDarray([[2.0000, 1.0000],
              [0.6667, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.true_divide, t1, t2, out, where)


def _truediv(self, other):
    try:
        return div(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__truediv__ = _truediv
DNDarray.__truediv__.__doc__ = div.__doc__
DNDarray.__rtruediv__ = lambda self, other: _truediv(other, self)
DNDarray.__rtruediv__.__doc__ = div.__doc__

# Alias in compliance with numpy API
divide = div
"""Alias for :py:func:`div`"""


def div_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise in-place true division of values of two operands.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise divides its
    element(s) by the element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise division of `t1` and `t2`.
    Can be called as a DNDarray method or with the symbol `/=`. `divide_` is an alias for `div_`.

    Parameters
    ----------
    t1: DNDarray
        The first operand whose values are divided.
    t2: DNDarray or scalar
        The second operand by whose values is divided.

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Example
    ---------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> T1 /= T2
    >>> T1
    DNDarray([[0.5000, 1.0000],
              [1.5000, 2.0000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[2., 2.],
              [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> T2.div_(s)
    DNDarray([[1., 1.],
              [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[1., 1.],
              [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s
    2.0
    >>> v = ht.int32([-1, 2])
    >>> T2.divide_(v)
    DNDarray([[-1.0000,  0.5000],
              [-1.0000,  0.5000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[-1.0000,  0.5000],
              [-1.0000,  0.5000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> v
    DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)
    """

    def wrap_div_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.div_(b)

    try:
        return _operations.__binary_op(wrap_div_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__itruediv__ = div_
DNDarray.div_ = DNDarray.divide_ = div_


def divmod(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    out1: DNDarray = None,
    out2: DNDarray = None,
    /,
    out: Tuple[DNDarray, DNDarray] = (None, None),
    *,
    where: Union[bool, DNDarray] = True,
) -> Tuple[DNDarray, DNDarray]:
    """
    Element-wise division remainder and quotient from an integer division of values of operand
    ``t1`` by values of operand ``t2`` (i.e. C Library function divmod). Result has the sign as the
    dividend ``t1``. Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided (may be floats)
    t2: DNDarray or scalar
        The second operand by whose values is divided (may be floats)
    out1: DNDarray, optional
        The output array for the quotient. It must have a shape that the inputs broadcast to and
        matching split axis.
        If not provided, a freshly allocated array is returned. If provided, it must be of the same
        shape as the expected output. Only one of out1 and out can be provided.
    out2: DNDarray, optional
        The output array for the remainder. It must have a shape that the inputs broadcast to and
        matching split axis.
        If not provided, a freshly allocated array is returned. If provided, it must be of the same
        shape as the expected output. Only one of out2 and out can be provided.
    out: tuple of two DNDarrays, optional
        Tuple of two output arrays (quotient, remainder), respectively. Both must have a shape that
        the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned. If provided, they must be of the
        same shape as the expected output. out1 and out2 cannot be used at the same time.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out1`
        array will be set to the quotient value and the `out2` array will be set to the remainder
        value. Elsewhere, the `out1` and `out2` arrays will retain their original value. If an
        uninitialized `out1` and `out2` array is created via the default `out1=None` and
        `out2=None`, locations within them where the condition is False will remain uninitialized.
        If distributed, the split axis (after broadcasting if required) must match that of the
        `out1` and `out2` arrays.

    Examples
    --------
    >>> ht.divmod(2.0, 2.0)
    (DNDarray(1., dtype=ht.float32, device=cpu:0, split=None), DNDarray(0., dtype=ht.float32, device=cpu:0, split=None))
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.divmod(T1, T2)
    (DNDarray([[0., 1.],
               [1., 2.]], dtype=ht.float32, device=cpu:0, split=None), DNDarray([[1., 0.],
               [1., 0.]], dtype=ht.float32, device=cpu:0, split=None))
    >>> s = 2.0
    >>> ht.divmod(s, T1)
    (DNDarray([[2., 1.],
               [0., 0.]], dtype=ht.float32, device=cpu:0, split=None), DNDarray([[0., 0.],
               [2., 2.]], dtype=ht.float32, device=cpu:0, split=None))
    """
    if not isinstance(out, tuple):
        raise TypeError("out must be a tuple of two DNDarrays")
    if len(out) != 2:
        raise ValueError("out must be a tuple of two DNDarrays")
    if out[0] is not None:
        if out1 is None:
            out1 = out[0]
        else:
            raise TypeError("out[0] and out1 cannot be used at the same time")
    if out[1] is not None:
        if out2 is None:
            out2 = out[1]
        else:
            raise TypeError("out[1] and out2 cannot be used at the same time")

    # PyTorch has no divmod function
    d = floordiv(t1, t2, out1, where=where)
    m = mod(t1, t2, out2, where=where)

    return (d, m)


def _divmod(self, other):
    try:
        return divmod(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__divmod__ = _divmod
DNDarray.__divmod__.__doc__ = divmod.__doc__
DNDarray.__rdivmod__ = lambda self, other: _divmod(other, self)
DNDarray.__rdivmod__.__doc__ = divmod.__doc__


def floordiv(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise floor division of value(s) of operand ``t1`` by value(s) of operand ``t2``
    (i.e. ``t1//t2``), not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
    >>> ht.floordiv(T1, 1)
    DNDarray([[1., 2.],
              [1., 4.]], dtype=ht.float64, device=cpu:0, split=None)
    >>> T2 = ht.float32([1.5, 2.5])
    >>> ht.floordiv(T1, T2)
    DNDarray([[1., 0.],
              [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(
        torch.div, t1, t2, out, where, fn_kwargs={"rounding_mode": "floor"}
    )


def _floordiv(self, other):
    try:
        return floordiv(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__floordiv__ = _floordiv
DNDarray.__floordiv__.__doc__ = floordiv.__doc__
DNDarray.__rfloordiv__ = lambda self, other: _floordiv(other, self)
DNDarray.__rfloordiv__.__doc__ = floordiv.__doc__

# Alias in compliance with numpy API
floor_divide = floordiv
"""Alias for :py:func:`floordiv`"""


def floordiv_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise in-place floor division of values of two operands.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise divides its
    element(s) by the element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, then rounds down the result to the next
    integer, i.e. the element(s) of `t1` are overwritten by the results of element-wise floor
    division of `t1` and `t2`.
    Can be called as a DNDarray method or with the symbol `//=`. `floor_divide_` is an alias for
    `floordiv_`.

    Parameters
    ----------
    t1: DNDarray
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
    >>> s = 1
    >>> T1 //= s
    >>> T1
    DNDarray([[1., 2.],
              [1., 4.]], dtype=ht.float64, device=cpu:0, split=None)
    >>> s
    1
    >>> T2 = ht.float32([[1.5, 2.5], [1.0, 1.3]])
    >>> T1.floordiv_(T2)
    DNDarray([[0., 0.],
              [1., 3.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T1
    DNDarray([[0., 0.],
              [1., 3.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[1.5000, 2.5000],
              [1.0000, 1.3000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> v = ht.int32([-1, 2])
    >>> T1.floor_divide_(v)
    DNDarray([[-0.,  0.],
              [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T1
    DNDarray([[-0.,  0.],
              [-1.,  1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> v
    DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)
    """

    def wrap_floordiv_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.floor_divide_(b)

    try:
        return _operations.__binary_op(wrap_floordiv_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__ifloordiv__ = floordiv_
DNDarray.floordiv_ = DNDarray.floor_divide_ = floordiv_


def fmod(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise division remainder of values of operand ``t1`` by values of operand ``t2`` (i.e.
    C Library function fmod).
    Result has the sign as the dividend ``t1``. Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided (may be floats)
    t2: DNDarray or scalar
        The second operand by whose values is divided (may be floats)
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned. If provided, it must be of the same
        shape as the expected output.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.fmod(2.0, 2.0)
    DNDarray(0., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.fmod(T1, T2)
    DNDarray([[1., 0.],
          [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.fmod(s, T1)
    DNDarray([[0., 0.],
          [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.fmod, t1, t2, out, where)


def fmod_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    In-place computation of element-wise division remainder of values of operand `t1` by values of
    operand `t2` (i.e. C Library function fmod). The result has the same sign as the dividend `t1`.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t1: DNDarray
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided (may be floats)

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array(2)
    >>> T1.fmod_(T1)
    >>> T1
    DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
    >>> T2 = ht.float32([[1, 2], [3, 4]])
    >>> T3 = ht.int32([[2, 2], [2, 2]])
    >>> T2.fmod_(T3)
    DNDarray([[1., 0.],
              [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[1., 0.],
              [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T3
    DNDarray([[2, 2],
              [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
    >>> s = -3
    >>> T3.fmod_(s)
    DNDarray([[2, 2],
              [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
    >>> T3
    DNDarray([[2, 2],
              [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
    >>> s
    -3
    """

    def wrap_fmod_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.fmod_(b)

    try:
        return _operations.__binary_op(wrap_fmod_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.fmod_ = fmod_


def gcd(
    a: DNDarray,
    b: DNDarray,
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Returns the greatest common divisor of |a| and |b| element-wise.

    Parameters
    ----------
    a:   DNDarray
         The first input array, must be of integer type
    b:   DNDarray
         the second input array, must be of integer type
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.int(ht.ones(3)) * 9
    >>> T2 = ht.arange(3) + 1
    >>> ht.gcd(T1, T2)
    DNDarray([1, 1, 3], dtype=ht.int32, device=cpu:0, split=None)
    """
    try:
        res = _operations.__binary_op(torch.gcd, a, b, out, where)
    except RuntimeError:
        # every other possibility is caught by __binary_op
        raise TypeError(f"Expected integer input, got {a.dtype}, {b.dtype}")

    return res


def gcd_(t1: DNDarray, t2: DNDarray) -> DNDarray:
    """
    Returns the greatest common divisor of |t1| and |t2| element-wise and in-place.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
    greatest common divisor with the corresponding element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise gcd of `t1` and `t2`.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t1: DNDarray
         The first input array, must be of integer type
    t2: DNDarray
         The second input array, must be of integer type

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.int(ht.ones(3)) * 9
    >>> T2 = ht.arange(3) + 1
    >>> T1.gcd_(T2)
    DNDarray([1, 1, 3], dtype=ht.int32, device=cpu:0, split=None)
    >>> T1
    DNDarray([1, 1, 3], dtype=ht.int32, device=cpu:0, split=None)
    >>> T2
    DNDarray([1, 2, 3], dtype=ht.int32, device=cpu:0, split=None)
    >>> s = 2
    >>> T2.gcd_(2)
    DNDarray([1, 2, 1], dtype=ht.int32, device=cpu:0, split=None)
    >>> T2
    DNDarray([1, 2, 1], dtype=ht.int32, device=cpu:0, split=None)
    >>> s
    2
    """

    def wrap_gcd_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.gcd_(b)

    try:
        return _operations.__binary_op(wrap_gcd_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )
    except RuntimeError:
        raise TypeError(f"Expected integer input, got {t1.dtype}, {t2.dtype}")


DNDarray.gcd_ = gcd_


def hypot(
    a: DNDarray,
    b: DNDarray,
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Given the 'legs' of a right triangle, return its hypotenuse. Equivalent to
    :math:`sqrt(a^2 + b^2)`, element-wise.

    Parameters
    ----------
    a:   DNDarray
         The first input array
    b:   DNDarray
         the second input array
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> a = ht.array([2.])
    >>> b = ht.array([1.,3.,3.])
    >>> ht.hypot(a,b)
    DNDarray([2.2361, 3.6056, 3.6056], dtype=ht.float32, device=cpu:0, split=None)
    """
    try:
        res = _operations.__binary_op(torch.hypot, a, b, out, where)
    except RuntimeError:
        # every other possibility is caught by __binary_op
        raise TypeError(f"Not implemented for array dtype, got {a.dtype}, {b.dtype}")

    return res


def hypot_(t1: DNDarray, t2: DNDarray) -> DNDarray:
    """
    Given the 'legs' of a right triangle, return its hypotenuse in-place of the first input.
    Equivalent to :math:`sqrt(a^2 + b^2)`, element-wise.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t1:  DNDarray
         The first input array
    t2:  DNDarray
         the second input array

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array([1.,3.,3.])
    >>> T2 = ht.array(2.)
    >>> T1.hypot_(T2)
    DNDarray([2.2361, 3.6056, 3.6056], dtype=ht.float32, device=cpu:0, split=None)
    >>> T1
    DNDarray([2.2361, 3.6056, 3.6056], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
    """

    def wrap_hypot_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.hypot_(b)

    try:
        return _operations.__binary_op(wrap_hypot_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )
    except RuntimeError:
        raise TypeError(f"Not implemented for array dtype, got {t1.dtype}, {t2.dtype}")


DNDarray.hypot_ = hypot_


def invert(a: DNDarray, /, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Computes the bitwise NOT of the given input :class:`~heat.core.dndarray.DNDarray`. The input
    array must be of integral or Boolean types. For boolean arrays, it computes the logical NOT.
    Bitwise_not is an alias for invert.

    Parameters
    ---------
    a: DNDarray
        The input array to invert. Must be of integral or Boolean types
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output. The dtype of the output will be the one of the input array, unless it is
        logical, in which case it will be casted to int8. If not provided or None, a freshly-
        allocated array is returned.

    Examples
    --------
    >>> ht.invert(ht.array([13], dtype=ht.uint8))
    DNDarray([242], dtype=ht.uint8, device=cpu:0, split=None)
    >>> ht.bitwise_not(ht.array([-1, -2, 3], dtype=ht.int8))
    DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)
    """
    dt = heat_type_of(a)

    if heat_type_is_inexact(dt):
        raise TypeError("Operation is not supported for float types")

    return _operations.__local_op(torch.bitwise_not, a, out, no_cast=True)


DNDarray.__invert__ = lambda self: invert(self)
DNDarray.__invert__.__doc__ = invert.__doc__

# alias for invert
bitwise_not = invert
"""Alias for :py:func:`invert`"""


def invert_(t: DNDarray) -> DNDarray:
    """
    Computes the bitwise NOT of the given input :class:`~heat.core.dndarray.DNDarray` in-place. The
    elements of the input array must be of integer or Boolean types. For boolean arrays, it computes
    the logical NOT.
    Can only be called as a DNDarray method. `bitwise_not_` is an alias for `invert_`.

    Parameters
    ----------
    t:  DNDarray
        The input array to invert. Must be of integral or Boolean types

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array(13, dtype=ht.uint8)
    >>> T1.invert_()
    DNDarray(242, dtype=ht.uint8, device=cpu:0, split=None)
    >>> T1
    DNDarray(242, dtype=ht.uint8, device=cpu:0, split=None)
    >>> T2 = ht.array([-1, -2, 3], dtype=ht.int8)
    >>> T2.invert_()
    DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)
    >>> T2
    DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)
    >>> T3 = ht.array([[True, True], [False, True]])
    >>> T3.invert_()
    DNDarray([[False, False],
              [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> T3
    DNDarray([[False, False],
              [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
    """
    dt = heat_type_of(t)

    if heat_type_is_inexact(dt):
        raise TypeError("Operation is not supported for float types")

    def wrap_bitwise_not_(a: torch.Tensor, out=None) -> torch.Tensor:
        return a.bitwise_not_()

    return _operations.__local_op(wrap_bitwise_not_, t, no_cast=True, out=t)


DNDarray.invert_ = DNDarray.bitwise_not_ = invert_


def lcm(
    a: DNDarray,
    b: DNDarray,
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Returns the lowest common multiple of |a| and |b| element-wise.

    Parameters
    ----------
    a:   DNDarray or scalar
         The first input (array), must be of integer type
    b:   DNDarray or scalar
         the second input (array), must be of integer type
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> a = ht.array([6, 12, 15])
    >>> b = ht.array([3, 4, 5])
    >>> ht.lcm(a,b)
    DNDarray([ 6, 12, 15], dtype=ht.int64, device=cpu:0, split=None)
    >>> s = 2
    >>> ht.lcm(s,a)
    DNDarray([ 6, 12, 30], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.lcm(b,s)
    DNDarray([ 6,  4, 10], dtype=ht.int64, device=cpu:0, split=None)
    """
    try:
        res = _operations.__binary_op(torch.lcm, a, b, out, where)
    except RuntimeError:
        # every other possibility is caught by __binary_op
        raise TypeError(f"Expected integer input, got {a.dtype}, {b.dtype}")

    return res


def lcm_(t1: DNDarray, t2: Union[DNDarray, int]) -> DNDarray:
    """
    Returns the lowest common multiple of |t1| and |t2| element-wise and in-place.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
    lowest common multiple with the corresponding element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise gcd of the absolute values of `t1` and `t2`.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t1:  DNDarray
         The first input array, must be of integer type
    t2:  DNDarray or scalar
         the second input array, must be of integer type

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array([6, 12, 15])
    >>> T2 = ht.array([3, 4, 5])
    >>> T1.lcm_(T2)
    DNDarray([ 6, 12, 15], dtype=ht.int64, device=cpu:0, split=None)
    >>> T1
    DNDarray([ 6, 12, 15], dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray([3, 4, 5], dtype=ht.int64, device=cpu:0, split=None)
    >>> s = 2
    >>> T2.lcm_(s)
    DNDarray([ 6,  4, 10], dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray([ 6,  4, 10], dtype=ht.int64, device=cpu:0, split=None)
    >>> s
    2
    """

    def wrap_lcm_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.lcm_(b)

    try:
        return _operations.__binary_op(wrap_lcm_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )
    except RuntimeError:
        raise TypeError(f"Expected integer input, got {t1.dtype}, {t2.dtype}")


DNDarray.lcm_ = lcm_


def left_shift(
    t1: DNDarray,
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Shift the bits of an integer to the left.

    Parameters
    ----------
    t1: DNDarray
        Input array
    t2: DNDarray or float
        Integer number of zero bits to add
    out: DNDarray, optional
        Output array for the result. Must have the same shape as the expected output. The dtype of
        the output will be the one of the input array, unless it is logical, in which case it will
        be casted to int8. If not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the shifted value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.left_shift(ht.array([1,2,3]), 1)
    DNDarray([2, 4, 6], dtype=ht.int64, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))
    arrs = [t1, t2]
    for dt in range(2):
        if heat_type_is_inexact(dtypes[dt]):
            raise TypeError("Operation is not supported for float types")
        elif dtypes[dt] == types.bool:
            arrs[dt] = types.int(arrs[dt])

    return _operations.__binary_op(torch.bitwise_left_shift, t1, t2, out, where)


def _lshift(self, other):
    try:
        return left_shift(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__lshift__ = _lshift
DNDarray.__lshift__.__doc__ = left_shift.__doc__
DNDarray.__rlshift__ = lambda self, other: _lshift(other, self)
DNDarray.__rlshift__.__doc__ = left_shift.__doc__


def left_shift_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    In-place version of `left_shift`.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise shifts the bits
    of each element in-place that many positions to the left as the element(s) of the second operand
    (scalar or :class:`~heat.core.dndarray.DNDarray`) indicate, i.e. the element(s) of `t1` are
    overwritten by the results of element-wise bitwise left shift of `t1` for `t2` positions.
    Can be called as a DNDarray method or with the symbol `<<=`. Only works for inputs with integer
    elements.

    Parameters
    ----------
    t1: DNDarray
        Input array
    t2: DNDarray or float
        Integer number of zero bits to add

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array([1,2,3])
    >>> s = 1
    >>> T1.left_shift_(s)
    DNDarray([2, 4, 6], dtype=ht.int64, device=cpu:0, split=None)
    >>> T1
    DNDarray([2, 4, 6], dtype=ht.int64, device=cpu:0, split=None)
    >>> s
    1
    >>> T2 = ht.array([-1, 1, 0])
    >>> T1 <<= T2
    >>> T1
    DNDarray([0, 8, 6], dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray([-1,  1,  0], dtype=ht.int64, device=cpu:0, split=None)
    """
    dtypes = dtype1, dtype2 = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if not heat_type_is_exact(dt):
            raise TypeError(
                "Operation is only supported for inputs whose elements are integers, but your "
                + f"inputs have the datatypes {dtype1} and {dtype2}."
            )

    def wrap_bitwise_left_shift_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.bitwise_left_shift_(b)

    try:
        return _operations.__binary_op(wrap_bitwise_left_shift_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__ilshift__ = left_shift_
DNDarray.left_shift_ = left_shift_


def mul(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise multiplication (NOT matrix multiplication) of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    multiplied as argument.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the multiplication
    t2: DNDarray or scalar
        The second operand involved in the multiplication
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If not provided or
        None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the multiplied value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.mul(2.0, 4.0)
    DNDarray(8., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> s = 3.0
    >>> ht.mul(T1, s)
    DNDarray([[ 3.,  6.],
              [ 9., 12.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.mul(T1, T2)
    DNDarray([[2., 4.],
              [6., 8.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.mul, t1, t2, out, where)


def _mul(self, other):
    try:
        return mul(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__mul__ = _mul
DNDarray.__mul__.__doc__ = mul.__doc__
DNDarray.__rmul__ = lambda self, other: _mul(other, self)
DNDarray.__rmul__.__doc__ = mul.__doc__

# Alias in compliance with numpy API
multiply = mul
"""Alias for :py:func:`mul`"""


def mul_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise in-place multiplication of values of two operands.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise multiplies the
    element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
    i.e. the element(s) of `t1` are overwritten by the results of element-wise multiplication of
    `t1` and `t2`.
    Can be called as a DNDarray method or with the symbol `*=`. `multiply_` is an alias for `mul_`.

    Parameters
    ----------
    t1: DNDarray
        The first operand involved in the multiplication.
    t2: DNDarray or scalar
        The second operand involved in the multiplication.

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> T1 *= T2
    >>> T1
    DNDarray([[2., 4.],
              [6., 8.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[2., 2.],
              [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> T2.mul_(s)
    DNDarray([[4., 4.],
              [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[4., 4.],
              [4., 4.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s
    2.0
    >>> v = ht.int32([-1, 2])
    >>> T2.multiply_(v)
    DNDarray([[-4.,  8.],
              [-4.,  8.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[-4.,  8.],
              [-4.,  8.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> v
    DNDarray([-1,  2], dtype=ht.int32, device=cpu:0, split=None)
    """

    def wrap_mul_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.mul_(b)

    try:
        return _operations.__binary_op(wrap_mul_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__imul__ = mul_
DNDarray.mul_ = DNDarray.multiply_ = mul_


def nan_to_num(
    a: DNDarray,
    nan: float = 0.0,
    posinf: float = None,
    neginf: float = None,
    out: Optional[DNDarray] = None,
) -> DNDarray:
    """
    Replaces NaNs, positive infinity values, and negative infinity values in the input 'a' with the
    values specified by nan, posinf, and neginf, respectively. By default, NaNs are replaced with
    zero, positive infinity is replaced with the greatest finite value representable by input's
    dtype, and negative infinity is replaced with the least finite value representable by input's
    dtype.

    Parameters
    ----------
    a : DNDarray
        Input array.
    nan : float, optional
        Value to be used to replace NaNs. Default value is 0.0.
    posinf : float, optional
        Value to replace positive infinity values with. If None, positive infinity values are
        replaced with the greatest finite value of the input's dtype. Default value is None.
    neginf : float, optional
        Value to replace negative infinity values with. If None, negative infinity values are
        replaced with the greatest negative finite value of the input's dtype. Default value is
        None.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.

    Examples
    --------
    >>> x = ht.array([float('nan'), float('inf'), -float('inf')])
    >>> ht.nan_to_num(x)
    DNDarray([ 0.0000e+00,  3.4028e+38, -3.4028e+38], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(
        torch.nan_to_num, a, out=out, no_cast=True, nan=nan, posinf=posinf, neginf=neginf
    )


def nan_to_num_(
    t: DNDarray, nan: float = 0.0, posinf: float = None, neginf: float = None
) -> DNDarray:
    """
    Replaces NaNs, positive infinity values, and negative infinity values in the input 't' in-place
    with the values specified by nan, posinf, and neginf, respectively. By default, NaNs are
    replaced with zero, positive infinity is replaced with the greatest finite value representable
    by input's dtype, and negative infinity is replaced with the least finite value representable by
    input's dtype.
    Can only be called as a DNDarray method.

    Parameters
    ----------
    t:      DNDarray
            Input array.
    nan:    float, optional
            Value to be used to replace NaNs. Default value is 0.0.
    posinf: float, optional
            Value to replace positive infinity values with. If None, positive infinity values are
            replaced with the greatest finite value of the input's dtype. Default value is None.
    neginf: float, optional
            Value to replace negative infinity values with. If None, negative infinity values are
            replaced with the greatest negative finite value of the input's dtype. Default value is
            None.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array([float('nan'), float('inf'), -float('inf')])
    >>> T1.nan_to_num_()
    DNDarray([ 0.0000e+00,  3.4028e+38, -3.4028e+38], dtype=ht.float32, device=cpu:0, split=None)
    >>> T1
    DNDarray([ 0.0000e+00,  3.4028e+38, -3.4028e+38], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2 = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf])
    >>> T2.nan_to_num_(nan=0, posinf=1, neginf=-1)
    DNDarray([ 1.,  2.,  3.,  0.,  1., -1.], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([ 1.,  2.,  3.,  0.,  1., -1.], dtype=ht.float32, device=cpu:0, split=None)
    """

    def wrap_nan_to_num_(
        a: torch.Tensor, nan=nan, posinf=posinf, neginf=neginf, out=None
    ) -> torch.Tensor:
        return a.nan_to_num_(nan=nan, posinf=posinf, neginf=neginf)

    return _operations.__local_op(
        wrap_nan_to_num_, t, out=t, no_cast=True, nan=nan, posinf=posinf, neginf=neginf
    )


DNDarray.nan_to_num_ = nan_to_num_


def nanprod(
    a: DNDarray,
    axis: Union[int, Tuple[int, ...]] = None,
    out: DNDarray = None,
    keepdims: bool = None,
) -> DNDarray:
    """
    Return the product of array elements over a given axis treating Not a Numbers (NaNs) as one.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which a product is performed. The default, ``axis=None``, will calculate
        the product of all the elements in the input array. If axis is negative it counts from the
        last to the first axis.
        If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple
        instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input array.

    Examples
    --------
    >>> ht.nanprod(ht.array([4.,ht.nan]))
    DNDarray(4., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.nanprod(ht.array([
        [1.,ht.nan],
        [3.,4.]]))
    DNDarray(12., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.nanprod(ht.array([
        [1.,ht.nan],
        [ht.nan,4.]
    ]), axis=1)
    DNDarray([ 1., 4.], dtype=ht.float32, device=cpu:0, split=None)
    """
    b = nan_to_num(a, nan=1)

    return _operations.__reduce_op(
        b, torch.prod, MPI.PROD, axis=axis, out=out, neutral=1, keepdims=keepdims
    )


def nansum(
    a: DNDarray,
    axis: Union[int, Tuple[int, ...]] = None,
    out: DNDarray = None,
    keepdims: bool = None,
) -> DNDarray:
    """
    Sum of array elements over a given axis treating Not a Numbers (NaNs) as zero. An array with the
    same shape as ``self.__array`` except for the specified axis which becomes one, e.g.
    ``a.shape=(1, 2, 3)`` => ``ht.ones((1, 2, 3)).sum(axis=1).shape=(1, 1, 3)``

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis along which a sum is performed. The default, ``axis=None``, will sum all of the
        elements of the input array. If ``axis`` is negative it counts from the last to the first
        axis. If ``axis`` is a tuple of ints, a sum is performed on all of the axes specified in the
        tuple instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input
        array.

    Examples
    --------
    >>> ht.sum(ht.ones(2))
    DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3,3)))
    DNDarray(9., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3,3)).astype(ht.int))
    DNDarray(9, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3,2,1)), axis=-3)
    DNDarray([[3.],
              [3.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__reduce_op(
        a, torch.nansum, MPI.SUM, axis=axis, out=out, neutral=0, keepdims=keepdims
    )


def neg(a: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Element-wise negation of `a`.

    Parameters
    ----------
    a:   DNDarray
         The input array.
    out: DNDarray, optional
         The output array. It must have a shape that the inputs broadcast to

    Examples
    --------
    >>> ht.neg(ht.array([-1, 1]))
    DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
    >>> -ht.array([-1., 1.])
    DNDarray([ 1., -1.], dtype=ht.float32, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(a)

    return _operations.__local_op(torch.neg, a, out, no_cast=True)


DNDarray.__neg__ = lambda self: neg(self)
DNDarray.__neg__.__doc__ = neg.__doc__

# Alias in compliance with numpy API
negative = neg
"""Alias for :py:func:`neg`"""


def neg_(t: DNDarray) -> DNDarray:
    """
    Element-wise in-place negation of `t`.
    Can only be called as a DNDarray method. `negative_` is an alias for `neg_`.

    Parameter
    ----------
    t:  DNDarray
        The input array

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array([-1, 1])
    >>> T1.neg_()
    DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
    >>> T1
    DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
    >>> T2 = ht.array([[-1., 2.5], [4. , 0.]])
    >>> T2.neg_()
    DNDarray([[ 1.0000, -2.5000],
              [-4.0000, -0.0000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[ 1.0000, -2.5000],
              [-4.0000, -0.0000]], dtype=ht.float32, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(t)

    def wrap_neg_(a: torch.Tensor, out=None) -> torch.Tensor:
        return a.neg_()

    return _operations.__local_op(wrap_neg_, t, out=t, no_cast=True)


DNDarray.neg_ = DNDarray.negative_ = neg_


def pos(a: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Element-wise positive of `a`.

    Parameters
    ----------
    a:   DNDarray
         The input array.
    out: DNDarray, optional
         The output array. It must have a shape that the inputs broadcast to.

    Notes
    -----
    Equivalent to a.copy().

    Examples
    --------
    >>> ht.pos(ht.array([-1, 1]))
    DNDarray([-1,  1], dtype=ht.int64, device=cpu:0, split=None)
    >>> +ht.array([-1., 1.])
    DNDarray([-1.,  1.], dtype=ht.float32, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(a)

    def torch_pos(torch_tensor, out=None):
        return out.copy_(torch_tensor)

    if out is not None:
        return _operations.__local_op(torch_pos, a, out, no_cast=True)

    return a.copy()


DNDarray.__pos__ = lambda self: pos(self)
DNDarray.__pos__.__doc__ = pos.__doc__

# Alias in compliance with numpy API
positive = pos
"""Alias for :py:func:`pos`"""


def pow(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise power function of values of operand ``t1`` to the power of values of operand
    ``t2`` (i.e ``t1**t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values represent the base
    t2: DNDarray or scalar
        The second operand whose values represent the exponent
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If
        not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the exponentiated value. Elsewhere, the `out` array will retain its
        original value. If an uninitialized `out` array is created via the default `out=None`,
        locations within it where the condition is False will remain uninitialized. If distributed,
        the split axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.pow (3.0, 2.0)
    DNDarray(9., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[3, 3], [2, 2]])
    >>> ht.pow(T1, T2)
    DNDarray([[ 1.,  8.],
            [ 9., 16.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 3.0
    >>> ht.pow(T1, s)
    DNDarray([[ 1.,  8.],
            [27., 64.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    # early exit for integer scalars
    if isinstance(t2, int):
        try:
            result = torch.pow(t1.larray, t2)
            return DNDarray(
                result,
                gshape=t1.gshape,
                dtype=t1.dtype,
                device=t1.device,
                split=t1.split,
                comm=t1.comm,
                balanced=t1.balanced,
            )
        except AttributeError:
            # t1 is no DNDarray
            pass
    elif isinstance(t1, int):
        try:
            result = torch.pow(t1, t2.larray)
            return DNDarray(
                result,
                gshape=t2.gshape,
                dtype=t2.dtype,
                device=t2.device,
                split=t2.split,
                comm=t2.comm,
                balanced=t2.balanced,
            )
        except AttributeError:
            # t2 is no DNDarray
            pass
    return _operations.__binary_op(torch.pow, t1, t2, out, where)


def _pow(self, other, modulo=None):
    if modulo is not None:
        return NotImplemented

    try:
        return pow(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__pow__ = _pow
DNDarray.__pow__.__doc__ = pow.__doc__
DNDarray.__rpow__ = lambda self, other, modulo=None: _pow(other, self, modulo)
DNDarray.__rpow__.__doc__ = pow.__doc__


# Alias in compliance with numpy API
power = pow
"""Alias for :py:func:`pow`"""


def pow_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise in-place exponentation.
    Takes the element(s) of the first operand (:class:`~heat.core.dndarray.DNDarray`) element-wise
    to the power of the corresponding element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise exponentiation of `t1` and `t2`.
    Can be called as a DNDarray method or with the symbol `**=`. `power_` is an alias for `pow_`.

    Parameters
    ----------
    t1: DNDarray
        The first operand whose values represent the base
    t2: DNDarray or scalar
        The second operand whose values represent the exponent

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[3, 3], [2, 2]])
    >>> T1 **= T2
    >>> T1
    DNDarray([[ 1.,  8.],
              [ 9., 16.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[3., 3.],
              [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = -1.0
    >>> T2.pow_(s)
    DNDarray([[0.3333, 0.3333],
              [0.5000, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[0.3333, 0.3333],
              [0.5000, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s
    -1.0
    >>> v = ht.int32([-3, 2])
    >>> T2.power_(v)
    DNDarray([[27.0000,  0.1111],
              [ 8.0000,  0.2500]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[27.0000,  0.1111],
              [ 8.0000,  0.2500]], dtype=ht.float32, device=cpu:0, split=None)
    >>> v
    DNDarray([-3,  2], dtype=ht.int32, device=cpu:0, split=None)
    """

    def wrap_pow_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.pow_(b)

    try:
        return _operations.__binary_op(wrap_pow_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__ipow__ = pow_
DNDarray.pow_ = DNDarray.power_ = pow_


def prod(
    a: DNDarray,
    axis: Union[int, Tuple[int, ...]] = None,
    out: DNDarray = None,
    keepdims: bool = None,
) -> DNDarray:
    """
    Return the product of array elements over a given axis in form of a DNDarray shaped as a but
    with the specified axis removed.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which a product is performed. The default, ``axis=None``, will calculate
        the product of all the elements in the input array. If axis is negative it counts from the
        last to the first axis. If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input
        array.

    Examples
    --------
    >>> ht.prod(ht.array([1.,2.]))
    DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.prod(ht.array([
        [1.,2.],
        [3.,4.]]))
    DNDarray(24., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.prod(ht.array([
        [1.,2.],
        [3.,4.]
    ]), axis=1)
    DNDarray([ 2., 12.], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__reduce_op(
        a, torch.prod, MPI.PROD, axis=axis, out=out, neutral=1, keepdims=keepdims
    )


DNDarray.prod = lambda self, axis=None, out=None, keepdims=None: prod(self, axis, out, keepdims)
DNDarray.prod.__doc__ = prod.__doc__


def remainder(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise division remainder of values of operand ``t1`` by values of operand ``t2`` (i.e.
    ``t1%t2``). Result has the same sign as the divisor ``t2``.
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.remainder(2, 2)
    DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
    >>> T1 = ht.int32([[1, 2], [3, 4]])
    >>> T2 = ht.int32([[2, 2], [2, 2]])
    >>> ht.remainder(T1, T2)
    DNDarray([[1, 0],
            [1, 0]], dtype=ht.int32, device=cpu:0, split=None)
    >>> s = 2
    >>> ht.remainder(s, T1)
    DNDarray([[0, 0],
            [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.remainder, t1, t2, out, where)


# Alias support
mod = remainder
"""Alias for :py:func:`remainder`"""


def _mod(self, other):
    try:
        return mod(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__mod__ = _mod
DNDarray.__mod__.__doc__ = mod.__doc__
DNDarray.__rmod__ = lambda self, other: _mod(other, self)
DNDarray.__rmod__.__doc__ = mod.__doc__


def remainder_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise in-place division remainder of values of two operands. The result has the same sign
    as the divisor.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise computes the
    modulo regarding the element(s) of the second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) in-place, i.e. the element(s) of `t1` are overwritten by
    the results of element-wise `t1` modulo `t2`.
    Can be called as a DNDarray method or with the symbol `%=`. `mod_` is an alias for `remainder_`.

    Parameters
    ----------
    t1: DNDarray
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array(2)
    >>> T1 %= T1
    >>> T1
    DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
    >>> T2 = ht.float32([[1, 2], [3, 4]])
    >>> T3 = ht.int32([[2, 2], [2, 2]])
    >>> T2.mod_(T3)
    DNDarray([[1., 0.],
              [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[1., 0.],
              [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T3
    DNDarray([[2, 2],
              [2, 2]], dtype=ht.int32, device=cpu:0, split=None)
    >>> s = -3
    >>> T3.remainder_(s)
    DNDarray([[-1, -1],
              [-1, -1]], dtype=ht.int32, device=cpu:0, split=None)
    >>> T3
    DNDarray([[-1, -1],
              [-1, -1]], dtype=ht.int32, device=cpu:0, split=None)
    >>> s
    -3
    """

    def wrap_remainder_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.remainder_(b)

    try:
        return _operations.__binary_op(wrap_remainder_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__imod__ = remainder_
DNDarray.mod_ = DNDarray.remainder_ = remainder_


def right_shift(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Shift the bits of an integer to the right.

    Parameters
    ----------
    t1: DNDarray or scalar
        Input array
    t2: DNDarray or scalar
        Integer number of bits to remove
    out: DNDarray, optional
        Output array for the result. Must have the same shape as the expected output. The dtype of
        the output will be the one of the input array, unless it is logical, in which case it will
        be casted to int8. If not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the shifted value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.right_shift(ht.array([1,2,3]), 1)
    DNDarray([0, 1, 1], dtype=ht.int64, device=cpu:0, split=None)
    """
    dtypes = (heat_type_of(t1), heat_type_of(t2))
    arrs = [t1, t2]
    for dt in range(2):
        if heat_type_is_inexact(dtypes[dt]):
            raise TypeError("Operation is not supported for float types")
        elif dtypes[dt] == types.bool:
            arrs[dt] = types.int(arrs[dt])

    return _operations.__binary_op(torch.bitwise_right_shift, t1, t2, out, where)


def _rshift(self, other):
    try:
        return right_shift(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__rshift__ = _rshift
DNDarray.__rshift__.__doc__ = right_shift.__doc__
DNDarray.__rrshift__ = lambda self, other: _rshift(other, self)
DNDarray.__rrshift__.__doc__ = right_shift.__doc__


def right_shift_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    In-place version of `right_shift`.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise shifts the bits
    of each element in-place that many positions to the right as the element(s) of the second
    operand (scalar or :class:`~heat.core.dndarray.DNDarray`) indicate, i.e. the element(s) of `t1`
    are overwritten by the results of element-wise bitwise right shift of `t1` for `t2` positions.
    Can be called as a DNDarray method or with the symbol `>>=`. Only works for inputs with integer
    elements.

    Parameters
    ----------
    t1: DNDarray
        Input array
    t2: DNDarray or float
        Integer number of zero bits to remove

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.array([1,2,32])
    >>> s = 1
    >>> T1.right_shift_(s)
    DNDarray([ 0,  1, 16], dtype=ht.int64, device=cpu:0, split=None)
    >>> T1
    DNDarray([0, 1, 1], dtype=ht.int64, device=cpu:0, split=None)
    >>> s
    1
    >>> T2 = ht.array([2, -3, 2])
    >>> T1 >>= T2
    >>> T1
    DNDarray([0, 0, 4], dtype=ht.int64, device=cpu:0, split=None)
    >>> T2
    DNDarray([ 2, -3,  2], dtype=ht.int64, device=cpu:0, split=None)
    """
    dtypes = dtype1, dtype2 = (heat_type_of(t1), heat_type_of(t2))

    for dt in dtypes:
        if not heat_type_is_exact(dt):
            raise TypeError(
                "Operation is only supported for inputs whose elements are integers, but your "
                + f"inputs have the datatypes {dtype1} and {dtype2}."
            )

    def wrap_bitwise_right_shift_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.bitwise_right_shift_(b)

    try:
        return _operations.__binary_op(wrap_bitwise_right_shift_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__irshift__ = right_shift_
DNDarray.right_shift_ = right_shift_


def sub(
    t1: Union[DNDarray, float],
    t2: Union[DNDarray, float],
    /,
    out: Optional[DNDarray] = None,
    *,
    where: Union[bool, DNDarray] = True,
) -> DNDarray:
    """
    Element-wise subtraction of values of operand ``t2`` from values of operands ``t1`` (i.e
    ``t1-t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand from which values are subtracted
    t2: DNDarray or scalar
        The second operand whose values are subtracted
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If
        not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the subtracted value. Elsewhere, the `out` array will retain its
        original value. If an uninitialized `out` array is created via the default `out=None`,
        locations within it where the condition is False will remain uninitialized. If distributed,
        the split axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.sub(4.0, 1.0)
    DNDarray(3., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.sub(T1, T2)
    DNDarray([[-1.,  0.],
              [ 1.,  2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.sub(s, T1)
    DNDarray([[ 1.,  0.],
              [-1., -2.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.sub, t1, t2, out, where)


def _sub(self, other):
    try:
        return sub(self, other)
    except TypeError:
        return NotImplemented


DNDarray.__sub__ = _sub
DNDarray.__sub__.__doc__ = sub.__doc__
DNDarray.__rsub__ = lambda self, other: _sub(other, self)
DNDarray.__rsub__.__doc__ = sub.__doc__


# Alias in compliance with numpy API
subtract = sub
"""
Alias for :py:func:`sub`
"""


def sub_(t1: DNDarray, t2: Union[DNDarray, float]) -> DNDarray:
    """
    Element-wise in-place substitution of values of two operands.
    Takes the first operand (:class:`~heat.core.dndarray.DNDarray`) and element-wise subtracts the
    element(s) of the second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) in-place,
    i.e. the element(s) of `t1` are overwritten by the results of element-wise subtraction of `t2`
    from `t1`.
    Can be called as a DNDarray method or with the symbol `-=`. `subtract_` is an alias for `sub_`.

    Parameters
    ----------
    t1: DNDarray
        The first operand involved in the subtraction
    t2: DNDarray or scalar
        The second operand involved in the subtraction

    Raises
    ------
    ValueError
        If both inputs are DNDarrays that do not have the same split axis and the shapes of their
        underlying torch.tensors differ, s.t. we cannot process them directly without resplitting.
    TypeError
        If the data type of `t2` cannot be cast to the data type of `t1`. Although the
        corresponding out-of-place operation may work, for the in-place version the requirements
        are stricter, because the data type of `t1` does not change.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> T1 -= T2
    >>> T1
    DNDarray([[-1., 0.],
              [ 1., 2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[2., 2.],
              [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.sub_(T2, s)
    DNDarray([[0., 0.],
              [0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[0., 0.],
              [0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s
    2.0
    >>> v = ht.int32([-3, 2])
    >>> T2.subtract_(v)
    DNDarray([[ 3., -2.],
              [ 3., -2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2
    DNDarray([[ 3., -2.],
              [ 3., -2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> v
    DNDarray([-3,  2], dtype=ht.int32, device=cpu:0, split=None)
    """

    def wrap_sub_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.sub_(b)

    try:
        return _operations.__binary_op(wrap_sub_, t1, t2, out=t1)
    except NotImplementedError:
        raise ValueError(
            f"In-place operation not allowed: operands are distributed along different axes. \n Operand 1 with shape {t1.shape} is split along axis {t1.split}. \n Operand 2 with shape {t2.shape} is split along axis {t2.split}."
        )


DNDarray.__isub__ = sub_
DNDarray.sub_ = DNDarray.subtract_ = sub_


def sum(
    a: DNDarray,
    axis: Union[int, Tuple[int, ...]] = None,
    out: DNDarray = None,
    keepdims: bool = None,
) -> DNDarray:
    """
    Sum of array elements over a given axis. An array with the same shape as ``self.__array`` except
    for the specified axis which becomes one, e.g.
    ``a.shape=(1, 2, 3)`` => ``ht.ones((1, 2, 3)).sum(axis=1).shape=(1, 1, 3)``

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis along which a sum is performed. The default, ``axis=None``, will sum all of the
        elements of the input array. If ``axis`` is negative it counts from the last to the first
        axis. If ``axis`` is a tuple of ints, a sum is performed on all of the axes specified in the
        tuple instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input
        array.

    Examples
    --------
    >>> ht.sum(ht.ones(2))
    DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3,3)))
    DNDarray(9., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3,3)).astype(ht.int))
    DNDarray(9, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3,2,1)), axis=-3)
    DNDarray([[3.],
              [3.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    # TODO: make me more numpy API complete Issue #101
    return _operations.__reduce_op(
        a, torch.sum, MPI.SUM, axis=axis, out=out, neutral=0, keepdims=keepdims
    )


DNDarray.sum = lambda self, axis=None, out=None, keepdims=None: sum(self, axis, out, keepdims)
