from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Array, Device, Dtype, NestedSequence, SupportsBufferProtocol
from ._dtypes import _all_dtypes, default_float, default_int

import heat as ht


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns evenly spaced values within the half-open interval ``[start, stop)``
    as a one-dimensional array.

    Parameters
    ----------
    start : Union[int, float]
        If ``stop`` is specified, the start of interval (inclusive); otherwise,
        the end of the interval (exclusive). If ``stop`` is not specified,
        the default starting value is ``0``.
    stop : Optional[Union[int, float]]
        The end of the interval. Default: ``None``.
    step : Union[int, float]
        the distance between two adjacent elements (``out[i+1] - out[i]``). Must
        not be ``0``; may be negative, this results in an empty array if
        ``stop >= start``. Default: ``1``.
    dtype : Optional[Dtype]
        Output array data type. If ``dtype`` is ``None``, the output array data
        type is inferred from ``start``, ``stop`` and ``step``.
    device : Optional[Device]
        Device on which to place the created array. Default: ``None``.
    """
    from ._array_object import Array

    if dtype is None:
        if isinstance(start, float) or isinstance(stop, float) or isinstance(step, float):
            dtype = default_float
        else:
            dtype = default_int
    if stop is not None and (stop - start > 0) != (step > 0):
        return empty(0, dtype=dtype, device=device)
    if stop is None:
        return Array._new(ht.arange(0, start, step, dtype=dtype, device=device))
    else:
        return Array._new(ht.arange(start, stop, step, dtype=dtype, device=device))


def asarray(
    obj: Union[
        Array,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    """
    Convert the input to an array.

    Parameters
    ----------
    obj : Union[Array, bool, int, float, NestedSequence[bool | int | float], SupportsBufferProtocol]
        Object to be converted to an array. May be a Python scalar,
        a (possibly nested) sequence of Python scalars, or an object
        supporting the Python buffer protocol. Default: ``None``.
    dtype : Optional[Dtype]
        Output array data type. If ``dtype`` is ``None``, the output array data
        type is inferred from the data type(s) in ``obj``.
    device : Optional[Device]
        Device on which to place the created array. If ``device`` is ``None`` and
        ``x`` is an array, the output array device is inferred from ``x``.
        Default: ``None``.
    copy : Optional[bool]
        Boolean indicating whether or not to copy the input.
    """
    # _array_object imports in this file are inside the functions to avoid
    # circular imports
    from ._array_object import Array

    if isinstance(obj, Array):
        if dtype is not None and obj.dtype != dtype:
            copy = True
        if not copy:
            return obj
        obj = obj._array
    if dtype is None:
        if isinstance(obj, int) and (obj > 2**64 or obj < -(2**63)):
            # TODO: This won't handle large integers in lists.
            raise OverflowError("Integer out of bounds for array dtypes")
        elif isinstance(obj, float):
            dtype = default_float
    res = ht.asarray(obj, dtype=dtype, copy=copy, device=device)
    return Array._new(res)


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns an uninitialized array having a specified shape.

    Parameters
    ----------
    shape : Union[int, Tuple[int, ...]]
        Output array shape.
    dtype : Optional[Dtype]
        Output array data type. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array. Default: ``None``.
    """
    from ._array_object import Array

    if dtype is None:
        dtype = default_float
    return Array._new(ht.empty(shape, dtype=dtype, device=device))


def empty_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    """
    Returns an uninitialized array with the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x : Array
        Input array from which to derive the output array shape.
    dtype : Optional[Dtype]
        Output array data type. If ``dtype`` is ``None``, the output array data
        type is inferred from x. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array. Default: ``None``.
    """
    from ._array_object import Array

    return Array._new(ht.empty_like(x._array, dtype=dtype, device=device))


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns a new array having a specified ``shape`` and filled with ``fill_value``.

    Parameters
    ----------
    shape : Union[int, Tuple[int, ...]]
        Output array shape.
    fill_value : Union[int, float]
        Fill value.
    dtype : Optional[Dtype]
        Output array data type. If ``dtype`` is ``None``, the output array data
        type is inferred from ``fill_value``. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array.
    """
    from ._array_object import Array

    if isinstance(fill_value, Array) and fill_value.ndim == 0:
        fill_value = fill_value._array
    if dtype is None:
        if isinstance(fill_value, int):
            dtype = default_int
        elif isinstance(fill_value, float):
            dtype = default_float
    res = ht.full(shape, fill_value, dtype=dtype, device=device)
    return Array._new(res)


def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape : Union[int, Tuple[int, ...]]
        Output array shape.
    dtype : Optional[Dtype]
        Output array data type. If ``dtype`` is ``None``, the output array data
        type is the default floating-point data type (``float64``).
        Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array.
    """
    from ._array_object import Array

    if dtype is None:
        dtype = default_float
    return Array._new(ht.zeros(shape, dtype=dtype, device=device))
