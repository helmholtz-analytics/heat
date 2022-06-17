from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Array, Device, Dtype, NestedSequence, SupportsBufferProtocol
from ._dtypes import _all_dtypes, default_float, default_int

import heat as ht


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
    obj : Union[array, bool, int, float, NestedSequence[bool | int | float], SupportsBufferProtocol]
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
