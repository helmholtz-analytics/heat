from __future__ import annotations

import heat as ht
from typing import TYPE_CHECKING, Union, Optional, Tuple

from ._dtypes import _all_dtypes, default_float

if TYPE_CHECKING:
    from ._typing import Array, Dtype, Device, NestedSequence, SupportsBufferProtocol


def _check_valid_dtype(dtype):
    # Note: Only spelling dtypes as the dtype objects is supported.

    # We use this instead of "dtype in _all_dtypes" because the dtype objects
    # define equality with the sorts of things we want to disallow.
    for d in (None,) + _all_dtypes:
        if dtype is d:
            return
    raise ValueError(f"{dtype} is not supported")


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
    """
    # _array_object imports in this file are inside the functions to avoid
    # circular imports
    from ._array_object import Array

    # _check_valid_dtype(dtype)
    # if device not in ["cpu", None]:
    #     raise ValueError(f"Unsupported device {device!r}")
    if isinstance(obj, Array):
        if dtype is not None and obj.dtype != dtype:
            copy = True
        # if copy:
        #     return Array._new(np.array(obj._array, copy=True, dtype=dtype))
        if not copy:
            return obj
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
    Returns a new array having a specified `shape` and filled with `fill_value`.
    """
    from ._array_object import Array

    # _check_valid_dtype(dtype)
    # if device not in ["cpu", None]:
    #     raise ValueError(f"Unsupported device {device!r}")
    if isinstance(fill_value, Array) and fill_value.ndim == 0:
        fill_value = fill_value._array
    if dtype is None:
        dtype = default_float
    res = ht.full(shape, fill_value, dtype=dtype, device=device)
    # if res.dtype not in _all_dtypes:
    #     # This will happen if the fill value is not something that NumPy
    #     # coerces to one of the acceptable dtypes.
    #     raise TypeError("Invalid input to full")
    return Array._new(res)


def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns a new array having a specified `shape` and filled with zeros.
    """
    from ._array_object import Array

    # _check_valid_dtype(dtype)
    # if device not in ["cpu", None]:
    #     raise ValueError(f"Unsupported device {device!r}")
    if dtype is None:
        dtype = default_float
    return Array._new(ht.zeros(shape, dtype=dtype, device=device))
