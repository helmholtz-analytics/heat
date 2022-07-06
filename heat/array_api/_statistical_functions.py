from __future__ import annotations

from ._dtypes import _numeric_dtypes, _floating_dtypes, _integer_dtypes, default_float, default_int
from ._array_object import Array

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Dtype

import heat as ht


def max(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the maximum value of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which maximum values are computed. By default, the maximum
        value is computed over the entire array. If a tuple of integers, maximum
        values are computed over multiple axes. Default: ``None``.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in max")
    res = ht.max(x._array, axis=axis, keepdim=True)
    if axis is None:
        if keepdims:
            output_shape = tuple(1 for _ in range(x.ndim))
        else:
            output_shape = ()
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axis = [a if a >= 0 else a + x.ndim for a in axis]
        if keepdims:
            output_shape = tuple(1 if i in axis else dim for i, dim in enumerate(x.shape))
        else:
            output_shape = tuple(dim for i, dim in enumerate(x.shape) if i not in axis)
    return Array._new(res.reshape(output_shape))


def mean(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the arithmetic mean of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which arithmetic means are computed. By default, the mean
        is computed over the entire array. If a tuple of integers, arithmetic means
        are computed over multiple axes. Default: ``None``.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in mean")
    if axis == ():
        return x
    res = ht.mean(x._array, axis=axis)
    if axis is None:
        if keepdims:
            output_shape = tuple(1 for _ in range(x.ndim))
        else:
            output_shape = ()
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axis = [a if a >= 0 else a + x.ndim for a in axis]
        if keepdims:
            output_shape = tuple(1 if i in axis else dim for i, dim in enumerate(x.shape))
        else:
            output_shape = tuple(dim for i, dim in enumerate(x.shape) if i not in axis)
    return Array._new(res.astype(x.dtype).reshape(output_shape))


def min(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the minimum value of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which minimum  values are computed. By default, the minimum
        value is computed over the entire array. If a tuple of integers, minimum
        values are computed over multiple axes. Default: ``None``.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in min")
    res = ht.min(x._array, axis=axis, keepdim=True)
    if axis is None:
        if keepdims:
            output_shape = tuple(1 for _ in range(x.ndim))
        else:
            output_shape = ()
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axis = [a if a >= 0 else a + x.ndim for a in axis]
        if keepdims:
            output_shape = tuple(1 if i in axis else dim for i, dim in enumerate(x.shape))
        else:
            output_shape = tuple(dim for i, dim in enumerate(x.shape) if i not in axis)
    return Array._new(res.reshape(output_shape))


def sum(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the sum of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which sums are computed. By default, the sum is
        computed over the entire array. If a tuple of integers, sums are computed
        over multiple axes. Default: ``None``.
    dtype : Optional[Dtype]
        Data type of the returned array.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")
    # Note: sum() and prod() always upcast integers to (u)int64 and float32 to
    # float64 for dtype=None. `np.sum` does that too for integers, but not for
    # float32, so we need to special-case it here
    # if dtype is None and x.dtype == float32:
    #     dtype = float64
    res = ht.sum(x._array, axis=axis, keepdim=True)
    if not keepdims or x._array.ndim == 0:
        res = ht.squeeze(res, axis=axis)
    if dtype is None:
        if x.dtype in _floating_dtypes:
            dtype = default_float
        elif x.dtype in _integer_dtypes:
            dtype = default_int
    if dtype is not None:
        res.astype(dtype, copy=False)
    return Array._new(res)
