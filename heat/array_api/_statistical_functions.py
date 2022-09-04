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
    res = ht.max(x._array, axis=axis, keepdim=keepdims)
    return Array._new(res)


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
    return Array._new(res.astype(x.dtype))


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
    res = ht.min(x._array, axis=axis, keepdim=keepdims)
    return Array._new(res)


def prod(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the product of input array ``x`` elements.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which products are computed. By default, the product is
        computed over the entire array. If a tuple of integers, products are computed
        over multiple axes. Default: ``None``.
    dtype : Optional[Dtype]
        Data type of the returned array.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in prod")
    res = ht.prod(x._array, axis=axis, keepdim=keepdims)
    if dtype is None:
        if x.dtype in _floating_dtypes:
            dtype = default_float
        elif x.dtype in _integer_dtypes:
            dtype = default_int
    if dtype is not None:
        res.astype(dtype, copy=False)
    return Array._new(res)


def std(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the standard deviation of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which standard deviations are computed. By default, the
        standard deviation is computed over the entire array. If a tuple of integers,
        standard deviations are computed over multiple axes. Default: ``None``.
    correction :  Union[int, float]
        Degrees of freedom adjustment.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in std")
    res = ht.std(x._array, axis=axis, ddof=int(correction))
    return Array._new(res)


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
    res = ht.sum(x._array, axis=axis, keepdim=keepdims)
    if dtype is None:
        if x.dtype in _floating_dtypes:
            dtype = default_float
        elif x.dtype in _integer_dtypes:
            dtype = default_int
    if dtype is not None:
        res.astype(dtype, copy=False)
    return Array._new(res)


def var(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the variance of the input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. Must have a floating-point data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which variances are computed. By default, the
        variance is computed over the entire array. If a tuple of integers,
        variances are computed over multiple axes. Default: ``None``.
    correction :  Union[int, float]
        Degrees of freedom adjustment.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in var")
    if axis == ():
        return x
    res = ht.var(x._array, axis=axis, ddof=int(correction))
    return Array._new(res.astype(x.dtype))
