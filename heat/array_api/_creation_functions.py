from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Array, Device, Dtype, NestedSequence, SupportsBufferProtocol
from ._dtypes import _all_dtypes, _floating_dtypes, default_float, default_int, bool as api_bool

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


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns a two-dimensional array with ones on the ``k`` h diagonal and zeros elsewhere.

    Parameters
    ----------
    n_rows : int
        Number of rows in the output array.
    n_cols : Optional[int]
        Number of columns in the output array. If ``None``, the default number of
        columns in the output array is equal to ``n_rows``. Default: ``None``.
    k : int
        Index of the diagonal. A positive value refers to an upper diagonal, a negative
        value to a lower diagonal, and ``0`` to the main diagonal. Default: ``0``.
    dtype : Optional[Dtype]
        Output array data type. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array. Default: ``None``.
    """
    from ._array_object import Array

    if k != 0:
        raise ValueError("k option not implemented yet")

    if dtype is None:
        dtype = default_float
    if n_cols is None:
        n_cols = n_rows
    return Array._new(ht.eye((n_rows, n_cols), dtype=dtype, device=device))


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
        if isinstance(fill_value, bool):
            dtype = api_bool
        elif isinstance(fill_value, int):
            dtype = default_int
        elif isinstance(fill_value, float):
            dtype = default_float
    res = ht.full(shape, fill_value, dtype=dtype, device=device)
    return Array._new(res)


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns a new array filled with ``fill_value`` and having the same ``shape``
    as an input array ``x``.

    Parameters
    ----------
    x : Array
        Input array from which to derive the output array shape.
    fill_value : Union[int, float]
        Fill value.
    dtype : Optional[Dtype]
        Output array data type. If ``dtype`` is ``None``, the output array data
        type is inferred from x. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array. Default: ``None``.
    """
    from ._array_object import Array

    res = ht.full_like(x._array, fill_value, dtype=dtype, device=device)
    if res.dtype not in _all_dtypes:
        # This will happen if the fill value is not something that Heat
        # coerces to one of the acceptable dtypes.
        raise TypeError("Invalid input to full_like")
    return Array._new(res)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> Array:
    """
    Returns evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start : Union[int, float]
        The start of the interval.
    stop : Union[int, float]
        The end of the interval.
    num : int
        Number of samples.
    dtype : Optional[Dtype]
        Output array data type. Must be a floating-point data type. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array. Default: ``None``.
    endpoint : bool
        Boolean indicating whether to include ``stop`` in the interval. Default: ``True``.
    """
    from ._array_object import Array

    if dtype is None:
        dtype = default_float
    elif dtype not in _floating_dtypes:
        raise TypeError("Only floating dtypes allowed in linspace")

    return Array._new(ht.linspace(start, stop, num, dtype=dtype, device=device, endpoint=endpoint))


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    """
    Returns coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays : Array
        An arbitrary number of one-dimensional arrays representing grid coordinates.
        Each array must have the same numeric data type.
    indexing : str
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or
        one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
        respectively), the ``indexing`` keyword has no effect and is ignored.
        Default: ``'xy'``.
    """
    from ._array_object import Array

    if len({a.dtype for a in arrays}) > 1:
        raise ValueError("meshgrid inputs must all have the same dtype")

    return [
        Array._new(array) for array in ht.meshgrid(*[a._array for a in arrays], indexing=indexing)
    ]


def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    """
    Returns a new array having a specified ``shape`` and filled with ones.

    Parameters
    ----------
    shape : Union[int, Tuple[int, ...]]
        Output array shape.
    dtype : Optional[Dtype]
        Output array data type. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array.
    """
    from ._array_object import Array

    if dtype is None:
        dtype = default_float
    return Array._new(ht.ones(shape, dtype=dtype, device=device))


def ones_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    """
    Returns a new array filled with ones and having the same shape as an input array ``x``.

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

    return Array._new(ht.ones_like(x._array, dtype=dtype, device=device))


def tril(x: Array, /, *, k: int = 0) -> Array:
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x : Array
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions
        form ``MxN`` matrices.
    k : int
        Diagonal above which to zero elements. If ``k = 0``, the diagonal is the
        main diagonal. If ``k < 0``, the diagonal is below the main diagonal.
        If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.
    """
    from ._array_object import Array

    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for tril")
    return Array._new(ht.tril(x._array, k=k))


def triu(x: Array, /, *, k: int = 0) -> Array:
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x : Array
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions
        form ``MxN`` matrices.
    k : int
        Diagonal below which to zero elements. If ``k = 0``, the diagonal is the
        main diagonal. If ``k < 0``, the diagonal is below the main diagonal.
        If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.
    """
    from ._array_object import Array

    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for triu")
    return Array._new(ht.triu(x._array, k=k))


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
        Output array data type. Default: ``None``.
    device : Optional[Device]
        Device on which to place the created array.
    """
    from ._array_object import Array

    if dtype is None:
        dtype = default_float
    return Array._new(ht.zeros(shape, dtype=dtype, device=device))


def zeros_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    """
    Returns a new array filled with zeros and having the same shape as an input array x.

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

    return Array._new(ht.zeros_like(x._array, dtype=dtype))
