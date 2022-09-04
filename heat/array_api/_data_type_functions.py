from __future__ import annotations

from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, List, Tuple

if TYPE_CHECKING:
    from ._typing import Dtype

import heat as ht
from heat.core.stride_tricks import broadcast_shape


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    """
    Copies an array to a specified data type irrespective of Type Promotion Rules.

    Parameters
    ----------
    x : Array
        Array to cast.
    dtype : Dtype
        Desired data type.
    copy : bool
        If ``True``, a newly allocated array is returned. If ``False`` and the
        specified ``dtype`` matches the data type of the input array, the
        input array is returned; otherwise, a newly allocated is returned.
        Default: ``True``.
    """
    if not copy and dtype == x.dtype:
        return x
    return Array._new(x._array.astype(dtype, copy=True))


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """
    Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays : Array
        An arbitrary number of to-be broadcasted arrays.
    """
    from ._array_object import Array

    # if len(arrays) <= 1:
    #     return arrays
    # output_shape = arrays[0].shape
    # for a in arrays[1:]:
    #     output_shape = broadcast_shape(output_shape, a.shape)
    return [Array._new(array) for array in ht.broadcast_arrays(*[a._array for a in arrays])]


def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Broadcasts an array to a specified shape.

    Parameters
    ----------
    x : Array
        Array to broadcast.
    shape : Tuple[int, ...]
        Array shape. Must be compatible with x.
    """
    from ._array_object import Array

    return Array._new(ht.broadcast_to(x._array, shape))


def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    """
    Determines if one data type can be cast to another data type according to
    Type Promotion Rules.


    Parameters
    ----------
    from : Union[Dtype, Array]
        Input data type or array from which to cast.
    to : Dtype
        Desired data type.
    """
    if isinstance(from_, Array):
        from_ = from_.dtype
    elif from_ not in _all_dtypes:
        raise TypeError(f"{from_=}, but should be an array_api array or dtype")
    if to not in _all_dtypes:
        raise TypeError(f"{to=}, but should be a dtype")
    try:
        # We promote `from_` and `to` together. We then check if the promoted
        # dtype is `to`, which indicates if `from_` can (up)cast to `to`.
        dtype = _result_type(from_, to)
        return to == dtype
    except TypeError:
        # _result_type() raises if the dtypes don't promote together
        return False


@dataclass
class finfo_object:
    """
    Internal object for return type of ``finfo``.
    """

    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float


@dataclass
class iinfo_object:
    """
    Internal object for return type of ``iinfo``.
    """

    bits: int
    max: int
    min: int


def finfo(type: Union[Dtype, Array], /) -> finfo_object:
    """
    Machine limits for floating-point data types.

    Parameters
    ----------
    type : Union[Dtype, Array]
        The kind of floating-point data-type about which to get information.
    """
    fi = ht.finfo(type)
    return finfo_object(
        fi.bits,
        fi.eps,
        fi.max,
        fi.min,
        fi.tiny,
    )


def iinfo(type: Union[Dtype, Array], /) -> iinfo_object:
    """
    Machine limits for integer data types.

    Parameters
    ----------
    type : Union[Dtype, Array]
        The kind of integer data-type about which to get information.
    """
    ii = ht.iinfo(type)
    return iinfo_object(ii.bits, int(ii.max), int(ii.min))


def result_type(*arrays_and_dtypes: Union[Array, Dtype]) -> Dtype:
    """
    Returns the dtype that results from applying the type promotion rules
    to the arguments.

    Parameters
    ----------
    arrays_and_dtypes : Union[Array, Dtype]
        An arbitrary number of input arrays and/or dtypes.
    """
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif isinstance(a, ht.DNDarray) or a not in _all_dtypes:
            raise TypeError("result_type() inputs must be array_api arrays or dtypes")
        A.append(a)

    if len(A) == 0:
        raise ValueError("at least one array or dtype is required")
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = _result_type(t, t2)
        return t
