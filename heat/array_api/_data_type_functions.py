from __future__ import annotations

from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ._typing import Dtype

import heat as ht


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


@dataclass
class finfo_object:
    """
    Internal object for return type of finfo.
    """

    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float


@dataclass
class iinfo_object:
    """
    Internal object for return type of iinfo.
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
