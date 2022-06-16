from __future__ import annotations

import heat as ht

from dataclasses import dataclass
from typing import Union

from ._array_object import Array
from ._typing import Dtype


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    """
    Copies an array to a specified data type irrespective of Type Promotion Rules.
    """
    if not copy and dtype == x.dtype:
        return x
    return Array._new(x._array.astype(dtype=dtype))


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
    """
    ii = ht.iinfo(type)
    return iinfo_object(ii.bits, int(ii.max), int(ii.min))
