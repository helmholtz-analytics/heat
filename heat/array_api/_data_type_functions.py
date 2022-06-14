import heat as ht

from dataclasses import dataclass
from typing import Union

from ._typing import Array, Dtype


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
    Array API compatible wrapper for `ht.finfo`.
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
    Array API compatible wrapper for :`ht.iinfo`.
    """
    ii = ht.iinfo(type)
    return iinfo_object(ii.bits, int(ii.max), int(ii.min))
