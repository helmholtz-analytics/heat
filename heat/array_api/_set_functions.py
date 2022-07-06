from __future__ import annotations

from ._array_object import Array

from typing import NamedTuple
from ._typing import int64

import heat as ht


class UniqueInverseResult(NamedTuple):
    """
    Internal object for return type of ``unique_inverse``.
    """

    values: Array
    inverse_indices: Array


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    """
    Returns the unique elements of an input array ``x`` and the indices from the
    set of unique elements that reconstruct ``x``.

    Parameters
    ----------
    x : Array
        Input array. If ``x`` has more than one dimension, the function flattens ``x``
        and returns the unique elements of the flattened array.
    """
    values, inverse_indices = ht.unique(x._array, return_inverse=True)
    inverse_indices = inverse_indices.astype(int64)
    return UniqueInverseResult(Array._new(values), Array._new(inverse_indices))


def unique_values(x: Array, /) -> Array:
    """
    Returns the unique elements of an input array ``x``.

    Parameters
    ----------
    x : Array
        Input array. If ``x`` has more than one dimension, the function flattens ``x``
        and returns the unique elements of the flattened array.
    """
    return Array._new(ht.unique(x._array))
