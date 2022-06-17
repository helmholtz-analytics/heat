from __future__ import annotations

from ._array_object import Array

from typing import Optional, Tuple

import heat as ht


def reshape(x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None) -> Array:
    """
    Reshapes an array without changing its data.

    Parameters
    ----------
    x : Array
        Input array to reshape.
    shape : Tuple[int, ...]
        A new shape compatible with the original shape. One shape dimension
        is allowed to be ``-1``. When a shape dimension is ``-1``, the
        corresponding output array shape dimension is inferred from the length
        of the array and the remaining dimensions.
    copy : Optional[bool]
        Boolean indicating whether or not to copy the input array.
    """
    res = ht.reshape(x._array, shape)
    if not copy:
        x._array = res
        return x
    return Array._new(res)
