from __future__ import annotations

from ._array_object import Array

from typing import Optional, Tuple, Union

import heat as ht


def all(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Tests whether all input array elements evaluate to ``True`` along a
    specified axis.

    Parameters
    ----------
    x : Array
        Input array.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which to perform a logical AND reduction. By
        default, a logical AND reduction is performed over the entire array.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions. Otherwise, if ``False``, the reduced axes
        (dimensions) are be included in the result. Default: ``False``.
    """
    res = ht.all(x._array, axis=axis, keepdim=True)
    if not keepdims or x._array.ndim == 0:
        res = ht.squeeze(res, axis=axis)

    return Array._new(res)
