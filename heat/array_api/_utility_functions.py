import heat as ht
from typing import Optional, Tuple, Union

from ._array_object import Array


def all(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Tests whether all input array elements evaluate to `True` along a specified axis.
    """
    res = ht.all(x._array, axis=axis, keepdim=True)
    if not keepdims or x._array.ndim == 0:
        res = ht.squeeze(res, axis=axis)

    return Array._new(res)
