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
    singleton = x._array.shape == () or (1,)
    res = ht.all(x._array, axis=axis, keepdim=keepdims)
    if singleton:
        res = ht.squeeze(res)
    return Array._new(res)
