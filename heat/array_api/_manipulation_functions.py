import heat as ht
from typing import Tuple, Optional

from ._array_object import Array


def reshape(x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None) -> Array:
    """
    Reshapes an array without changing its data.
    """
    return Array._new(ht.reshape(x._array, shape))
