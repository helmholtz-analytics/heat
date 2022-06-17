from __future__ import annotations

from ._array_object import Array

from typing import Optional, Tuple

import heat as ht


def reshape(x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None) -> Array:
    """
    Reshapes an array without changing its data.
    """
    return Array._new(ht.reshape(x._array, shape))
