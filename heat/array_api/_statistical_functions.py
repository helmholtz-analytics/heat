from __future__ import annotations

from ._dtypes import _numeric_dtypes
from ._array_object import Array

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ._typing import Dtype

import heat as ht


def sum(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> Array:
    """
    Calculates the sum of the input array `x`.

    Parameters
    ----------
    x : Array
        Input array. Must have a numeric data type.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis or axes along which sums are computed. By default, the sum is
        computed over the entire array. If a tuple of integers, sums are computed
        over multiple axes. Default: `None`.
    dtype : Optional[Dtype]
        Data type of the returned array.
    keepdims : bool
        If `True`, the reduced axes (dimensions) are included in the result as
        singleton dimensions. Otherwise, if `False`, the reduced axes
        (dimensions) are be included in the result.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sum")
    # Note: sum() and prod() always upcast integers to (u)int64 and float32 to
    # float64 for dtype=None. `np.sum` does that too for integers, but not for
    # float32, so we need to special-case it here
    # if dtype is None and x.dtype == float32:
    #     dtype = float64
    res = ht.sum(x._array, axis=axis, keepdim=True)
    if not keepdims or x._array.ndim == 0:
        res = ht.squeeze(res, axis=axis)
    return Array._new(res)
