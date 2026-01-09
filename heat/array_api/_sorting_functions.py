from __future__ import annotations

from ._array_object import Array

import heat as ht


def sort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Returns a sorted copy of an input array x.

    Parameters
    ----------
    x : Array
        Input array.
    axis : int
        Axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending : bool
        Sort order. If ``True``, the array must be sorted in descending order (by
        value). If ``False``, the array must be sorted in ascending order (by value).
        Default: ``False``.
    stable : bool
        Sort stability. If ``True``, the returned array maintains the relative order
        of ``x`` values which compare as equal. If ``False``, the returned array may
        or may not maintain the relative order of ``x`` values which compare as equal.
        Default: ``True``.
    """
    if stable:
        raise ValueError("Stable sorting not yet implemented")
    res = ht.sort(x._array, axis=axis, descending=descending)
    return Array._new(res[0])
