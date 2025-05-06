from __future__ import annotations

from ._array_object import Array
from ._data_type_functions import result_type

from typing import Optional, Tuple, Union, List

import heat as ht


def concat(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0) -> Array:
    """
    Joins a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays : Union[Tuple[Array, ...], List[Array]]
        Input arrays to join. The arrays must have the same shape,
        except in the dimension specified by ``axis``.
    axis : Optional[int]
        Axis along which the arrays will be joined. If ``axis`` is ``None``,
        arrays are flattened before concatenation. Default: ``0``.
    """
    result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    if axis is None:
        arrays = tuple(ht.flatten(a) for a in arrays)
        axis = 0
    return Array._new(ht.concatenate(arrays, axis=axis))


def expand_dims(x: Array, /, *, axis: int = 0) -> Array:
    """
    Expands the shape of an array by inserting a new axis (dimension) of
    size one at the position specified by ``axis``.

    Parameters
    ----------
    x : Array
        Input array.
    axis : int
        Axis position (zero-based). If ``x`` has rank (i.e, number of dimensions) ``N``,
        a valid ``axis`` must reside on the closed-interval ``[-N-1, N]``.
    """
    if axis < -x.ndim - 1 or axis > x.ndim:
        raise IndexError("Invalid axis")
    return Array._new(ht.expand_dims(x._array, axis))


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """
    Reverses the order of elements in an array along the given ``axis``. The
    shape of the array is preserved.

    Parameters
    ----------
    x : Array
        Input array.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis (or axes) along which to flip. If ``axis`` is ``None``, the function
        flips all input array axes.
    """
    return Array._new(ht.flip(x._array, axis=axis))


def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """
    Permutes the axes (dimensions) of an array ``x``.

    Parameters
    ----------
    x : Array
        Input array.
    axes : Tuple[int, ...]
        Tuple containing a permutation of ``(0, 1, ..., N-1)`` where ``N`` is the
        number of axes (dimensions) of ``x``.
    """
    return Array._new(ht.transpose(x._array, list(axes)))


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


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    """
    Rolls array elements along a specified axis. Array elements that roll
    beyond the last position are re-introduced at the first position.
    Array elements that roll beyond the first position are re-introduced
    at the last position.

    Parameters
    ----------
    x : Array
        Input array.
    shift : Union[int, Tuple[int, ...]]
        Number of places by which the elements are shifted. If ``shift`` is a
        tuple, then ``axis`` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding element in ``shift``.
        If ``shift`` is an ``int`` and ``axis`` a tuple, then the same ``shift``
        is used for all specified axes.
    axis : Optional[Union[int, Tuple[int, ...]]]
        Axis (or axes) along which elements to shift. If ``axis`` is ``None``,
        the array is flattened, shifted, and then restored to its original shape.
        Default: ``None``.
    """
    return Array._new(ht.roll(x._array, shift, axis=axis))


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """
    Removes singleton dimensions (axes) from ``x``.

    Parameters
    ----------
    x : Array
        Input array.
    axis : Union[int, Tuple[int, ...]]
        Axis (or axes) to squeeze.

    Raises
    ------
    ``ValueError``, if an axis is selected with shape entry greater than one.
    """
    return Array._new(ht.squeeze(x._array, axis=axis))


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays : Union[Tuple[array, ...], List[array]]
        Input arrays to join. Each array must have the same shape.
    axis : int
        Axis along which the arrays will be joined.
    """
    # Call result type here just to raise on disallowed type combinations
    result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    return Array._new(ht.stack(arrays, axis=axis))
