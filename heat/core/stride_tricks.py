"""
A collection of functions used for inferring or correcting things before major computation
"""

import itertools
import numpy as np
import torch

from typing import Tuple, Union


def broadcast_shape(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Infers, if possible, the broadcast output shape of two operands a and b. Inspired by stackoverflow post:
    https://stackoverflow.com/questions/24743753/test-if-an-array-is-broadcastable-to-a-shape

    Parameters
    ----------
    shape_a : Tuple[int,...]
        Shape of first operand
    shape_b : Tuple[int,...]
        Shape of second operand

    Raises
    -------
    ValueError
        If the two shapes cannot be broadcast.

    Examples
    --------
    >>> import heat as ht
    >>> ht.core.stride_tricks.broadcast_shape((5,4),(4,))
    (5, 4)
    >>> ht.core.stride_tricks.broadcast_shape((1,100,1),(10,1,5))
    (10, 100, 5)
    >>> ht.core.stride_tricks.broadcast_shape((8,1,6,1),(7,1,5,))
    (8,7,6,5))
    >>> ht.core.stride_tricks.broadcast_shape((2,1),(8,4,3))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "heat/core/stride_tricks.py", line 42, in broadcast_shape
        "operands could not be broadcast, input shapes {} {}".format(shape_a, shape_b)
    ValueError: operands could not be broadcast, input shapes (2, 1) (8, 4, 3)
    """
    try:
        resulting_shape = torch.broadcast_shapes(shape_a, shape_b)
    except AttributeError:  # torch < 1.8
        it = itertools.zip_longest(shape_a[::-1], shape_b[::-1], fillvalue=1)
        resulting_shape = max(len(shape_a), len(shape_b)) * [None]
        for i, (a, b) in enumerate(it):
            if a == 0 and b == 1 or b == 0 and a == 1:
                resulting_shape[i] = 0
            elif a == 1 or b == 1 or a == b:
                resulting_shape[i] = max(a, b)
            else:
                raise ValueError(
                    f"operands could not be broadcast, input shapes {shape_a} {shape_b}"
                )
        return tuple(resulting_shape[::-1])
    except TypeError:
        raise TypeError(f"operand 1 must be tuple of ints, not {type(shape_a)}")
    except NameError:
        raise TypeError(f"operands must be tuples of ints, not {shape_a} and {shape_b}")
    except RuntimeError:
        raise ValueError(f"operands could not be broadcast, input shapes {shape_a} {shape_b}")

    return tuple(resulting_shape)


def sanitize_axis(
    shape: Tuple[int, ...], axis: Union[int, None, Tuple[int, ...]]
) -> Union[int, None, Tuple[int, ...]]:
    """
    Checks conformity of an axis with respect to a given shape. The axis will be converted to its positive equivalent
    and is checked to be within bounds

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of an array
    axis : ints or Tuple[int, ...] or None
        The axis to be sanitized

    Raises
    -------
    ValueError
        if the axis cannot be sanitized, i.e. out of bounds.
    TypeError
        if the axis is not integral.

    Examples
    -------
    >>> import heat as ht
    >>> ht.core.stride_tricks.sanitize_axis((5,4,4),1)
    1
    >>> ht.core.stride_tricks.sanitize_axis((5,4,4),-1)
    2
    >>> ht.core.stride_tricks.sanitize_axis((5, 4), (1,))
    (1,)
    >>> ht.core.stride_tricks.sanitize_axis((5, 4), 1.0)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "heat/heat/core/stride_tricks.py", line 99, in sanitize_axis
        raise TypeError("axis must be None or int or tuple, but was {}".format(type(axis)))
    TypeError: axis must be None or int or tuple, but was <class 'float'>

    """
    # scalars are handled like unsplit matrices
    if len(shape) == 0:
        axis = None

    if axis is not None and not isinstance(axis, int) and not isinstance(axis, tuple):
        raise TypeError(f"axis must be None or int or tuple, but was {type(axis)}")
    if isinstance(axis, tuple):
        axis = tuple(dim + len(shape) if dim < 0 else dim for dim in axis)
        for dim in axis:
            if dim < 0 or dim >= len(shape):
                raise ValueError(f"axis {axis} is out of bounds for shape {shape}")
        return axis

    if axis is None or 0 <= axis < len(shape):
        return axis
    elif axis < 0:
        axis += len(shape)

    if axis < 0 or axis >= len(shape):
        raise ValueError(f"axis {axis} is out of bounds for shape {shape}")

    return axis


def sanitize_shape(shape: Union[int, Tuple[int, ...]], lval: int = 0) -> Tuple[int, ...]:
    """
    Verifies and normalizes the given shape.

    Parameters
    ----------
    shape : int or Tupe[int,...]
        Shape of an array.
    lval : int
        Lowest legal value

    Raises
    -------
    ValueError
        If the shape contains illegal values, e.g. negative numbers.
    TypeError
        If the given shape is neither and int or a sequence of ints.

    Examples
    --------
    >>> import heat as ht
    >>> ht.core.stride_tricks.sanitize_shape(3)
    (3,)
    >>> ht.core.stride_tricks.sanitize_shape([1, 2, 3])
    (1, 2, 3,)
    >>> ht.core.stride_tricks.sanitize_shape(1.0)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "heat/heat/core/stride_tricks.py", line 159, in sanitize_shape
        raise TypeError("expected sequence object with length >= 0 or a single integer")
    TypeError: expected sequence object with length >= 0 or a single integer
    """
    shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)

    for dimension in shape:
        if issubclass(type(dimension), np.integer):
            dimension = int(dimension)
        if not isinstance(dimension, int):
            raise TypeError("expected sequence object with length >= 0 or a single integer")
        if dimension < lval:
            raise ValueError("negative dimensions are not allowed")

    return shape


def sanitize_slice(sl: slice, max_dim: int) -> slice:
    """
    Remove None-types from a slice

    Parameters
    ----------
    sl : slice
        slice to adjust
    max_dim : int
        maximum index for the given slice

    Raises
    ------
    TypeError
        if sl is not a slice.
    """
    if not isinstance(sl, slice):
        raise TypeError("This function is only for slices!")

    new_sl = [None] * 3
    new_sl[0] = 0 if sl.start is None else sl.start
    if new_sl[0] < 0:
        new_sl[0] += max_dim

    new_sl[1] = max_dim if sl.stop is None else sl.stop
    if new_sl[1] < 0:
        new_sl[1] += max_dim

    new_sl[2] = 1 if sl.step is None else sl.step

    return slice(new_sl[0], new_sl[1], new_sl[2])
