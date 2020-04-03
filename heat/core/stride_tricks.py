import itertools
import numpy as np


def broadcast_shape(shape_a, shape_b):
    """
    Infers, if possible, the broadcast output shape of two operands a and b. Inspired by stackoverflow post:
    https://stackoverflow.com/questions/24743753/test-if-an-array-is-broadcastable-to-a-shape

    Parameters
    ----------
    shape_a : tuple of ints
        shape of operand a
    shape_b : tuple of ints
        shape of operand b

    Returns
    -------
    broadcast_shape : tuple of ints
        the broadcast shape

    Raises
    -------
    ValueError
        If the two shapes cannot be broadcast.

    Examples
    -------
    >>> broadcast_shape((5,4),(4,))
    (5,4)

    >>> broadcast_shape((1,100,1),(10,1,5))
    (10,100,5)

    >>> broadcast_shape((8,1,6,1),(7,1,5,))
    (8,7,6,5))

    >>> broadcast_shape((2,1),(8,4,3))
    ValueError
    """

    it = itertools.zip_longest(shape_a[::-1], shape_b[::-1], fillvalue=1)
    resulting_shape = max(len(shape_a), len(shape_b)) * [None]
    for i, (a, b) in enumerate(it):
        if a == 1 or b == 1 or a == b:
            resulting_shape[i] = max(a, b)
        else:
            raise ValueError(
                "operands could not be broadcast, input shapes {} {}".format(shape_a, shape_b)
            )

    return tuple(resulting_shape[::-1])


def sanitize_axis(shape, axis):
    """
    Checks conformity of an axis with respect to a given shape. The axis will be converted to its positive equivalent
    and is checked to be within bounds

    Parameters
    ----------
    shape : tuple of ints
        shape of an array
    axis : ints or tuple of ints
        the axis to be sanitized

    Returns
    -------
    sane_axis : int or tuple of ints
        the sane axis

    Raises
    -------
    ValueError
        if the axis cannot be sanitized, i.e. out of bounds.
    TypeError
        if the the axis is not integral.

    Examples
    -------
    >>> sanitize_axis((5,4,4),1)
    1

    >>> sanitize_axis((5,4,4),-1)
    2

    >>> sanitize_axis((5, 4), (1,))
    (1,)

    >>> sanitize_axis((5, 4), 1.0)
    TypeError
    """
    # scalars are handled like unsplit matrices
    if len(shape) == 0:
        axis = None

    if axis is not None:
        if not isinstance(axis, int) and not isinstance(axis, tuple):
            raise TypeError("axis must be None or int or tuple, but was {}".format(type(axis)))
    if isinstance(axis, tuple):
        axis = tuple(dim + len(shape) if dim < 0 else dim for dim in axis)
        for dim in axis:
            if dim < 0 or dim >= len(shape):
                raise ValueError("axis {} is out of bounds for shape {}".format(axis, shape))
        return axis

    if axis is None or 0 <= axis < len(shape):
        return axis
    elif axis < 0:
        axis += len(shape)

    if axis < 0 or axis >= len(shape):
        raise ValueError("axis {} is out of bounds for shape {}".format(axis, shape))

    return axis


def sanitize_shape(shape):
    """
    Verifies and normalizes the given shape.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of an array.

    Returns
    -------
    sane_shape : tuple of ints
        The sanitized shape.

    Raises
    -------
    ValueError
        If the shape contains illegal values, e.g. negative numbers.
    TypeError
        If the given shape is neither and int or a sequence of ints.

    Examples
    --------
    >>> sanitize_shape(3)
    (3,)

    >>> sanitize_shape([1, 2, 3])
    (1, 2, 3,)

    >>> sanitize_shape(1.0)
    TypeError
    """
    shape = (shape,) if not hasattr(shape, "__iter__") else tuple(shape)

    for dimension in shape:
        if issubclass(type(dimension), np.integer):
            dimension = int(dimension)
        if not isinstance(dimension, int):
            raise TypeError("expected sequence object with length >= 0 or a single integer")
        if dimension < 0:
            raise ValueError("negative dimensions are not allowed")

    return shape
