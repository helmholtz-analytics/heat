import itertools


def sanitize_axis(shape, axis):
    """
    Checks conformity of an axis with respect to a given shape. The axis will be converted to its positive equivalent
    and is checked to be within bounds

    Parameters
    ----------
    shape : tuple of ints
        shape of an array
    axis : ints
        the axis to be sanitized

    Returns
    -------
    sane_axis : int
        the sane axis

    Raises
    -------
    ValueError
        If the axis cannot be sanitized, i.e. out of bounds.
    """
    #TODO: test me
    if axis is None or 0 <= axis < len(shape):
        return axis
    elif axis < 0:
        axis += len(shape)

    if axis < 0 or axis >= len(shape):
        raise ValueError('axis axis {} is out of bounds for shape {}'.format(axis, shape))
    return axis


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
    """
    #TODO: test me
    it = itertools.zip_longest(shape_a[::-1], shape_b[::-1], fillvalue=1)
    resulting_shape = max(len(shape_a), len(shape_b)) * [None]
    for i, (a, b) in enumerate(it):
        if a == 1 or b == 1 or a == b:
            resulting_shape[i] = max(a, b)
        else:
            raise ValueError('operands could not be broadcast, input shapes {} {}'.format(shape_a, shape_b))

    return tuple(resulting_shape[::-1])
