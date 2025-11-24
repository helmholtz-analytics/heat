Module heat.core.stride_tricks
==============================
A collection of functions used for inferring or correcting things before major computation

Functions
---------

`broadcast_shape(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) ‑> Tuple[int, ...]`
:   Infers, if possible, the broadcast output shape of two operands a and b. Inspired by stackoverflow post:
    https://stackoverflow.com/questions/24743753/test-if-an-array-is-broadcastable-to-a-shape

    Parameters
    ----------
    shape_a : Tuple[int,...]
        Shape of first operand
    shape_b : Tuple[int,...]
        Shape of second operand

    Raises
    ------
    ValueError
        If the two shapes cannot be broadcast.

    Examples
    --------
    >>> import heat as ht
    >>> ht.core.stride_tricks.broadcast_shape((5, 4), (4,))
    (5, 4)
    >>> ht.core.stride_tricks.broadcast_shape((1, 100, 1), (10, 1, 5))
    (10, 100, 5)
    >>> ht.core.stride_tricks.broadcast_shape(
    ...     (8, 1, 6, 1),
    ...     (
    ...         7,
    ...         1,
    ...         5,
    ...     ),
    ... )
    (8,7,6,5))
    >>> ht.core.stride_tricks.broadcast_shape((2, 1), (8, 4, 3))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "heat/core/stride_tricks.py", line 42, in broadcast_shape
        "operands could not be broadcast, input shapes {} {}".format(shape_a, shape_b)
    ValueError: operands could not be broadcast, input shapes (2, 1) (8, 4, 3)

`broadcast_shapes(*shapes: Tuple[int, ...]) ‑> Tuple[int, ...]`
:   Infers, if possible, the broadcast output shape of multiple operands.

    Parameters
    ----------
    *shapes : Tuple[int,...]
        Shapes of operands.

    Returns
    -------
    Tuple[int, ...]
        The broadcast output shape.

    Raises
    ------
    ValueError
        If the shapes cannot be broadcast.

    Examples
    --------
    >>> import heat as ht
    >>> ht.broadcast_shapes((5, 4), (4,))
    (5, 4)
    >>> ht.broadcast_shapes((1, 100, 1), (10, 1, 5))
    (10, 100, 5)
    >>> ht.broadcast_shapes(
    ...     (8, 1, 6, 1),
    ...     (
    ...         7,
    ...         1,
    ...         5,
    ...     ),
    ... )
    (8,7,6,5))
    >>> ht.broadcast_shapes((2, 1), (8, 4, 3))
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "heat/core/stride_tricks.py", line 100, in broadcast_shapes
        "operands could not be broadcast, input shapes {}".format(shapes))
    ValueError: operands could not be broadcast, input shapes ((2, 1), (8, 4, 3))

`sanitize_axis(shape: Tuple[int, ...], axis: int | Tuple[int, ...] | None) ‑> int | Tuple[int, ...] | None`
:   Checks conformity of an axis with respect to a given shape. The axis will be converted to its positive equivalent
    and is checked to be within bounds

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of an array
    axis : ints or Tuple[int, ...] or None
        The axis to be sanitized

    Raises
    ------
    ValueError
        if the axis cannot be sanitized, i.e. out of bounds.
    TypeError
        if the axis is not integral.

    Examples
    --------
    >>> import heat as ht
    >>> ht.core.stride_tricks.sanitize_axis((5, 4, 4), 1)
    1
    >>> ht.core.stride_tricks.sanitize_axis((5, 4, 4), -1)
    2
    >>> ht.core.stride_tricks.sanitize_axis((5, 4), (1,))
    (1,)
    >>> ht.core.stride_tricks.sanitize_axis((5, 4), 1.0)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "heat/heat/core/stride_tricks.py", line 99, in sanitize_axis
        raise TypeError("axis must be None or int or tuple, but was {}".format(type(axis)))
    TypeError: axis must be None or int or tuple, but was <class 'float'>

`sanitize_shape(shape: int | Tuple[int, ...], lval: int = 0) ‑> Tuple[int, ...]`
:   Verifies and normalizes the given shape.

    Parameters
    ----------
    shape : int or Tupe[int,...]
        Shape of an array.
    lval : int
        Lowest legal value

    Raises
    ------
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

`sanitize_slice(sl: slice, max_dim: int) ‑> slice`
:   Remove None-types from a slice

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
