Module heat.core.relational
===========================
Functions for relational oprations, i.e. equal/no equal...

Functions
---------

`eq(x, y) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise comparision.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    compared as argument.
    Returns False if the operands are not scalars or :class:`~heat.core.dndarray.DNDarray`

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand involved in the comparison
    y: DNDarray or scalar
        The second operand involved in the comparison

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.eq(x, 3.0)
    DNDarray([[False, False],
              [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.eq(x, y)
    DNDarray([[False,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.eq(x, slice(None))
    False

`equal(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> bool`
:   Overall comparison of equality between two :class:`~heat.core.dndarray.DNDarray`. Returns ``True`` if two arrays
    have the same size and elements, and ``False`` otherwise.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand involved in the comparison
    y: DNDarray or scalar
        The second operand involved in the comparison

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.equal(x, ht.float32([[1, 2], [3, 4]]))
    True
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.equal(x, y)
    False
    >>> ht.equal(x, 3.0)
    False

`ge(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Returns a D:class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich greater than or equal comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x>=y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand to be compared greater than or equal to second operand
    y: DNDarray or scalar
       The second operand to be compared less than or equal to first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.ge(x, 3.0)
    DNDarray([[False, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.ge(x, y)
    DNDarray([[False,  True],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)

`greater(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich greater than comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x>y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
       The first operand to be compared greater than second operand
    y: DNDarray or scalar
       The second operand to be compared less than first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.gt(x, 3.0)
    DNDarray([[False, False],
              [False,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.gt(x, y)
    DNDarray([[False, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)

`greater_equal(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Returns a D:class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich greater than or equal comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x>=y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand to be compared greater than or equal to second operand
    y: DNDarray or scalar
       The second operand to be compared less than or equal to first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.ge(x, 3.0)
    DNDarray([[False, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.ge(x, y)
    DNDarray([[False,  True],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)

`gt(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich greater than comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x>y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
       The first operand to be compared greater than second operand
    y: DNDarray or scalar
       The second operand to be compared less than first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.gt(x, 3.0)
    DNDarray([[False, False],
              [False,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.gt(x, y)
    DNDarray([[False, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)

`le(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Return a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich less than or equal comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x<=y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
       The first operand to be compared less than or equal to second operand
    y: DNDarray or scalar
       The second operand to be compared greater than or equal to first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.le(x, 3.0)
    DNDarray([[ True,  True],
              [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.le(x, y)
    DNDarray([[ True,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)

`less(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich less than comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x<y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand to be compared less than second operand
    y: DNDarray or scalar
        The second operand to be compared greater than first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.lt(x, 3.0)
    DNDarray([[ True,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.lt(x, y)
    DNDarray([[ True, False],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)

`less_equal(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Return a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich less than or equal comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x<=y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
       The first operand to be compared less than or equal to second operand
    y: DNDarray or scalar
       The second operand to be compared greater than or equal to first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.le(x, 3.0)
    DNDarray([[ True,  True],
              [ True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.le(x, y)
    DNDarray([[ True,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)

`lt(x: Union[DNDarray, float, int], y: Union[DNDarray, float, int]) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich less than comparison between values from operand ``x`` with respect to values of
    operand ``y`` (i.e. ``x<y``), not commutative. Takes the first and second operand (scalar or
    :class:`~heat.core.dndarray.DNDarray`) whose elements are to be compared as argument.

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand to be compared less than second operand
    y: DNDarray or scalar
        The second operand to be compared greater than first operand

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.lt(x, 3.0)
    DNDarray([[ True,  True],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.lt(x, y)
    DNDarray([[ True, False],
              [False, False]], dtype=ht.bool, device=cpu:0, split=None)

`ne(x, y) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich comparison of non-equality between values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    compared as argument.
    Returns True if the operands are not scalars or :class:`~heat.core.dndarray.DNDarray`

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand involved in the comparison
    y: DNDarray or scalar
        The second operand involved in the comparison

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.ne(x, 3.0)
    DNDarray([[ True,  True],
              [False,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.ne(x, y)
    DNDarray([[ True, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.ne(x, slice(None))
    True

`not_equal(x, y) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the results of element-wise rich comparison of non-equality between values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    compared as argument.
    Returns True if the operands are not scalars or :class:`~heat.core.dndarray.DNDarray`

    Parameters
    ----------
    x: DNDarray or scalar
        The first operand involved in the comparison
    y: DNDarray or scalar
        The second operand involved in the comparison

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.float32([[1, 2], [3, 4]])
    >>> ht.ne(x, 3.0)
    DNDarray([[ True,  True],
              [False,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.float32([[2, 2], [2, 2]])
    >>> ht.ne(x, y)
    DNDarray([[ True, False],
              [ True,  True]], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.ne(x, slice(None))
    True
