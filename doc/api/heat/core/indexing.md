Module heat.core.indexing
=========================
Functions relating to indices of items within DNDarrays, i.e. `where()`

Functions
---------

`nonzero(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Return a :class:`~heat.core.dndarray.DNDarray` containing the indices of the elements that are non-zero.. (using ``torch.nonzero``)
    If ``x`` is split then the result is split in the 0th dimension. However, this :class:`~heat.core.dndarray.DNDarray`
    can be UNBALANCED as it contains the indices of the non-zero elements on each node.
    Returns an array with one entry for each dimension of ``x``, containing the indices of the non-zero elements in that dimension.
    The values in ``x`` are always tested and returned in row-major, C-style order.
    The corresponding non-zero values can be obtained with: ``x[nonzero(x)]``.

    Parameters
    ----------
    x: DNDarray
        Input array

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.array([[3, 0, 0], [0, 4, 1], [0, 6, 0]], split=0)
    >>> ht.nonzero(x)
    DNDarray([[0, 0],
              [1, 1],
              [1, 2],
              [2, 1]], dtype=ht.int64, device=cpu:0, split=0)
    >>> y = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=0)
    >>> y > 3
    DNDarray([[False, False, False],
              [ True,  True,  True],
              [ True,  True,  True]], dtype=ht.bool, device=cpu:0, split=0)
    >>> ht.nonzero(y > 3)
    DNDarray([[1, 0],
              [1, 1],
              [1, 2],
              [2, 0],
              [2, 1],
              [2, 2]], dtype=ht.int64, device=cpu:0, split=0)
    >>> y[ht.nonzero(y > 3)]
    DNDarray([4, 5, 6, 7, 8, 9], dtype=ht.int64, device=cpu:0, split=0)

`where(cond: heat.core.dndarray.DNDarray, x: int | float | heat.core.dndarray.DNDarray | None = None, y: int | float | heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return a :class:`~heat.core.dndarray.DNDarray` containing elements chosen from ``x`` or ``y`` depending on condition.
    Result is a :class:`~heat.core.dndarray.DNDarray` with elements from ``x`` where cond is ``True``,
    and elements from ``y`` elsewhere (``False``).

    Parameters
    ----------
    cond : DNDarray
        Condition of interest, where true yield ``x`` otherwise yield ``y``
    x : DNDarray or int or float, optional
        Values from which to choose. ``x``, ``y`` and condition need to be broadcastable to some shape.
    y : DNDarray or int or float, optional
        Values from which to choose. ``x``, ``y`` and condition need to be broadcastable to some shape.

    Raises
    ------
    NotImplementedError
        if splits of the two input :class:`~heat.core.dndarray.DNDarray` differ
    TypeError
        if only x or y is given or both are not DNDarrays or numerical scalars

    Notes
    -----
    When only condition is provided, this function is a shorthand for :func:`nonzero`.

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.arange(10, split=0)
    >>> ht.where(x < 5, x, 10 * x)
    DNDarray([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90], dtype=ht.int64, device=cpu:0, split=0)
    >>> y = ht.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    >>> ht.where(y < 4, y, -1)
    DNDarray([[ 0,  1,  2],
              [ 0,  2, -1],
              [ 0,  3, -1]], dtype=ht.int64, device=cpu:0, split=None)
