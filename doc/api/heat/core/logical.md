Module heat.core.logical
========================
Logical functions for the DNDarrays

Functions
---------

`all(x: heat.core.dndarray.DNDarray, axis: int | Tuple[int] | None = None, out: heat.core.dndarray.DNDarray | None = None, keepdims: bool = False) ‑> heat.core.dndarray.DNDarray | bool`
:   Test whether all array elements along a given axis evaluate to ``True``.
    A new boolean or :class:`~heat.core.dndarray.DNDarray` is returned unless out is specified, in which case a
    reference to ``out`` is returned.

    Parameters
    ----------
    x : DNDarray
        Input array or object that can be converted to an array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which a logical AND reduction is performed. The default (``axis=None``) is to perform a
        logical AND over all the dimensions of the input array. ``axis`` may be negative, in which case it counts
        from the last to the first axis.
    out : DNDarray, optional
        Alternate output array in which to place the result. It must have the same shape as the expected output
        and its type is preserved.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
    --------
    >>> x = ht.random.randn(4, 5)
    >>> x
    DNDarray([[ 0.7199,  1.3718,  1.5008,  0.3435,  1.2884],
              [ 0.1532, -0.0968,  0.3739,  1.7843,  0.5614],
              [ 1.1522,  1.9076,  1.7638,  0.4110, -0.2803],
              [-0.5475, -0.0271,  0.8564, -1.5870,  1.3108]], dtype=ht.float32, device=cpu:0, split=None)
    >>> y = x < 0.5
    >>> y
    DNDarray([[False, False, False,  True, False],
              [ True,  True,  True, False, False],
              [False, False, False,  True,  True],
              [ True,  True, False,  True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(y)
    DNDarray([False], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(y, axis=0)
    DNDarray([False, False, False, False, False], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(x, axis=1)
    DNDarray([True, True, True, True], dtype=ht.bool, device=cpu:0, split=None)
    >>> out = ht.zeros(5)
    >>> ht.all(y, axis=0, out=out)
    DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)
    >>> out
    DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)

`allclose(x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) ‑> bool`
:   Test whether two tensors are element-wise equal within a tolerance. Returns ``True`` if ``|x-y|<=atol+rtol*|y|``
    for all elements of ``x`` and ``y``, ``False`` otherwise

    Parameters
    ----------
    x : DNDarray
        First array to compare
    y : DNDarray
        Second array to compare
    atol: float, optional
        Absolute tolerance.
    rtol: float, optional
        Relative tolerance (with respect to ``y``).
    equal_nan: bool, optional
        Whether to compare NaN’s as equal. If ``True``, NaN’s in ``x`` will be considered equal to NaN’s in ``y`` in
        the output array.

    Examples
    --------
    >>> x = ht.float32([[2, 2], [2, 2]])
    >>> ht.allclose(x, x)
    True
    >>> y = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])
    >>> ht.allclose(x, y)
    False
    >>> ht.allclose(x, y, atol=1e-04)
    True

`any(x, axis: int | None = None, out: heat.core.dndarray.DNDarray | None = None, keepdims: bool = False) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` containing the result of the test whether any array elements along a
    given axis evaluate to ``True``.
    The returning array is one dimensional unless axis is not ``None``.

    Parameters
    ----------
    x : DNDarray
        Input tensor
    axis : int, optional
        Axis along which a logic OR reduction is performed. With ``axis=None``, the logical OR is performed over all
        dimensions of the array.
    out : DNDarray, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output.
        The output is a array with ``datatype=bool``.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
    --------
    >>> x = ht.float32([[0.3, 0, 0.5]])
    >>> x.any()
    DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
    >>> x.any(axis=0)
    DNDarray([ True, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    >>> x.any(axis=1)
    DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.int32([[0, 0, 1], [0, 0, 0]])
    >>> res = ht.zeros(3, dtype=ht.bool)
    >>> y.any(axis=0, out=res)
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    >>> res
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)

`isclose(x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) ‑> heat.core.dndarray.DNDarray`
:   Returns a boolean :class:`~heat.core.dndarray.DNDarray`, with elements ``True`` where ``a`` and ``b`` are equal
    within the given tolerance. If both ``x`` and ``y`` are scalars, returns a single boolean value.

    Parameters
    ----------
    x : DNDarray
        Input array to compare.
    y : DNDarray
        Input array to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan : bool
        Whether to compare NaN’s as equal. If ``True``, NaN’s in x will be considered equal to NaN’s in y in the output
        array.

`isfinite(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Test element-wise for finiteness (not infinity or not Not a Number) and return result as a boolean
    :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input tensor

    Examples
    --------
    >>> ht.isfinite(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([ True, False, False, False], dtype=ht.bool, device=cpu:0, split=None)

`isinf(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Test element-wise for positive or negative infinity and return result as a boolean
    :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input tensor

    Examples
    --------
    >>> ht.isinf(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False,  True,  True, False], dtype=ht.bool, device=cpu:0, split=None)

`isnan(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Test element-wise for NaN and return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, False, False,  True], dtype=ht.bool, device=cpu:0, split=None)

`isneginf(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Test if each element of `x` is negative infinite, return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor
    out : DNDarray, optional
          Alternate output array in which to place the result. It must have the same shape as the expected output
          and its type is preserved.

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, False, True, False], dtype=ht.bool, device=cpu:0, split=None)

`isposinf(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None)`
:   Test if each element of `x` is positive infinite, return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor
    out : DNDarray, optional
          Alternate output array in which to place the result. It must have the same shape as the expected output
          and its type is preserved.

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, True, False, False], dtype=ht.bool, device=cpu:0, split=None)

`logical_and(x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Compute the truth value of ``x`` AND ``y`` element-wise. Returns a boolean :class:`~heat.core.dndarray.DNDarray` containing the truth value of ``x`` AND ``y`` element-wise.

    Parameters
    ----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    --------
    >>> ht.logical_and(ht.array([True, False]), ht.array([False, False]))
    DNDarray([False, False], dtype=ht.bool, device=cpu:0, split=None)

`logical_not(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Computes the element-wise logical NOT of the given input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input array
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the expected output.
        The output is a :class:`~heat.core.dndarray.DNDarray` with ``datatype=bool``.

    Examples
    --------
    >>> ht.logical_not(ht.array([True, False]))
    DNDarray([False,  True], dtype=ht.bool, device=cpu:0, split=None)

`logical_or(x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Returns boolean :class:`~heat.core.dndarray.DNDarray` containing the element-wise logical NOT of the given
    input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    --------
    >>> ht.logical_or(ht.array([True, False]), ht.array([False, False]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)

`logical_xor(x: heat.core.dndarray.DNDarray, y: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Computes the element-wise logical XOR of the given input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    --------
    >>> ht.logical_xor(ht.array([True, False, True]), ht.array([True, False, False]))
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)

`signbit(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Checks if signbit is set element-wise (less than zero).

    Parameters
    ----------
    x : DNDarray
        The input array.
    out : DNDarray, optional
        The output array.

    Examples
    --------
    >>> a = ht.array([2, -1.3, 0])
    >>> ht.signbit(a)
    DNDarray([False,  True, False], dtype=ht.bool, device=cpu:0, split=None)
