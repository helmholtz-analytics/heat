Module heat.core.arithmetics
============================
Arithmetic functions for DNDarrays

Functions
---------

`add(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise addition of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose
    elements are to be added as argument and returns a ``DNDarray`` containing the results of
    element-wise addition of ``t1`` and ``t2``.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the addition
    t2: DNDarray or scalar
        The second operand involved in the addition
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the added value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Examples
    --------
    >>> import heat as ht
    >>> ht.add(1.0, 4.0)
    DNDarray(5., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.add(T1, T2)
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.add(T1, s)
    DNDarray([[3., 4.],
              [5., 6.]], dtype=ht.float32, device=cpu:0, split=None)

`bitwise_and(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Compute the bitwise AND of two :class:`~heat.core.dndarray.DNDarray` ``t1`` and ``t2``
    element-wise. Only integer and boolean types are handled. If ``t1.shape!=t2.shape``, they must
    be broadcastable to a common shape (which becomes the shape of the output)

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the added value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.bitwise_and(13, 17)
    DNDarray(1, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(14, 13)
    DNDarray(12, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([14, 3]), 13)
    DNDarray([12,  1], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([11, 7]), ht.array([4, 25]))
    DNDarray([0, 1], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([2, 5, 255]), ht.array([3, 14, 16]))
    DNDarray([ 2,  4, 16], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_and(ht.array([True, True]), ht.array([False, True]))
    DNDarray([False,  True], dtype=ht.bool, device=cpu:0, split=None)

`bitwise_not(a: DNDarray, /, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Computes the bitwise NOT of the given input :class:`~heat.core.dndarray.DNDarray`. The input
    array must be of integral or Boolean types. For boolean arrays, it computes the logical NOT.
    Bitwise_not is an alias for invert.

    Parameters
    ----------
    a: DNDarray
        The input array to invert. Must be of integral or Boolean types
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output. The dtype of the output will be the one of the input array, unless it is
        logical, in which case it will be casted to int8. If not provided or None, a freshly-
        allocated array is returned.

    Examples
    --------
    >>> ht.invert(ht.array([13], dtype=ht.uint8))
    DNDarray([242], dtype=ht.uint8, device=cpu:0, split=None)
    >>> ht.bitwise_not(ht.array([-1, -2, 3], dtype=ht.int8))
    DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)

`bitwise_or(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Compute the bit-wise OR of two :class:`~heat.core.dndarray.DNDarray` ``t1`` and ``t2``
    element-wise. Only integer and boolean types are handled. If ``t1.shape!=t2.shape``, they must
    be broadcastable to a common shape (which becomes the shape of the output)

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the added value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.bitwise_or(13, 16)
    DNDarray(29, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(32, 2)
    DNDarray(34, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([33, 4]), 1)
    DNDarray([33,  5], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([33, 4]), ht.array([1, 2]))
    DNDarray([33,  6], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([2, 5, 255]), ht.array([4, 4, 4]))
    DNDarray([  6,   5, 255], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([2, 5, 255, 2147483647], dtype=ht.int32),
                      ht.array([4, 4, 4, 2147483647], dtype=ht.int32))
    DNDarray([         6,          5,        255, 2147483647], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.bitwise_or(ht.array([True, True]), ht.array([False, True]))
    DNDarray([True, True], dtype=ht.bool, device=cpu:0, split=None)

`bitwise_xor(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Compute the bit-wise XOR of two arrays ``t1`` and ``t2`` element-wise.
    Only integer and boolean types are handled. If ``x1.shape!=x2.shape``, they must be
    broadcastable to a common shape (which becomes the shape of the output).

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the operation
    t2: DNDarray or scalar
        The second operand involved in the operation
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the added value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.bitwise_xor(13, 17)
    DNDarray(28, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(31, 5)
    DNDarray(26, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(ht.array([31, 3]), 5)
    DNDarray([26,  6], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(ht.array([31, 3]), ht.array([5, 6]))
    DNDarray([26,  5], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bitwise_xor(ht.array([True, True]), ht.array([False, True]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)

`copysign(a: DNDarray, b: Union[DNDarray, float, int], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Create a new floating-point tensor with the magnitude of 'a' and the sign of 'b', element-wise

    Parameters
    ----------
    a:  DNDarray
        The input array
    b:  DNDarray or Number
        value(s) whose signbit(s) are applied to the magnitudes in 'a'
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.copysign(ht.array([3, 2, -8, -2, 4]), 1)
    DNDarray([3, 2, 8, 2, 4], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.copysign(ht.array([3.0, 2.0, -8.0, -2.0, 4.0]), ht.array([1.0, -1.0, 1.0, -1.0, 1.0]))
    DNDarray([ 3., -2.,  8., -2.,  4.], dtype=ht.float32, device=cpu:0, split=None)

`cumprod(a: DNDarray, axis: int, dtype: datatype = None, out=None) ‑> heat.core.dndarray.DNDarray`
:   Return the cumulative product of elements along a given axis.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : int
        Axis along which the cumulative product is computed.
    dtype : datatype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If ``dtype`` is not specified, it
        defaults to the datatype of ``a``, unless ``a`` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Examples
    --------
    >>> a = ht.full((3, 3), 2)
    >>> ht.cumprod(a, 0)
    DNDarray([[2., 2., 2.],
            [4., 4., 4.],
            [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)

`cumproduct(a: DNDarray, axis: int, dtype: datatype = None, out=None) ‑> heat.core.dndarray.DNDarray`
:   Return the cumulative product of elements along a given axis.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : int
        Axis along which the cumulative product is computed.
    dtype : datatype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If ``dtype`` is not specified, it
        defaults to the datatype of ``a``, unless ``a`` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Examples
    --------
    >>> a = ht.full((3, 3), 2)
    >>> ht.cumprod(a, 0)
    DNDarray([[2., 2., 2.],
            [4., 4., 4.],
            [8., 8., 8.]], dtype=ht.float32, device=cpu:0, split=None)

`cumsum(a: DNDarray, axis: int, dtype: datatype = None, out=None) ‑> heat.core.dndarray.DNDarray`
:   Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : int
        Axis along which the cumulative sum is computed.
    dtype : datatype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If ``dtype`` is not specified, it defaults
        to the datatype of ``a``, unless ``a`` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.

    Examples
    --------
    >>> a = ht.ones((3, 3))
    >>> ht.cumsum(a, 0)
    DNDarray([[1., 1., 1.],
              [2., 2., 2.],
              [3., 3., 3.]], dtype=ht.float32, device=cpu:0, split=None)

`diff(a: DNDarray, n: int = 1, axis: int = -1, prepend: Union[int, float, DNDarray] = None, append: Union[int, float, DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Calculate the n-th discrete difference along the given axis.
    The first difference is given by ``out[i]=a[i+1]-a[i]`` along the given axis, higher differences
    are calculated by using diff recursively. The shape of the output is the same as ``a`` except
    along axis where the dimension is smaller by ``n``. The datatype of the output is the same as
    the datatype of the difference between any two elements of ``a``. The split does not change. The
    output array is balanced.

    Parameters
    ----------
    a : DNDarray
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
        ``n=2`` is equivalent to ``diff(diff(a))``
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    prepend : Union[int, float, DNDarray]
        Value to prepend along axis prior to performing the difference.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes. Otherwise the dimension and
        shape must match a except along axis.
    append : Union[int, float, DNDarray]
        Values to append along axis prior to performing the difference.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes. Otherwise the dimension and
        shape must match a except along axis.

`div(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise true division of values of operand ``t1`` by values of operands ``t2`` (i.e ``t1/t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided.
    t2: DNDarray or scalar
        The second operand by whose values is divided.
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the divided value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Example
    ---------
    >>> ht.div(2.0, 2.0)
    DNDarray(1., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.div(T1, T2)
    DNDarray([[0.5000, 1.0000],
              [1.5000, 2.0000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.div(s, T1)
    DNDarray([[2.0000, 1.0000],
              [0.6667, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)

`divide(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise true division of values of operand ``t1`` by values of operands ``t2`` (i.e ``t1/t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided.
    t2: DNDarray or scalar
        The second operand by whose values is divided.
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the divided value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Example
    ---------
    >>> ht.div(2.0, 2.0)
    DNDarray(1., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.div(T1, T2)
    DNDarray([[0.5000, 1.0000],
              [1.5000, 2.0000]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.div(s, T1)
    DNDarray([[2.0000, 1.0000],
              [0.6667, 0.5000]], dtype=ht.float32, device=cpu:0, split=None)

`divmod(t1: Union[DNDarray, float], t2: Union[DNDarray, float], out1: DNDarray = None, out2: DNDarray = None, /, out: Tuple[DNDarray, DNDarray] = (None, None), *, where: Union[bool, DNDarray] = True) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Element-wise division remainder and quotient from an integer division of values of operand
    ``t1`` by values of operand ``t2`` (i.e. C Library function divmod). Result has the sign as the
    dividend ``t1``. Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided (may be floats)
    t2: DNDarray or scalar
        The second operand by whose values is divided (may be floats)
    out1: DNDarray, optional
        The output array for the quotient. It must have a shape that the inputs broadcast to and
        matching split axis.
        If not provided, a freshly allocated array is returned. If provided, it must be of the same
        shape as the expected output. Only one of out1 and out can be provided.
    out2: DNDarray, optional
        The output array for the remainder. It must have a shape that the inputs broadcast to and
        matching split axis.
        If not provided, a freshly allocated array is returned. If provided, it must be of the same
        shape as the expected output. Only one of out2 and out can be provided.
    out: tuple of two DNDarrays, optional
        Tuple of two output arrays (quotient, remainder), respectively. Both must have a shape that
        the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned. If provided, they must be of the
        same shape as the expected output. out1 and out2 cannot be used at the same time.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out1`
        array will be set to the quotient value and the `out2` array will be set to the remainder
        value. Elsewhere, the `out1` and `out2` arrays will retain their original value. If an
        uninitialized `out1` and `out2` array is created via the default `out1=None` and
        `out2=None`, locations within them where the condition is False will remain uninitialized.
        If distributed, the split axis (after broadcasting if required) must match that of the
        `out1` and `out2` arrays.

    Examples
    --------
    >>> ht.divmod(2.0, 2.0)
    (DNDarray(1., dtype=ht.float32, device=cpu:0, split=None), DNDarray(0., dtype=ht.float32, device=cpu:0, split=None))
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.divmod(T1, T2)
    (DNDarray([[0., 1.],
               [1., 2.]], dtype=ht.float32, device=cpu:0, split=None), DNDarray([[1., 0.],
               [1., 0.]], dtype=ht.float32, device=cpu:0, split=None))
    >>> s = 2.0
    >>> ht.divmod(s, T1)
    (DNDarray([[2., 1.],
               [0., 0.]], dtype=ht.float32, device=cpu:0, split=None), DNDarray([[0., 0.],
               [2., 2.]], dtype=ht.float32, device=cpu:0, split=None))

`floor_divide(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise floor division of value(s) of operand ``t1`` by value(s) of operand ``t2``
    (i.e. ``t1//t2``), not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
    >>> ht.floordiv(T1, 1)
    DNDarray([[1., 2.],
              [1., 4.]], dtype=ht.float64, device=cpu:0, split=None)
    >>> T2 = ht.float32([1.5, 2.5])
    >>> ht.floordiv(T1, T2)
    DNDarray([[1., 0.],
              [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)

`floordiv(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise floor division of value(s) of operand ``t1`` by value(s) of operand ``t2``
    (i.e. ``t1//t2``), not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
    >>> ht.floordiv(T1, 1)
    DNDarray([[1., 2.],
              [1., 4.]], dtype=ht.float64, device=cpu:0, split=None)
    >>> T2 = ht.float32([1.5, 2.5])
    >>> ht.floordiv(T1, T2)
    DNDarray([[1., 0.],
              [1., 1.]], dtype=ht.float32, device=cpu:0, split=None)

`fmod(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise division remainder of values of operand ``t1`` by values of operand ``t2`` (i.e.
    C Library function fmod).
    Result has the sign as the dividend ``t1``. Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided (may be floats)
    t2: DNDarray or scalar
        The second operand by whose values is divided (may be floats)
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned. If provided, it must be of the same
        shape as the expected output.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.fmod(2.0, 2.0)
    DNDarray(0., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.fmod(T1, T2)
    DNDarray([[1., 0.],
          [1., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.fmod(s, T1)
    DNDarray([[0., 0.],
          [2., 2.]], dtype=ht.float32, device=cpu:0, split=None)

`gcd(a: DNDarray, b: DNDarray, /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Returns the greatest common divisor of |a| and |b| element-wise.

    Parameters
    ----------
    a:   DNDarray
         The first input array, must be of integer type
    b:   DNDarray
         the second input array, must be of integer type
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> import heat as ht
    >>> T1 = ht.int(ht.ones(3)) * 9
    >>> T2 = ht.arange(3) + 1
    >>> ht.gcd(T1, T2)
    DNDarray([1, 1, 3], dtype=ht.int32, device=cpu:0, split=None)

`hypot(t1: DNDarray, t2: DNDarray, /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Given the 'legs' of a right triangle, return its hypotenuse. Equivalent to
    :math:`sqrt(a^2 + b^2)`, element-wise.

    Parameters
    ----------
    t1:   DNDarray
         The first input array
    t2:   DNDarray
         the second input array
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> a = ht.array([2.0])
    >>> b = ht.array([1.0, 3.0, 3.0])
    >>> ht.hypot(a, b)
    DNDarray([2.2361, 3.6056, 3.6056], dtype=ht.float32, device=cpu:0, split=None)

`invert(a: DNDarray, /, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Computes the bitwise NOT of the given input :class:`~heat.core.dndarray.DNDarray`. The input
    array must be of integral or Boolean types. For boolean arrays, it computes the logical NOT.
    Bitwise_not is an alias for invert.

    Parameters
    ----------
    a: DNDarray
        The input array to invert. Must be of integral or Boolean types
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output. The dtype of the output will be the one of the input array, unless it is
        logical, in which case it will be casted to int8. If not provided or None, a freshly-
        allocated array is returned.

    Examples
    --------
    >>> ht.invert(ht.array([13], dtype=ht.uint8))
    DNDarray([242], dtype=ht.uint8, device=cpu:0, split=None)
    >>> ht.bitwise_not(ht.array([-1, -2, 3], dtype=ht.int8))
    DNDarray([ 0,  1, -4], dtype=ht.int8, device=cpu:0, split=None)

`lcm(a: DNDarray, b: DNDarray, /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Returns the lowest common multiple of |a| and |b| element-wise.

    Parameters
    ----------
    a:   DNDarray or scalar
         The first input (array), must be of integer type
    b:   DNDarray or scalar
         the second input (array), must be of integer type
    out: DNDarray, optional
        The output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> a = ht.array([6, 12, 15])
    >>> b = ht.array([3, 4, 5])
    >>> ht.lcm(a, b)
    DNDarray([ 6, 12, 15], dtype=ht.int64, device=cpu:0, split=None)
    >>> s = 2
    >>> ht.lcm(s, a)
    DNDarray([ 6, 12, 30], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.lcm(b, s)
    DNDarray([ 6,  4, 10], dtype=ht.int64, device=cpu:0, split=None)

`left_shift(t1: DNDarray, t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Shift the bits of an integer to the left.

    Parameters
    ----------
    t1: DNDarray
        Input array
    t2: DNDarray or float
        Integer number of zero bits to add
    out: DNDarray, optional
        Output array for the result. Must have the same shape as the expected output. The dtype of
        the output will be the one of the input array, unless it is logical, in which case it will
        be casted to int8. If not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the shifted value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.left_shift(ht.array([1, 2, 3]), 1)
    DNDarray([2, 4, 6], dtype=ht.int64, device=cpu:0, split=None)

`mod(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise division remainder of values of operand ``t1`` by values of operand ``t2`` (i.e.
    ``t1%t2``). Result has the same sign as the divisor ``t2``.
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.remainder(2, 2)
    DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
    >>> T1 = ht.int32([[1, 2], [3, 4]])
    >>> T2 = ht.int32([[2, 2], [2, 2]])
    >>> ht.remainder(T1, T2)
    DNDarray([[1, 0],
            [1, 0]], dtype=ht.int32, device=cpu:0, split=None)
    >>> s = 2
    >>> ht.remainder(s, T1)
    DNDarray([[0, 0],
            [2, 2]], dtype=ht.int32, device=cpu:0, split=None)

`mul(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise multiplication (NOT matrix multiplication) of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    multiplied as argument.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the multiplication
    t2: DNDarray or scalar
        The second operand involved in the multiplication
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If not provided or
        None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the multiplied value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.mul(2.0, 4.0)
    DNDarray(8., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> s = 3.0
    >>> ht.mul(T1, s)
    DNDarray([[ 3.,  6.],
              [ 9., 12.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.mul(T1, T2)
    DNDarray([[2., 4.],
              [6., 8.]], dtype=ht.float32, device=cpu:0, split=None)

`multiply(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise multiplication (NOT matrix multiplication) of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.core.dndarray.DNDarray`) whose elements are to be
    multiplied as argument.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand involved in the multiplication
    t2: DNDarray or scalar
        The second operand involved in the multiplication
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If not provided or
        None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the multiplied value. Elsewhere, the `out` array will retain its original value. If
        an uninitialized `out` array is created via the default `out=None`, locations within it where the
        condition is False will remain uninitialized. If distributed, the split axis (after broadcasting
        if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.mul(2.0, 4.0)
    DNDarray(8., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> s = 3.0
    >>> ht.mul(T1, s)
    DNDarray([[ 3.,  6.],
              [ 9., 12.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.mul(T1, T2)
    DNDarray([[2., 4.],
              [6., 8.]], dtype=ht.float32, device=cpu:0, split=None)

`nan_to_num(a: DNDarray, nan: float = 0.0, posinf: float = None, neginf: float = None, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Replaces NaNs, positive infinity values, and negative infinity values in the input 'a' with the
    values specified by nan, posinf, and neginf, respectively. By default, NaNs are replaced with
    zero, positive infinity is replaced with the greatest finite value representable by input's
    dtype, and negative infinity is replaced with the least finite value representable by input's
    dtype.

    Parameters
    ----------
    a : DNDarray
        Input array.
    nan : float, optional
        Value to be used to replace NaNs. Default value is 0.0.
    posinf : float, optional
        Value to replace positive infinity values with. If None, positive infinity values are
        replaced with the greatest finite value of the input's dtype. Default value is None.
    neginf : float, optional
        Value to replace negative infinity values with. If None, negative infinity values are
        replaced with the greatest negative finite value of the input's dtype. Default value is
        None.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.

    Examples
    --------
    >>> x = ht.array([float("nan"), float("inf"), -float("inf")])
    >>> ht.nan_to_num(x)
    DNDarray([ 0.0000e+00,  3.4028e+38, -3.4028e+38], dtype=ht.float32, device=cpu:0, split=None)

`nanprod(a: DNDarray, axis: Union[int, Tuple[int, ...]] = None, out: DNDarray = None, keepdims: bool = None) ‑> heat.core.dndarray.DNDarray`
:   Return the product of array elements over a given axis treating Not a Numbers (NaNs) as one.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which a product is performed. The default, ``axis=None``, will calculate
        the product of all the elements in the input array. If axis is negative it counts from the
        last to the first axis.
        If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple
        instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input array.

    Examples
    --------
    >>> ht.nanprod(ht.array([4.0, ht.nan]))
    DNDarray(4., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.nanprod(ht.array([
        [1.,ht.nan],
        [3.,4.]]))
    DNDarray(12., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.nanprod(ht.array([
        [1.,ht.nan],
        [ht.nan,4.]
    ]), axis=1)
    DNDarray([ 1., 4.], dtype=ht.float32, device=cpu:0, split=None)

`nansum(a: DNDarray, axis: Union[int, Tuple[int, ...]] = None, out: DNDarray = None, keepdims: bool = None) ‑> heat.core.dndarray.DNDarray`
:   Sum of array elements over a given axis treating Not a Numbers (NaNs) as zero. An array with the
    same shape as ``self.__array`` except for the specified axis which becomes one, e.g.
    ``a.shape=(1, 2, 3)`` => ``ht.ones((1, 2, 3)).sum(axis=1).shape=(1, 1, 3)``

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis along which a sum is performed. The default, ``axis=None``, will sum all of the
        elements of the input array. If ``axis`` is negative it counts from the last to the first
        axis. If ``axis`` is a tuple of ints, a sum is performed on all of the axes specified in the
        tuple instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input
        array.

    Examples
    --------
    >>> ht.sum(ht.ones(2))
    DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3, 3)))
    DNDarray(9., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3, 3)).astype(ht.int))
    DNDarray(9, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3, 2, 1)), axis=-3)
    DNDarray([[3.],
              [3.]], dtype=ht.float32, device=cpu:0, split=None)

`neg(a: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Element-wise negation of `a`.

    Parameters
    ----------
    a:   DNDarray
         The input array.
    out: DNDarray, optional
         The output array. It must have a shape that the inputs broadcast to

    Examples
    --------
    >>> ht.neg(ht.array([-1, 1]))
    DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
    >>> -ht.array([-1.0, 1.0])
    DNDarray([ 1., -1.], dtype=ht.float32, device=cpu:0, split=None)

`negative(a: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Element-wise negation of `a`.

    Parameters
    ----------
    a:   DNDarray
         The input array.
    out: DNDarray, optional
         The output array. It must have a shape that the inputs broadcast to

    Examples
    --------
    >>> ht.neg(ht.array([-1, 1]))
    DNDarray([ 1, -1], dtype=ht.int64, device=cpu:0, split=None)
    >>> -ht.array([-1.0, 1.0])
    DNDarray([ 1., -1.], dtype=ht.float32, device=cpu:0, split=None)

`pos(a: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Element-wise positive of `a`.

    Parameters
    ----------
    a:   DNDarray
         The input array.
    out: DNDarray, optional
         The output array. It must have a shape that the inputs broadcast to.

    Notes
    -----
    Equivalent to a.copy().

    Examples
    --------
    >>> ht.pos(ht.array([-1, 1]))
    DNDarray([-1,  1], dtype=ht.int64, device=cpu:0, split=None)
    >>> +ht.array([-1.0, 1.0])
    DNDarray([-1.,  1.], dtype=ht.float32, device=cpu:0, split=None)

`positive(a: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Element-wise positive of `a`.

    Parameters
    ----------
    a:   DNDarray
         The input array.
    out: DNDarray, optional
         The output array. It must have a shape that the inputs broadcast to.

    Notes
    -----
    Equivalent to a.copy().

    Examples
    --------
    >>> ht.pos(ht.array([-1, 1]))
    DNDarray([-1,  1], dtype=ht.int64, device=cpu:0, split=None)
    >>> +ht.array([-1.0, 1.0])
    DNDarray([-1.,  1.], dtype=ht.float32, device=cpu:0, split=None)

`pow(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise power function of values of operand ``t1`` to the power of values of operand
    ``t2`` (i.e ``t1**t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values represent the base
    t2: DNDarray or scalar
        The second operand whose values represent the exponent
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If
        not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the exponentiated value. Elsewhere, the `out` array will retain its
        original value. If an uninitialized `out` array is created via the default `out=None`,
        locations within it where the condition is False will remain uninitialized. If distributed,
        the split axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.pow(3.0, 2.0)
    DNDarray(9., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[3, 3], [2, 2]])
    >>> ht.pow(T1, T2)
    DNDarray([[ 1.,  8.],
            [ 9., 16.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 3.0
    >>> ht.pow(T1, s)
    DNDarray([[ 1.,  8.],
            [27., 64.]], dtype=ht.float32, device=cpu:0, split=None)

`power(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise power function of values of operand ``t1`` to the power of values of operand
    ``t2`` (i.e ``t1**t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values represent the base
    t2: DNDarray or scalar
        The second operand whose values represent the exponent
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If
        not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the exponentiated value. Elsewhere, the `out` array will retain its
        original value. If an uninitialized `out` array is created via the default `out=None`,
        locations within it where the condition is False will remain uninitialized. If distributed,
        the split axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.pow(3.0, 2.0)
    DNDarray(9., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[3, 3], [2, 2]])
    >>> ht.pow(T1, T2)
    DNDarray([[ 1.,  8.],
            [ 9., 16.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 3.0
    >>> ht.pow(T1, s)
    DNDarray([[ 1.,  8.],
            [27., 64.]], dtype=ht.float32, device=cpu:0, split=None)

`prod(a: DNDarray, axis: Union[int, Tuple[int, ...]] = None, out: DNDarray = None, keepdims: bool = None) ‑> heat.core.dndarray.DNDarray`
:   Return the product of array elements over a given axis in form of a DNDarray shaped as a but
    with the specified axis removed.

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which a product is performed. The default, ``axis=None``, will calculate
        the product of all the elements in the input array. If axis is negative it counts from the
        last to the first axis. If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input
        array.

    Examples
    --------
    >>> ht.prod(ht.array([1.0, 2.0]))
    DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.prod(ht.array([
        [1.,2.],
        [3.,4.]]))
    DNDarray(24., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.prod(ht.array([
        [1.,2.],
        [3.,4.]
    ]), axis=1)
    DNDarray([ 2., 12.], dtype=ht.float32, device=cpu:0, split=None)

`remainder(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise division remainder of values of operand ``t1`` by values of operand ``t2`` (i.e.
    ``t1%t2``). Result has the same sign as the divisor ``t2``.
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand whose values are divided
    t2: DNDarray or scalar
        The second operand by whose values is divided
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis.
        If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the divided value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.remainder(2, 2)
    DNDarray(0, dtype=ht.int64, device=cpu:0, split=None)
    >>> T1 = ht.int32([[1, 2], [3, 4]])
    >>> T2 = ht.int32([[2, 2], [2, 2]])
    >>> ht.remainder(T1, T2)
    DNDarray([[1, 0],
            [1, 0]], dtype=ht.int32, device=cpu:0, split=None)
    >>> s = 2
    >>> ht.remainder(s, T1)
    DNDarray([[0, 0],
            [2, 2]], dtype=ht.int32, device=cpu:0, split=None)

`right_shift(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Shift the bits of an integer to the right.

    Parameters
    ----------
    t1: DNDarray or scalar
        Input array
    t2: DNDarray or scalar
        Integer number of bits to remove
    out: DNDarray, optional
        Output array for the result. Must have the same shape as the expected output. The dtype of
        the output will be the one of the input array, unless it is logical, in which case it will
        be casted to int8. If not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the shifted value. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations
        within it where the condition is False will remain uninitialized. If distributed, the split
        axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.right_shift(ht.array([1, 2, 3]), 1)
    DNDarray([0, 1, 1], dtype=ht.int64, device=cpu:0, split=None)

`sub(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise subtraction of values of operand ``t2`` from values of operands ``t1`` (i.e
    ``t1-t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand from which values are subtracted
    t2: DNDarray or scalar
        The second operand whose values are subtracted
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If
        not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the subtracted value. Elsewhere, the `out` array will retain its
        original value. If an uninitialized `out` array is created via the default `out=None`,
        locations within it where the condition is False will remain uninitialized. If distributed,
        the split axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.sub(4.0, 1.0)
    DNDarray(3., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.sub(T1, T2)
    DNDarray([[-1.,  0.],
              [ 1.,  2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.sub(s, T1)
    DNDarray([[ 1.,  0.],
              [-1., -2.]], dtype=ht.float32, device=cpu:0, split=None)

`subtract(t1: Union[DNDarray, float], t2: Union[DNDarray, float], /, out: Optional[DNDarray] = None, *, where: Union[bool, DNDarray] = True) ‑> heat.core.dndarray.DNDarray`
:   Element-wise subtraction of values of operand ``t2`` from values of operands ``t1`` (i.e
    ``t1-t2``).
    Operation is not commutative.

    Parameters
    ----------
    t1: DNDarray or scalar
        The first operand from which values are subtracted
    t2: DNDarray or scalar
        The second operand whose values are subtracted
    out: DNDarray, optional
        Output array. It must have a shape that the inputs broadcast to and matching split axis. If
        not provided or None, a freshly-allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out`
        array will be set to the subtracted value. Elsewhere, the `out` array will retain its
        original value. If an uninitialized `out` array is created via the default `out=None`,
        locations within it where the condition is False will remain uninitialized. If distributed,
        the split axis (after broadcasting if required) must match that of the `out` array.

    Examples
    --------
    >>> ht.sub(4.0, 1.0)
    DNDarray(3., dtype=ht.float32, device=cpu:0, split=None)
    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.sub(T1, T2)
    DNDarray([[-1.,  0.],
              [ 1.,  2.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> s = 2.0
    >>> ht.sub(s, T1)
    DNDarray([[ 1.,  0.],
              [-1., -2.]], dtype=ht.float32, device=cpu:0, split=None)

`sum(a: DNDarray, axis: Union[int, Tuple[int, ...]] = None, out: DNDarray = None, keepdims: bool = None) ‑> heat.core.dndarray.DNDarray`
:   Sum of array elements over a given axis. An array with the same shape as ``self.__array`` except
    for the specified axis which becomes one, e.g.
    ``a.shape=(1, 2, 3)`` => ``ht.ones((1, 2, 3)).sum(axis=1).shape=(1, 1, 3)``

    Parameters
    ----------
    a : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis along which a sum is performed. The default, ``axis=None``, will sum all of the
        elements of the input array. If ``axis`` is negative it counts from the last to the first
        axis. If ``axis`` is a tuple of ints, a sum is performed on all of the axes specified in the
        tuple instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the
        expected output, but the datatype of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions
        with size one. With this option, the result will broadcast correctly against the input
        array.

    Examples
    --------
    >>> ht.sum(ht.ones(2))
    DNDarray(2., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3, 3)))
    DNDarray(9., dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3, 3)).astype(ht.int))
    DNDarray(9, dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.sum(ht.ones((3, 2, 1)), axis=-3)
    DNDarray([[3.],
              [3.]], dtype=ht.float32, device=cpu:0, split=None)
