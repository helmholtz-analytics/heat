Module heat.core.rounding
=========================
Rounding functions for DNDarrays

Functions
---------

`abs(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None, dtype: Type[heat.core.types.datatype] | None = None) ‑> heat.core.dndarray.DNDarray`
:   Returns :class:`~heat.core.dndarray.DNDarray` containing the elementwise abolute values of the input array ``x``.

    Parameters
    ----------
    x : DNDarray
        The array for which the compute the absolute value.
    out : DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or ``None``, a freshly-allocated array is returned.
    dtype : datatype, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Raises
    ------
    TypeError
        If dtype is not a heat type.

`absolute(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None, dtype: Type[heat.core.types.datatype] | None = None) ‑> heat.core.dndarray.DNDarray`
:   Calculate the absolute value element-wise.
    :func:`abs` is a shorthand for this function.

    Parameters
    ----------
    x : DNDarray
        The array for which the compute the absolute value.
    out : DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or ``None``, a freshly-allocated array is returned.
    dtype : datatype, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

`ceil(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return the ceil of the input, element-wise. Result is a :class:`~heat.core.dndarray.DNDarray` of the same shape as
    ``x``. The ceil of the scalar ``x`` is the smallest integer i, such that ``i>=x``. It is often denoted as
    :math:`\lceil x \rceil`.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the ceiled values.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> import heat as ht
    >>> ht.ceil(ht.arange(-2.0, 2.0, 0.4))
    DNDarray([-2., -1., -1., -0., -0.,  0.,  1.,  1.,  2.,  2.], dtype=ht.float32, device=cpu:0, split=None)

`clip(x: heat.core.dndarray.DNDarray, min, max, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Returns a :class:`~heat.core.dndarray.DNDarray` with the elements of this array, but where values
    ``<a_min`` are replaced with ``a_min``, and those ``>a_max`` with ``a_max``.

    Parameters
    ----------
    x : DNDarray
        Array containing elements to clip.
    min : scalar or None
        Minimum value. If ``None``, clipping is not performed on lower interval edge. Not more than one of ``a_min`` and
        ``a_max`` may be ``None``.
    max : scalar or None
        Maximum value. If ``None``, clipping is not performed on upper interval edge. Not more than one of ``a_min`` and
        ``a_max`` may be None.
    out : DNDarray, optional
        The results will be placed in this array. It may be the input array for in-place clipping. ``out`` must be of
        the right shape to hold the output. Its type is preserved.

    Raises
    ------
    ValueError
        if either min or max is not set

`fabs(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Calculate the absolute value element-wise and return floating-point class:`~heat.core.dndarray.DNDarray`.
    This function exists besides ``abs==absolute`` since it will be needed in case complex numbers will be introduced
    in the future.

    Parameters
    ----------
    x : DNDarray
        The array for which the compute the absolute value.
    out : DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or ``None``, a freshly-allocated array is returned.

`floor(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return the floor of the input, element-wise.
    The floor of the scalar ``x`` is the largest integer i, such that ``i<=x``.
    It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the floored values.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh :class:`~heat.core.dndarray.DNDarray` is allocated.

    Examples
    --------
    >>> import heat as ht
    >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
    DNDarray([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None)

`modf(x: heat.core.dndarray.DNDarray, out: Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray] | None = None) ‑> Tuple[heat.core.dndarray.DNDarray, heat.core.dndarray.DNDarray]`
:   Return the fractional and integral parts of a :class:`~heat.core.dndarray.DNDarray`, element-wise.
    The fractional and integral parts are negative if the given number is negative.

    Parameters
    ----------
    x : DNDarray
        Input array
    out : Tuple[DNDarray, DNDarray], optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or ``None``, a freshly-allocated array is returned.

    Raises
    ------
    TypeError
        if ``x`` is not a :class:`~heat.core.dndarray.DNDarray`
    TypeError
        if ``out`` is not None or a tuple of :class:`~heat.core.dndarray.DNDarray`
    ValueError
        if ``out`` is a tuple of length unqual 2

    Examples
    --------
    >>> import heat as ht
    >>> ht.modf(ht.arange(-2.0, 2.0, 0.4))
    (DNDarray([ 0.0000, -0.6000, -0.2000, -0.8000, -0.4000,  0.0000,  0.4000,  0.8000,  0.2000,  0.6000], dtype=ht.float32, device=cpu:0, split=None), DNDarray([-2., -1., -1., -0., -0.,  0.,  0.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None))

`round(x: heat.core.dndarray.DNDarray, decimals: int = 0, out: heat.core.dndarray.DNDarray | None = None, dtype: Type[heat.core.types.datatype] | None = None) ‑> heat.core.dndarray.DNDarray`
:   Calculate the rounded value element-wise.

    Parameters
    ----------
    x : DNDarray
        The array for which the compute the rounded value.
    decimals: int, optional
        Number of decimal places to round to.
        If decimals is negative, it specifies the number of positions to the left of the decimal point.
    out : DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or ``None``, a freshly-allocated array is returned.
    dtype : datatype, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Raises
    ------
    TypeError
        if dtype is not a heat data type

    Examples
    --------
    >>> import heat as ht
    >>> ht.round(ht.arange(-2.0, 2.0, 0.4))
    DNDarray([-2., -2., -1., -1., -0.,  0.,  0.,  1.,  1.,  2.], dtype=ht.float32, device=cpu:0, split=None)

`sgn(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Returns an indication of the sign of a number, element-wise. The definition for complex values is equivalent to :math:`x / |x|`.

    Parameters
    ----------
    x : DNDarray
        Input array
    out : DNDarray, optional
        A location in which to store the results.

    See Also
    --------
    :func:`sign`
        Equivalent function on non-complex arrays. The definition for complex values is equivalent to :math:`x / \sqrt{x \cdot x}`

    Examples
    --------
    >>> a = ht.array([-1, -0.5, 0, 0.5, 1])
    >>> ht.sign(a)
    DNDarray([-1., -1.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sgn(ht.array([5 - 2j, 3 + 4j]))
    DNDarray([(0.9284766912460327-0.3713906705379486j),  (0.6000000238418579+0.800000011920929j)], dtype=ht.complex64, device=cpu:0, split=None)

`sign(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Returns an indication of the sign of a number, element-wise. The definition for complex values is equivalent to :math:`x / \sqrt{x \cdot x}`.

    Parameters
    ----------
    x : DNDarray
        Input array
    out : DNDarray, optional
        A location in which to store the results.

    See Also
    --------
    :func:`sgn`
        Equivalent function on non-complex arrays. The definition for complex values is equivalent to :math:`x / |x|`.

    Examples
    --------
    >>> a = ht.array([-1, -0.5, 0, 0.5, 1])
    >>> ht.sign(a)
    DNDarray([-1., -1.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sign(ht.array([5 - 2j, 3 + 4j]))
    DNDarray([(1+0j), (1+0j)], dtype=ht.complex64, device=cpu:0, split=None)

`trunc(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return the trunc of the input, element-wise.
    The truncated value of the scalar ``x`` is the nearest integer ``i`` which is closer to zero than ``x`` is. In short, the
    fractional part of the signed number ``x`` is discarded.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the trunced values.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> import heat as ht
    >>> ht.trunc(ht.arange(-2.0, 2.0, 0.4))
    DNDarray([-2., -1., -1., -0., -0.,  0.,  0.,  0.,  1.,  1.], dtype=ht.float32, device=cpu:0, split=None)
