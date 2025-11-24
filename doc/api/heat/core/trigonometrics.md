Module heat.core.trigonometrics
===============================
Trig functions

Functions
---------

`acos(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the trigonometric arccos, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arccos`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the trigonometric cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.arccos(ht.array([-1.0, -0.0, 0.83]))
    DNDarray([3.1416, 1.5708, 0.5917], dtype=ht.float32, device=cpu:0, split=None)

`acosh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse hyperbolic cosine, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [1., +infinity] are returned as ``NaN``. If ``out`` was provided, ``acosh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the inverse hyperbolic cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.acosh(ht.array([1.0, 10.0, 20.0]))
    DNDarray([0.0000, 2.9932, 3.6883], dtype=ht.float32, device=cpu:0, split=None)

`arccos(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the trigonometric arccos, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arccos`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the trigonometric cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.arccos(ht.array([-1.0, -0.0, 0.83]))
    DNDarray([3.1416, 1.5708, 0.5917], dtype=ht.float32, device=cpu:0, split=None)

`arccosh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse hyperbolic cosine, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [1., +infinity] are returned as ``NaN``. If ``out`` was provided, ``acosh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the inverse hyperbolic cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.acosh(ht.array([1.0, 10.0, 20.0]))
    DNDarray([0.0000, 2.9932, 3.6883], dtype=ht.float32, device=cpu:0, split=None)

`arcsin(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the trigonometric arcsin, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arcsin`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the trigonometric cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.arcsin(ht.array([-1.0, -0.0, 0.83]))
    DNDarray([-1.5708, -0.0000,  0.9791], dtype=ht.float32, device=cpu:0, split=None)

`arcsinh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse hyperbolic sine, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-infinity., +infinity] are returned as ``NaN``. If ``out`` was provided, ``asinh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the inverse hyperbolic sine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.asinh(ht.array([-10.0, 0.0, 10.0]))
    DNDarray([-2.9982,  0.0000,  2.9982], dtype=ht.float32, device=cpu:0, split=None)

`arctan(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the trigonometric arctan, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arctan`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the trigonometric cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.arctan(ht.arange(-6, 7, 2))
    DNDarray([-1.4056, -1.3258, -1.1071,  0.0000,  1.1071,  1.3258,  1.4056], dtype=ht.float32, device=cpu:0, split=None)

`arctan2(x1: DNDarray, x2: DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.
    Returns a new ``DNDarray`` with the signed angles in radians between vector (``x2``,``x1``) and vector (1,0)

    Parameters
    ----------
    x1 : DNDarray
         y-coordinates
    x2 : DNDarray
         x-coordinates. If ``x1.shape!=x2.shape``, they must be broadcastable to a common shape (which becomes the shape of the output).

    Examples
    --------
    >>> x = ht.array([-1, +1, +1, -1])
    >>> y = ht.array([-1, -1, +1, +1])
    >>> ht.arctan2(y, x) * 180 / ht.pi
    DNDarray([-135.0000,  -45.0000,   45.0000,  135.0000], dtype=ht.float64, device=cpu:0, split=None)

`arctanh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse hyperbolic tangent, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``atanh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the inverse hyperbolic tangent.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.atanh(ht.array([-1.0, -0.0, 0.83]))
    DNDarray([  -inf, -0.0000, 1.1881], dtype=ht.float32, device=cpu:0, split=None)

`asin(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the trigonometric arcsin, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arcsin`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the trigonometric cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.arcsin(ht.array([-1.0, -0.0, 0.83]))
    DNDarray([-1.5708, -0.0000,  0.9791], dtype=ht.float32, device=cpu:0, split=None)

`asinh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse hyperbolic sine, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-infinity., +infinity] are returned as ``NaN``. If ``out`` was provided, ``asinh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the inverse hyperbolic sine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.asinh(ht.array([-10.0, 0.0, 10.0]))
    DNDarray([-2.9982,  0.0000,  2.9982], dtype=ht.float32, device=cpu:0, split=None)

`atan(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the trigonometric arctan, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``arctan`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the trigonometric cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.arctan(ht.arange(-6, 7, 2))
    DNDarray([-1.4056, -1.3258, -1.1071,  0.0000,  1.1071,  1.3258,  1.4056], dtype=ht.float32, device=cpu:0, split=None)

`atan2(x1: DNDarray, x2: DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.
    Returns a new ``DNDarray`` with the signed angles in radians between vector (``x2``,``x1``) and vector (1,0)

    Parameters
    ----------
    x1 : DNDarray
         y-coordinates
    x2 : DNDarray
         x-coordinates. If ``x1.shape!=x2.shape``, they must be broadcastable to a common shape (which becomes the shape of the output).

    Examples
    --------
    >>> x = ht.array([-1, +1, +1, -1])
    >>> y = ht.array([-1, -1, +1, +1])
    >>> ht.arctan2(y, x) * 180 / ht.pi
    DNDarray([-135.0000,  -45.0000,   45.0000,  135.0000], dtype=ht.float64, device=cpu:0, split=None)

`atanh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse hyperbolic tangent, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Input elements outside [-1., 1.] are returned as ``NaN``. If ``out`` was provided, ``atanh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the inverse hyperbolic tangent.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.atanh(ht.array([-1.0, -0.0, 0.83]))
    DNDarray([  -inf, -0.0000, 1.1881], dtype=ht.float32, device=cpu:0, split=None)

`cos(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Return the trigonometric cosine, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trigonometric cosine.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Examples
    --------
    >>> ht.cos(ht.arange(-6, 7, 2))
    DNDarray([ 0.9602, -0.6536, -0.4161,  1.0000, -0.4161, -0.6536,  0.9602], dtype=ht.float32, device=cpu:0, split=None)

`cosh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the hyperbolic cosine, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``. If ``out`` was provided, ``cosh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the hyperbolic cosine.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.cosh(ht.arange(-6, 7, 2))
    DNDarray([201.7156,  27.3082,   3.7622,   1.0000,   3.7622,  27.3082, 201.7156], dtype=ht.float32, device=cpu:0, split=None)

`deg2rad(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Convert angles from degrees to radians.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in radians.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.deg2rad(ht.array([0.0, 20.0, 45.0, 78.0, 94.0, 120.0, 180.0, 270.0, 311.0]))
    DNDarray([0.0000, 0.3491, 0.7854, 1.3614, 1.6406, 2.0944, 3.1416, 4.7124, 5.4280], dtype=ht.float32, device=cpu:0, split=None)

`degrees(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Convert angles from radians to degrees.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in degrees.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.degrees(ht.array([0.0, 0.2, 0.6, 0.9, 1.2, 2.7, 3.14]))
    DNDarray([  0.0000,  11.4592,  34.3775,  51.5662,  68.7549, 154.6986, 179.9088], dtype=ht.float32, device=cpu:0, split=None)

`rad2deg(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Convert angles from radians to degrees.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in degrees.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.rad2deg(ht.array([0.0, 0.2, 0.6, 0.9, 1.2, 2.7, 3.14]))
    DNDarray([  0.0000,  11.4592,  34.3775,  51.5662,  68.7549, 154.6986, 179.9088], dtype=ht.float32, device=cpu:0, split=None)

`radians(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Convert angles from degrees to radians.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in radians.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.radians(ht.array([0.0, 20.0, 45.0, 78.0, 94.0, 120.0, 180.0, 270.0, 311.0]))
    DNDarray([0.0000, 0.3491, 0.7854, 1.3614, 1.6406, 2.0944, 3.1416, 4.7124, 5.4280], dtype=ht.float32, device=cpu:0, split=None)

`sin(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the trigonometric sine, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``. If ``out`` was provided, ``sin`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the trigonometric tangent.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.sin(ht.arange(-6, 7, 2))
    DNDarray([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794], dtype=ht.float32, device=cpu:0, split=None)

`sinh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the hyperbolic sine, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``. If ``out`` was provided, ``sinh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the hyperbolic sine.
    out : DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.sinh(ht.arange(-6, 7, 2))
    DNDarray([-201.7132,  -27.2899,   -3.6269,    0.0000,    3.6269,   27.2899,  201.7132], dtype=ht.float32, device=cpu:0, split=None)

`tan(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute tangent element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    Equivalent to :func:`sin`/:func:`cos` element-wise. If ``out`` was provided, ``tan`` is a reference to it.


    Parameters
    ----------
    x : DNDarray
        The value for which to compute the trigonometric tangent.
    out : DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.tan(ht.arange(-6, 7, 2))
    DNDarray([ 0.2910, -1.1578,  2.1850,  0.0000, -2.1850,  1.1578, -0.2910], dtype=ht.float32, device=cpu:0, split=None)

`tanh(x: DNDarray, out: Optional[DNDarray] = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the hyperbolic tangent, element-wise.
    Result is a ``DNDarray`` of the same shape as ``x``.
    If ``out`` was provided, ``tanh`` is a reference to it.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the hyperbolic tangent.
    out : DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.tanh(ht.arange(-6, 7, 2))
    DNDarray([-1.0000, -0.9993, -0.9640,  0.0000,  0.9640,  0.9993,  1.0000], dtype=ht.float32, device=cpu:0, split=None)
