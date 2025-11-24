Module heat.core.complex_math
=============================
Complex numbers module.

Functions
---------

`angle(x: heat.core.dndarray.DNDarray, deg: bool = False, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Calculate the element-wise angle of the complex argument.

    Parameters
    ----------
    x : DNDarray
        Input array for which to compute the angle.
    deg : bool, optional
        Return the angle in degrees (True) or radiands (False).
    out : DNDarray, optional
        Output array with the angles.

    Examples
    --------
    >>> ht.angle(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j]))
    DNDarray([ 0.0000,  1.5708,  0.7854,  2.3562, -0.7854], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.angle(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j]), deg=True)
    DNDarray([  0.,  90.,  45., 135., -45.], dtype=ht.float32, device=cpu:0, split=None)

`conj(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the complex conjugate, element-wise.

    Parameters
    ----------
    x : DNDarray
        Input array for which to compute the complex conjugate.
    out : DNDarray, optional
        Output array with the complex conjugates.

    Examples
    --------
    >>> ht.conjugate(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j]))
    DNDarray([ (1-0j),     -1j,  (1-1j), (-2-2j),  (3+3j)], dtype=ht.complex64, device=cpu:0, split=None)

`conjugate(x: heat.core.dndarray.DNDarray, out: heat.core.dndarray.DNDarray | None = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the complex conjugate, element-wise.

    Parameters
    ----------
    x : DNDarray
        Input array for which to compute the complex conjugate.
    out : DNDarray, optional
        Output array with the complex conjugates.

    Examples
    --------
    >>> ht.conjugate(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j]))
    DNDarray([ (1-0j),     -1j,  (1-1j), (-2-2j),  (3+3j)], dtype=ht.complex64, device=cpu:0, split=None)

`imag(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Return the imaginary part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    x : DNDarray
        Input array for which the imaginary part is returned.

    Examples
    --------
    >>> ht.imag(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j]))
    DNDarray([ 0.,  1.,  1.,  2., -3.], dtype=ht.float32, device=cpu:0, split=None)

`real(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Return the real part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    x : DNDarray
        Input array for which the real part is returned.

    Examples
    --------
    >>> ht.real(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j]))
    DNDarray([ 1.,  0.,  1., -2.,  3.], dtype=ht.float32, device=cpu:0, split=None)
