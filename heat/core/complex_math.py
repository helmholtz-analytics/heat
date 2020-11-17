import torch

from . import _operations
from . import constants
from . import factories
from . import trigonometrics
from . import types

__all__ = ["angle", "conj", "conjugate", "imag", "real"]


def angle(z, deg: bool = False, out=None):
    """
    Calculate the element-wise angle of the complex argument.

    Parameters
    ----------
    z : DNDarray
    deg : bool, optional
    out : DNDarray, optional

    Returns
    -------
    out : DNDarray

    Examples
    --------
    >>> ht.angle(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 0.0000,  1.5708,  0.7854,  2.3562, -0.7854], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.angle(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]), deg=True)
    DNDarray([  0.,  90.,  45., 135., -45.], dtype=ht.float32, device=cpu:0, split=None)
    """
    a = _operations.__local_op(torch.angle, z, out)

    if deg:
        a *= 180 / constants.pi

    return a


def conjugate(x, out=None):
    """
    Compute the complex conjugate, element-wise.

    Parameters
    ----------
    x : DNDarray
    out : DNDarray, optional

    Returns
    -------
    DNDarray

    Examples
    --------
    >>> ht.conjugate(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ (1-0j),     -1j,  (1-1j), (-2-2j),  (3+3j)], dtype=ht.complex64, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.conj, x, out)


# alias
conj = conjugate


def imag(val):
    """
    Return the imaginary part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    val : DNDarray

    Returns
    --------
    DNDarray

    Examples
    --------
    >>> ht.imag(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 0.,  1.,  1.,  2., -3.], dtype=ht.float32, device=cpu:0, split=None)
    """
    if types.heat_type_is_complexfloating(val.dtype):
        return _operations.__local_op(torch.imag, val, None)
    else:
        return factories.zeros_like(val)


def real(val):
    """
    Return the real part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    val : DNDarray
        input

    Returns
    ------
    DNDarray

    Examples
    --------
    >>> ht.real(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 1.,  0.,  1., -2.,  3.], dtype=ht.float32, device=cpu:0, split=None)
    """
    if types.heat_type_is_complexfloating(val.dtype):
        return _operations.__local_op(torch.real, val, None)
    else:
        return val
