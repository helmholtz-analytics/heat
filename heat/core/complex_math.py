"""
This module handles operations focussing on complex numbers.
"""

import torch
from typing import Optional

from . import _operations
from . import constants
from . import factories
from . import trigonometrics
from . import types
from .dndarray import DNDarray

__all__ = ["angle", "conj", "conjugate", "imag", "real"]


def angle(x: DNDarray, deg: bool = False, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Calculate the element-wise angle of the complex argument.

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
    >>> ht.angle(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 0.0000,  1.5708,  0.7854,  2.3562, -0.7854], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.angle(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]), deg=True)
    DNDarray([  0.,  90.,  45., 135., -45.], dtype=ht.float32, device=cpu:0, split=None)
    """
    a = _operations.__local_op(torch.angle, x, out)

    if deg:
        a *= 180 / constants.pi

    return a


def conjugate(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the complex conjugate, element-wise.

    Parameters
    ----------
    x : DNDarray
        Input array for which to compute the complex conjugate.
    out : DNDarray, optional
        Output array with the complex conjugates.

    Examples
    --------
    >>> ht.conjugate(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ (1-0j),     -1j,  (1-1j), (-2-2j),  (3+3j)], dtype=ht.complex64, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.conj, x, out)


# alias
conj = conjugate

# DNDarray method
DNDarray.conj = lambda self, out=None: conjugate(self, out)
DNDarray.conj.__doc__ = conjugate.__doc__


def imag(x: DNDarray) -> DNDarray:
    """
    Return the imaginary part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    x : DNDarray
        Input array for which the imaginary part is returned.

    Examples
    --------
    >>> ht.imag(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 0.,  1.,  1.,  2., -3.], dtype=ht.float32, device=cpu:0, split=None)
    """
    if types.heat_type_is_complexfloating(x.dtype):
        return _operations.__local_op(torch.imag, x, None)
    else:
        return factories.zeros_like(x)


def real(x: DNDarray) -> DNDarray:
    """
    Return the real part of the complex argument. The returned DNDarray and the input DNDarray share the same underlying storage.

    Parameters
    ----------
    x : DNDarray
        Input array for which the real part is returned.

    Examples
    --------
    >>> ht.real(ht.array([1.0, 1.0j, 1+1j, -2+2j, 3 - 3j]))
    DNDarray([ 1.,  0.,  1., -2.,  3.], dtype=ht.float32, device=cpu:0, split=None)
    """
    if types.heat_type_is_complexfloating(x.dtype):
        return _operations.__local_op(torch.real, x, None)
    else:
        return x
