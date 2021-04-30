"""
Trig functions
"""

from __future__ import annotations

import torch

from typing import Callable, Optional

from .constants import pi
from .dndarray import DNDarray
from ._operations import __local_op as local_op
from ._operations import __binary_op as binary_op
from . import types


__all__ = [
    "acos",
    "asin",
    "atan",
    "atan2",
    "arccos",
    "arcsin",
    "arctan",
    "arctan2",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "rad2deg",
    "radians",
    "sin",
    "sinh",
    "tan",
    "tanh",
]


def arccos(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the trigonometric arccos, element-wise.
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
    >>> ht.arccos(ht.array([-1.,-0., 0.83]))
    DNDarray([3.1416, 1.5708, 0.5917], dtype=ht.float32, device=cpu:0, split=None)
    """
    return local_op(torch.acos, x, out)


acos = arccos
acos.__doc__ = arccos.__doc__

DNDarray.acos: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: acos(
    self, out
)
DNDarray.acos.__doc__ = acos.__doc__


def arcsin(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
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
    >>> ht.arcsin(ht.array([-1.,-0., 0.83]))
    DNDarray([-1.5708, -0.0000,  0.9791], dtype=ht.float32, device=cpu:0, split=None)
    """
    return local_op(torch.asin, x, out)


asin = arcsin
asin.__doc__ = arcsin.__doc__

DNDarray.asin: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: asin(
    self, out
)
DNDarray.asin.__doc__ = asin.__doc__


def arctan(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the trigonometric arctan, element-wise.
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
    """
    return local_op(torch.atan, x, out)


atan = arctan
atan.__doc__ = arctan.__doc__

DNDarray.atan: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: atan(
    self, out
)
DNDarray.atan.__doc__ = atan.__doc__


def arctan2(x1: DNDarray, x2: DNDarray) -> DNDarray:
    """
    Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.
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
    """
    # Cast integer to float because torch.atan2() only supports integer types on PyTorch 1.5.0.
    x1 = x1.astype(types.promote_types(x1.dtype, types.float))
    x2 = x2.astype(types.promote_types(x2.dtype, types.float))

    return binary_op(torch.atan2, x1, x2)


atan2 = arctan2
atan2.__doc__ = arctan2.__doc__

DNDarray.atan2: Callable[[DNDarray, DNDarray], DNDarray] = lambda self, x2: atan2(self, x2)
DNDarray.atan2.__doc__ = atan2.__doc__


def cos(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Return the trigonometric cosine, element-wise.

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
    """
    return local_op(torch.cos, x, out)


DNDarray.cos: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: cos(
    self, out
)
DNDarray.cos.__doc__ = cos.__doc__


def cosh(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the hyperbolic cosine, element-wise.
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
    """
    return local_op(torch.cosh, x, out)


DNDarray.cosh: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: cosh(
    self, out
)
DNDarray.cosh.__doc__ = cosh.__doc__


def deg2rad(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in radians.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.deg2rad(ht.array([0.,20.,45.,78.,94.,120.,180., 270., 311.]))
    DNDarray([0.0000, 0.3491, 0.7854, 1.3614, 1.6406, 2.0944, 3.1416, 4.7124, 5.4280], dtype=ht.float32, device=cpu:0, split=None)
    """
    # deg2rad torch version
    def torch_deg2rad(torch_tensor):
        if not torch.is_tensor(torch_tensor):
            raise TypeError("Input is not a torch tensor but {}".format(type(torch_tensor)))
        return torch_tensor * pi / 180.0

    return local_op(torch_deg2rad, x, out)


def degrees(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in degrees.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.degrees(ht.array([0.,0.2,0.6,0.9,1.2,2.7,3.14]))
    DNDarray([  0.0000,  11.4592,  34.3775,  51.5662,  68.7549, 154.6986, 179.9088], dtype=ht.float32, device=cpu:0, split=None)
    """
    return rad2deg(x, out=out)


def rad2deg(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in degrees.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.rad2deg(ht.array([0.,0.2,0.6,0.9,1.2,2.7,3.14]))
    DNDarray([  0.0000,  11.4592,  34.3775,  51.5662,  68.7549, 154.6986, 179.9088], dtype=ht.float32, device=cpu:0, split=None)
    """
    # rad2deg torch version
    def torch_rad2deg(torch_tensor):
        if not torch.is_tensor(torch_tensor):
            raise TypeError("Input is not a torch tensor but {}".format(type(torch_tensor)))
        return 180.0 * torch_tensor / pi

    return local_op(torch_rad2deg, x, out=out)


def radians(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : DNDarray
        The value for which to compute the angles in radians.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.radians(ht.array([0., 20., 45., 78., 94., 120., 180., 270., 311.]))
    DNDarray([0.0000, 0.3491, 0.7854, 1.3614, 1.6406, 2.0944, 3.1416, 4.7124, 5.4280], dtype=ht.float32, device=cpu:0, split=None)
    """
    return deg2rad(x, out)


def sin(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the trigonometric sine, element-wise.
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
    """
    return local_op(torch.sin, x, out)


DNDarray.sin: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: sin(
    self, out
)
DNDarray.sin.__doc__ = sin.__doc__


def sinh(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the hyperbolic sine, element-wise.
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
    """
    return local_op(torch.sinh, x, out)


DNDarray.sinh: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: sinh(
    self, out
)
DNDarray.sinh.__doc__ = sinh.__doc__


def tan(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute tangent element-wise.
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
    """
    return local_op(torch.tan, x, out)


DNDarray.tan: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: tan(
    self, out
)
DNDarray.tan.__doc__ = tan.__doc__


def tanh(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the hyperbolic tangent, element-wise.
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
    """
    return local_op(torch.tanh, x, out)


DNDarray.tanh: Callable[[DNDarray, Optional[DNDarray]], DNDarray] = lambda self, out=None: tanh(
    self, out
)
DNDarray.tanh.__doc__ = tanh.__doc__
