"""
This module computes exponential and logarithmic operations.
"""

import torch
from typing import Optional

from . import _operations
from .dndarray import DNDarray

__all__ = [
    "exp",
    "expm1",
    "exp2",
    "log",
    "log2",
    "log10",
    "log1p",
    "logaddexp",
    "logaddexp2",
    "sqrt",
    "square",
]


def exp(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Calculate the exponential of all elements in the input array.
    Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the exponential.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.exp(ht.arange(5))
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.exp, x, out)


DNDarray.exp = lambda self, out=None: exp(self, out)
DNDarray.exp.__doc__ = exp.__doc__


def expm1(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Calculate :math:`exp(x) - 1` for all elements in the array.
    Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the exponential.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.expm1(ht.arange(5)) + 1.
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float64, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.expm1, x, out)


DNDarray.expm1 = lambda self, out=None: expm1(self, out)
DNDarray.expm1.__doc__ = expm1.__doc__


def exp2(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Calculate the exponential of two of all elements in the input array (:math:`2^x`).
    Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the exponential of two.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.exp2(ht.arange(5))
    DNDarray([ 1.,  2.,  4.,  8., 16.], dtype=ht.float32, device=cpu:0, split=None)
    """

    def local_exp2(xl, outl=None):
        return torch.pow(2, xl, out=outl)

    return _operations.__local_op(local_exp2, x, out)


DNDarray.exp2 = lambda self, out=None: exp2(self, out)
DNDarray.exp2.__doc__ = exp2.__doc__


def log(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Natural logarithm, element-wise.
    The natural logarithm is the inverse of the exponential function, so that :math:`log(exp(x)) = x`. The natural
    logarithm is logarithm in base e. Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
    Negative input elements are returned as :abbr:`NaN (Not a Number)`.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.log(ht.arange(5))
    DNDarray([  -inf, 0.0000, 0.6931, 1.0986, 1.3863], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.log, x, out)


DNDarray.log = lambda self, out=None: log(self, out)
DNDarray.log.__doc__ = log.__doc__


def log2(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the logarithm to the base 2 (:math:`log_2(x)`), element-wise.
    Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
    Negative input elements are returned as :abbr:`NaN (Not a Number)`.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.log2(ht.arange(5))
    DNDarray([  -inf, 0.0000, 1.0000, 1.5850, 2.0000], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.log2, x, out)


DNDarray.log2 = lambda self, out=None: log2(self, out)
DNDarray.log2.__doc__ = log2.__doc__


def log10(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compute the logarithm to the base 10 (:math:`log_{10}(x)`), element-wise.
    Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
    Negative input elements are returned as :abbr:`NaN (Not a Number)`.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.log10(ht.arange(5))
    DNDarray([  -inf, 0.0000, 0.3010, 0.4771, 0.6021], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.log10, x, out)


DNDarray.log10 = lambda self, out=None: log10(self, out)
DNDarray.log10.__doc__ = log10.__doc__


def log1p(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Return the natural logarithm of one plus the input array, element-wise.
    Result is a :class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
    Negative input elements are returned as :abbr:`NaN (Not a Number)`.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.log1p(ht.arange(5))
    DNDarray([0.0000, 0.6931, 1.0986, 1.3863, 1.6094], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.log1p, x, out)


DNDarray.log1p = lambda self, out=None: log1p(self, out)
DNDarray.log1p.__doc__ = log1p.__doc__


def logaddexp(x1: DNDarray, x2: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Calculates the logarithm of the sum of exponentiations :math:`log(exp(x1) + exp(x2))` for each element :math:`{x1}_i` of
    the input array x1 with the respective element :math:`{x2}_i` of the input array x2.

    Parameters
    ----------
    x1 : DNDarray
        first input array. Should have a floating-point data type.
    x2 : DNDarray
        second input array. Must be compatible with x1. Should have a floating-point data type.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    See Also
    --------
    :func:`logaddexp2`
        Logarithm of the sum of exponentiations of inputs in base-2.

    Examples
    --------
    >>> ht.logaddexp(ht.array([-1.0]), ht.array([-1.0, -2, -3]))
    DNDarray([-0.3069, -0.6867, -0.8731], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.logaddexp, x1, x2, out)


def logaddexp2(x1: DNDarray, x2: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Calculates the logarithm of the sum of exponentiations in base-2 :math:`log2(exp(x1) + exp(x2))` for each element :math:`{x1}_i` of
    the input array x1 with the respective element :math:`{x2}_i` of the input array x2.

    Parameters
    ----------
    x1 : DNDarray
        first input array. Should have a floating-point data type.
    x2 : DNDarray
        second input array. Must be compatible with x1. Should have a floating-point data type.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    See Also
    --------
    :func:`logaddexp`
        Logarithm of the sum of exponentiations of inputs.

    Examples
    --------
    >>> ht.logaddexp2(ht.array([-1.0]), ht.array([-1.0, -2, -3]))
    DNDarray([ 0.0000, -0.4150, -0.6781], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.logaddexp2, x1, x2, out)


def sqrt(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Return the non-negative square-root of a tensor element-wise.
    Result is a :py:class:`~heat.core.dndarray.DNDarray` of the same shape as ``x``.
    Negative input elements are returned as :abbr:`NaN (Not a Number)`.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the square-roots.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples
    --------
    >>> ht.sqrt(ht.arange(5))
    DNDarray([0.0000, 1.0000, 1.4142, 1.7321, 2.0000], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.sqrt(ht.arange(-5, 0))
    DNDarray([nan, nan, nan, nan, nan], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.sqrt, x, out)


DNDarray.sqrt = lambda self, out=None: sqrt(self, out)
DNDarray.sqrt.__doc__ = sqrt.__doc__


def square(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Return a new tensor with the squares of the elements of input.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the squares.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to :keyword:`None`, a fresh array is allocated.

    Examples:
    --------
    >>> a = ht.random.rand(4)
    >>> a
    DNDarray([0.8654, 0.1432, 0.9164, 0.6179], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.square(a)
    DNDarray([0.7488, 0.0205, 0.8397, 0.3818], dtype=ht.float32, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.square, x, out)


DNDarray.square = lambda self, out=None: square(self, out)
DNDarray.square.__doc__ = square.__doc__
