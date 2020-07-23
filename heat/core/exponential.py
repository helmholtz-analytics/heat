import torch
from typing import List, Dict, Any, TypeVar, Union, Tuple, Sequence

from . import _operations
from .dndarray import DNDarray

__all__ = ["exp", "expm1", "exp2", "log", "log2", "log10", "log1p", "sqrt"]


def exp(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    Calculate the exponential of all elements in the input array.
    Result is a :class:`DNDarray` of the same shape as ``x``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the exponential.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.exp(ht.arange(5))
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    """
    return _operations.__local_op(torch.exp, x, out)


DNDarray.exp = lambda self, out=None: exp(self, out)
DNDarray.exp.__doc__ = exp.__doc__


def expm1(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    Calculate :math:`exp(x)-1` for all elements in the array.
    Result is a :class:`DNDarray` of the same shape as ``x``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the exponential.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.expm1(ht.arange(5)) + 1.
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    """
    return _operations.__local_op(torch.expm1, x, out)


DNDarray.expm1 = lambda self, out=None: expm1(self, out)
DNDarray.expm1.__doc__ = expm1.__doc__


def exp2(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    Calculate the exponential of two of all elements in the input array (``2**x``) .
    Result is a :class:`DNDarray` of the same shape as ``x``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the ``2**x``.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.exp2(ht.arange(5))
    tensor([ 1.,  2.,  4.,  8., 16.], dtype=torch.float64)
    """

    def local_exp2(xl, outl=None):
        return torch.pow(2, xl, out=outl)

    return _operations.__local_op(local_exp2, x, out)


DNDarray.exp2 = lambda self, out=None: exp2(self, out)
DNDarray.exp2.__doc__ = exp2.__doc__


def log(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    Natural logarithm, element-wise.
    The natural logarithm is the inverse of the exponential function, so that ``log(exp(x))=x``. The natural
    logarithm is logarithm in base e. Result is a :class:`DNDarray` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.log(ht.arange(5))
    tensor([  -inf, 0.0000, 0.6931, 1.0986, 1.3863])
    """
    return _operations.__local_op(torch.log, x, out)


DNDarray.log = lambda self, out=None: log(self, out)
DNDarray.log.__doc__ = log.__doc__


def log2(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    log base 2, element-wise.
    Result is a :class:`DNDarray` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.log2(ht.arange(5))
    tensor([  -inf, 0.0000, 1.0000, 1.5850, 2.0000])
    """
    return _operations.__local_op(torch.log2, x, out)


DNDarray.log2 = lambda self, out=None: log2(self, out)
DNDarray.log2.__doc__ = log2.__doc__


def log10(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    log base 10, element-wise.
    Result is a :class:`DNDarray` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.log10(ht.arange(5))
    tensor([  -inf, 0.0000, 0.3010, 0.4771, 0.6021])
    """
    return _operations.__local_op(torch.log10, x, out)


DNDarray.log10 = lambda self, out=None: log10(self, out)
DNDarray.log10.__doc__ = log10.__doc__


def log1p(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    Return the natural logarithm of one plus the input array, element-wise.
    Result is a :class:`DNDarray` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the logarithm.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.log1p(ht.arange(5))
    array([0., 0.69314718, 1.09861229, 1.38629436, 1.60943791])
    """
    return _operations.__local_op(torch.log1p, x, out)


DNDarray.log1p = lambda self, out=None: log1p(self, out)
DNDarray.log1p.__doc__ = log1p.__doc__


def sqrt(x: DNDarray, out: Union[None, DNDarray] = None) -> DNDarray:
    """
    Return the non-negative square-root of a tensor element-wise.
    Result is a :class:`DNDarray` of the same shape as ``x``.
    Negative input elements are returned as ``NaN``.

    Parameters
    ----------
    x : DNDarray
        The array for which to compute the square-roots.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to ``None``, a fresh array is allocated.

    Examples
    --------
    >>> ht.sqrt(ht.arange(5))
    tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000])
    >>> ht.sqrt(ht.arange(-5, 0))
    tensor([nan, nan, nan, nan, nan])
    """
    return _operations.__local_op(torch.sqrt, x, out)


DNDarray.sqrt = lambda self, out=None: sqrt(self, out)
DNDarray.sqrt.__doc__ = sqrt.__doc__
