import torch

from . import _operations
from . import dndarray
from . import types

__all__ = ["abs", "absolute", "ceil", "clip", "fabs", "floor", "modf", "round", "trunc"]


def abs(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The values for which the compute the absolute value.
    out : ht.DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.DNDarray
        A tensor containing the absolute value of each element in x.
    """
    if dtype is not None and not issubclass(dtype, types.generic):
        raise TypeError("dtype must be a heat data type")

    absolute_values = _operations.__local_op(torch.abs, x, out)
    if dtype is not None:
        absolute_values._DNDarray__array = absolute_values._DNDarray__array.type(dtype.torch_type())
        absolute_values._DNDarray__dtype = dtype

    return absolute_values


def absolute(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    np.abs is a shorthand for this function.

    Parameters
    ----------
    x : ht.DNDarray
        The values for which the compute the absolute value.
    out : ht.DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.DNDarray
        A tensor containing the absolute value of each element in x.
    """
    return abs(x, out, dtype)


def ceil(x, out=None):
    """
    Return the ceil of the input, element-wise.

    The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as :math:`\\lceil x \\rceil`.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the ceiled values.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    ceiled : ht.DNDarray
        A tensor of the same shape as x, containing the ceiled valued of each element in this tensor. If out was
        provided, ceiled is a reference to it.

    Examples
    --------
    >>> ht.ceil(ht.arange(-2.0, 2.0, 0.4))
    tensor([-2., -1., -1., -0., -0., -0.,  1.,  1.,  2.,  2.])
    """
    return _operations.__local_op(torch.ceil, x, out)


def clip(a, a_min, a_max, out=None):
    """
    Parameters
    ----------
    a : ht.DNDarray
        Array containing elements to clip.
    a_min : scalar or None
        Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and
        a_max may be None.
    a_max : scalar or None
        Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and
        a_max may be None.
    out : ht.DNDarray, optional
        The results will be placed in this array. It may be the input array for in-place clipping. out must be of
        the right shape to hold the output. Its type is preserved.

    Returns
    -------
    clipped_values : ht.DNDarray
        A tensor with the elements of this tensor, but where values < a_min are replaced with a_min, and those >
        a_max with a_max.
    """
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("a must be a tensor")
    if a_min is None and a_max is None:
        raise ValueError("either a_min or a_max must be set")

    if out is None:
        return dndarray.DNDarray(
            a._DNDarray__array.clamp(a_min, a_max), a.shape, a.dtype, a.split, a.device, a.comm
        )
    if not isinstance(out, dndarray.DNDarray):
        raise TypeError("out must be a tensor")

    return a._DNDarray__array.clamp(a_min, a_max, out=out._DNDarray__array) and out


def fabs(x, out=None):
    """
    Calculate the absolute value element-wise and return floating-point tensor.
    This function exists besides abs==absolute since it will be needed in case complex numbers will be introduced in the future.

    Parameters
    ----------
    x : ht.tensor
        The values for which the compute the absolute value.
    out : ht.tensor, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    absolute_values : ht.tensor
        A tensor containing the absolute value of each element in x.
    """

    return abs(x, out, dtype=None)


def floor(x, out=None):
    """
    Return the floor of the input, element-wise.

    The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as :math:`\\lfloor x \\rfloor`.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the floored values.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    floored : ht.DNDarray
        A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
        provided, floored is a reference to it.

    Examples
    --------
    >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
    tensor([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.])
    """
    return _operations.__local_op(torch.floor, x, out)


def modf(x, out=None):
    """
    Return the fractional and integral parts of a tensor, element-wise.
    The fractional and integral parts are negative if the given number is negative.

    Parameters
    ----------
    x : ht.DNDarray
        Input tensor
    out : tuple(ht.DNDarray, ht.DNDarray), optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated tensor is returned.

    Returns
    -------
    tuple(ht.DNDarray: fractionalParts, ht.DNDarray: integralParts)

    fractionalParts : ht.DNDdarray
        Fractional part of x. This is a scalar if x is a scalar.

    integralParts : ht.DNDdarray
        Integral part of x. This is a scalar if x is a scalar.

    Examples
    --------
    >>> ht.modf(ht.arange(-2.0, 2.0, 0.4))
        (tensor([-2., -1., -1., -0., -0.,  0.,  0.,  0.,  1.,  1.]),
        tensor([ 0.0000, -0.6000, -0.2000, -0.8000, -0.4000,  0.0000,  0.4000,  0.8000, 0.2000,  0.6000]))
    """
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("expected x to be a ht.DNDarray, but was {}".format(type(x)))

    integralParts = trunc(x)
    fractionalParts = x - integralParts

    if out is not None:
        if not isinstance(out, tuple):
            raise TypeError(
                "expected out to be None or a tuple of ht.DNDarray, but was {}".format(type(out))
            )
        if len(out) != 2:
            raise ValueError(
                "expected out to be a tuple of length 2, but was of length {}".format(len(out))
            )
        if (not isinstance(out[0], dndarray.DNDarray)) or (
            not isinstance(out[1], dndarray.DNDarray)
        ):
            raise TypeError(
                "expected out to be None or a tuple of ht.DNDarray, but was ({}, {})".format(
                    type(out[0]), type(out[1])
                )
            )
        out[0]._DNDarray__array = fractionalParts._DNDarray__array
        out[1]._DNDarray__array = integralParts._DNDarray__array
        return out

    return (fractionalParts, integralParts)


def round(x, decimals=0, out=None, dtype=None):
    """
    Calculate the rounded value element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The values for which the compute the rounded value.
    decimals: int, optional
        Number of decimal places to round to (default: 0).
        If decimals is negative, it specifies the number of positions to the left of the decimal point.
    out : ht.DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.


    Returns
    -------
    rounded_values : ht.DNDarray
        A tensor containing the rounded value of each element in x.

    Examples
    --------
    >>> ht.round(ht.arange(-2.0, 2.0, 0.4))
        tensor([-2., -2., -1., -1., -0.,  0.,  0.,  1.,  1.,  2.])

    """
    if dtype is not None and not issubclass(dtype, types.generic):
        raise TypeError("dtype must be a heat data type")

    if decimals != 0:
        x *= 10 ** decimals

    rounded_values = _operations.__local_op(torch.round, x, out)

    if decimals != 0:
        rounded_values /= 10 ** decimals

    if dtype is not None:
        rounded_values._DNDarray__array = rounded_values._DNDarray__array.type(dtype.torch_type())
        rounded_values._DNDarray__dtype = dtype

    return rounded_values


def trunc(x, out=None):
    """
    Return the trunc of the input, element-wise.

    The truncated value of the scalar x is the nearest integer i which is closer to zero than x is. In short, the
    fractional part of the signed number x is discarded.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the trunced values.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    trunced : ht.DNDarray
        A tensor of the same shape as x, containing the trunced valued of each element in this tensor. If out was
        provided, trunced is a reference to it.

    Examples
    --------
    >>> ht.trunc(ht.arange(-2.0, 2.0, 0.4))
    tensor([-2., -1., -1., -0., -0.,  0.,  0.,  0.,  1.,  1.])
    """
    return _operations.__local_op(torch.trunc, x, out)
