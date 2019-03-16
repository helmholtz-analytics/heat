import torch

from . import types
from .operations import __local_operation as local_op

__all__ = [
    'abs',
    'absolute',
    'ceil',
    'floor'
]


def abs(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------

    x : ht.tensor
        The values for which the compute the absolute value.
    out : ht.tensor, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.tensor
        A tensor containing the absolute value of each element in x.
    """
    if dtype is not None and not issubclass(dtype, types.generic):
        raise TypeError('dtype must be a heat data type')

    absolute_values = local_op(torch.abs, x, out)
    if dtype is not None:
        absolute_values._tensor__array = absolute_values._tensor__array.type(
            dtype.torch_type())
        absolute_values._tensor__dtype = dtype

    return absolute_values


def absolute(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    np.abs is a shorthand for this function.

    Parameters
    ----------
    x : ht.tensor
        The values for which the compute the absolute value.
    out : ht.tensor, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.tensor
        A tensor containing the absolute value of each element in x.
    """
    return abs(x, out, dtype)


def ceil(x, out=None):
    r"""
    Return the ceil of the input, element-wise.

    The ceil of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lceil x \rceil.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the ceiled values.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    ceiled : ht.tensor
        A tensor of the same shape as x, containing the ceiled valued of each element in this tensor. If out was
        provided, ceiled is a reference to it.

    Examples
    --------
    >>> ht.ceil(ht.arange(-2.0, 2.0, 0.4))
    tensor([-2., -1., -1., -0., -0., -0.,  1.,  1.,  2.,  2.])
    """
    return local_op(torch.ceil, x, out)


def floor(x, out=None):
    r"""
    Return the floor of the input, element-wise.

    The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lfloor x \rfloor.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the floored values.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    floored : ht.tensor
        A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
        provided, floored is a reference to it.

    Examples
    --------
    >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
    tensor([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.])
    """
    return local_op(torch.floor, x, out)
