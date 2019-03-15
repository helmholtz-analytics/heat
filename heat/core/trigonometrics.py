import torch

from .operations import __local_operation as local_op

__all__ = [
    'cos',
    'sin',
    'tan'
]


def cos(x, out=None):
    """
    Return the trigonometric cosine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the trigonometric sine.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    cosine : ht.tensor
        A tensor of the same shape as x, containing the trigonometric cosine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.cos(ht.arange(-6, 7, 2))
    tensor([0.96017029, -0.65364362, -0.41614684,  1., -0.41614684, -0.65364362,  0.96017029])
    """
    return local_op(torch.cos, x, out)


def sin(x, out=None):
    """
    Return the trigonometric sine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the trigonometric sine.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    sine : ht.tensor
        A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sin(ht.arange(-6, 7, 2))
    tensor([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794])
    """
    return local_op(torch.sin, x, out)


def tan(x, out=None):
    """
    Compute tangent element-wise.

    Equivalent to ht.sin(x) / ht.cos(x) element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the trigonometric tangent.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    tangent : ht.tensor
        A tensor of the same shape as x, containing the trigonometric tangent of each element in this tensor.

    Examples
    --------
    >>> ht.tan(ht.arange(-6, 7, 2))
    tensor([ 0.29100619, -1.15782128,  2.18503986,  0., -2.18503986, 1.15782128, -0.29100619])
    """
    return local_op(torch.tan, x, out)
