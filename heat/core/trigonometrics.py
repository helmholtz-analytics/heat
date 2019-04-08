import torch

from .operations import __local_operation as local_op

__all__ = [
    'cos',
    'cosh',
    'sin',
    'sinh',
    'tan',
    'tanh'
]


def cos(x, out=None):
    """
    Return the trigonometric cosine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the trigonometric cosine.
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


def cosh(x, out=None):
    """
    Return the hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the hyperbolic cosine.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    hyperbolic cosine : ht.tensor
        A tensor of the same shape as x, containing the hyperbolic cosine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.cosh(ht.arange(-6, 7, 2))
    tensor([201.7156,  27.3082,   3.7622,   1.0000,   3.7622,  27.3082, 201.7156])
    """
    return local_op(torch.cosh, x, out)


def sin(x, out=None):
    """
    Return the trigonometric sine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the trigonometric tangent.

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


def sinh(x, out=None):
    """
    Return the hyperbolic sine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the hyperbolic sine.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    hyperbolic sine : ht.tensor
        A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sinh(ht.arange(-6, 7, 2))
    tensor([[-201.7132,  -27.2899,   -3.6269,    0.0000,    3.6269,   27.2899,  201.7132])
    """
    return local_op(torch.sinh, x, out)


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


def tanh(x, out=None):
    """
    Return the hyperbolic tangent, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the hyperbolic tangent.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    hyperbolic tangent : ht.tensor
        A tensor of the same shape as x, containing the hyperbolic tangent of each element in this tensor.

    Examples
    --------
    >>> ht.tanh(ht.arange(-6, 7, 2))
    tensor([-1.0000, -0.9993, -0.9640,  0.0000,  0.9640,  0.9993,  1.0000])
    """
    return local_op(torch.tanh, x, out)
