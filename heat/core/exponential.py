import torch

from . import operations

__all__ = ["exp", "expm1", "exp2", "log", "log2", "log10", "log1p", "sqrt"]


def exp(x, out=None):
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the exponential.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    exponentials : ht.DNDarray
        A tensor of the same shape as x, containing the positive exponentials of each element in this tensor. If out
        was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.exp(ht.arange(5))
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    """
    return operations.__local_op(torch.exp, x, out)


def expm1(x, out=None):
    """
    Calculate exp(x) - 1 for all elements in the array.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the exponential.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    exponentials : ht.DNDarray
        A tensor of the same shape as x, containing the positive exponentials of each element in this tensor. If out
        was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.expm1(ht.arange(5)) + 1.
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    """
    return operations.__local_op(torch.expm1, x, out)


def exp2(x, out=None):
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the 2**p.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    exponentials : ht.DNDarray
        A tensor of the same shape as x, containing the 2**p of each element p in this tensor. If out
        was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.exp2(ht.arange(5))
    tensor([ 1.,  2.,  4.,  8., 16.], dtype=torch.float64)
    """

    def local_exp2(xl, outl=None):
        return torch.pow(2, xl, out=outl)

    return operations.__local_op(local_exp2, x, out)


def log(x, out=None):
    """
    Natural logarithm, element-wise.

    The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x. The natural
    logarithm is logarithm in base e.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the logarithm.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    logarithms : ht.DNDarray
        A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.log(ht.arange(5))
    tensor([  -inf, 0.0000, 0.6931, 1.0986, 1.3863])
    """
    return operations.__local_op(torch.log, x, out)


def log2(x, out=None):
    """
    log base 2, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the logarithm.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    logarithms : ht.DNDarray
        A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.log2(ht.arange(5))
    tensor([  -inf, 0.0000, 1.0000, 1.5850, 2.0000])
    """
    return operations.__local_op(torch.log2, x, out)


def log10(x, out=None):
    """
    log base 10, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the logarithm.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    logarithms : ht.DNDarray
        A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.log10(ht.arange(5))
    tensor([  -inf, 0.0000, 0.3010, 0.4771, 0.6021])
    """
    return operations.__local_op(torch.log10, x, out)


def log1p(x, out=None):
    """
    Return the natural logarithm of one plus the input array, element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the logarithm.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    logarithms : ht.DNDarray
        A tensor of the same shape as x, containing the positive logarithms plus one of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.log1p(ht.arange(5))
    array([0., 0.69314718, 1.09861229, 1.38629436, 1.60943791])
    """
    return operations.__local_op(torch.log1p, x, out)


def sqrt(x, out=None):
    """
    Return the non-negative square-root of a tensor element-wise.

    Parameters
    ----------
    x : ht.DNDarray
        The value for which to compute the square-roots.
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Returns
    -------
    square_roots : ht.DNDarray
        A tensor of the same shape as x, containing the positive square-root of each element in x. Negative input
        elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sqrt(ht.arange(5))
    tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000])
    >>> ht.sqrt(ht.arange(-5, 0))
    tensor([nan, nan, nan, nan, nan])
    """
    return operations.__local_op(torch.sqrt, x, out)
