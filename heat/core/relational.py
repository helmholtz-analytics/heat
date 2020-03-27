import torch

from .communication import MPI
from . import operations

__all__ = ["eq", "equal", "ge", "gt", "le", "lt", "ne"]


def eq(t1, t2):
    """
    Element-wise rich comparison of equality between values from two operands, commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be compared as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand involved in the comparison
    t2: tensor or scalar
        The second operand involved in the comparison

    Returns
    -------
    result: ht.DNDarray
        A uint8-tensor holding 1 for all elements in which values of t1 are equal to values of t2, 0 for all other
        elements

    Examples:
    ---------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2],[3, 4]])
    >>> ht.eq(T1, 3.0)
    tensor([[0, 0],
            [1, 0]])

    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.eq(T1, T2)
    tensor([[0, 1],
            [0, 0]])
    """
    return operations.__binary_op(torch.eq, t1, t2)


def equal(t1, t2):
    """
    Overall comparison of equality between two tensors. Returns True if two tensors have the same size and elements,
    and False otherwise.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand involved in the comparison
    t2: tensor or scalar
        The second operand involved in the comparison

    Returns
    -------
    result: bool
        True if t1 and t2 have the same size and elements, False otherwise

    Examples:
    ---------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2],[3, 4]])
    >>> ht.equal(T1, ht.float32([[1, 2],[3, 4]]))
    True
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.eq(T1, T2)
    False
    >>> ht.eq(T1, 3.0)
    False
    """
    result_tensor = operations.__binary_op(torch.equal, t1, t2)

    if result_tensor._DNDarray__array.numel() == 1:
        result_value = result_tensor._DNDarray__array.item()
    else:
        result_value = True

    return result_tensor.comm.allreduce(result_value, MPI.LAND)


def ge(t1, t2):
    """
    Element-wise rich greater than or equal comparison between values from operand t1 with respect to values of
    operand t2 (i.e. t1 >= t2), not commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be compared as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand to be compared greater than or equal to second operand
    t2: tensor or scalar
       The second operand to be compared less than or equal to first operand

    Returns
    -------
    result: ht.DNDarray
        A uint8-tensor holding 1 for all elements in which values of t1 are greater than or equal tp values of t2,
        0 for all other elements

    Examples
    -------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2],[3, 4]])
    >>> ht.ge(T1, 3.0)
    tensor([[0, 0],
            [1, 1]], dtype=torch.uint8)

    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.ge(T1, T2)
    tensor([[0, 1],
            [1, 1]], dtype=torch.uint8)
    """
    return operations.__binary_op(torch.ge, t1, t2)


def gt(t1, t2):
    """
    Element-wise rich greater than comparison between values from operand t1 with respect to values of
    operand t2 (i.e. t1 > t2), not commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be compared as argument.

    Parameters
    ----------
    t1: tensor or scalar
       The first operand to be compared greater than second operand

    t2: tensor or scalar
       The second operand to be compared less than first operand

    Returns
    -------
    result: ht.DNDarray
       A uint8-tensor holding 1 for all elements in which values of t1 are greater than values of t2,
       0 for all other elements

    Examples
    -------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2],[3, 4]])
    >>> ht.gt(T1, 3.0)
    tensor([[0, 0],
            [0, 1]], dtype=torch.uint8)

    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.gt(T1, T2)
    tensor([[0, 0],
            [1, 1]], dtype=torch.uint8)
    """
    return operations.__binary_op(torch.gt, t1, t2)


def le(t1, t2):
    """
    Element-wise rich less than or equal comparison between values from operand t1 with respect to values of
    operand t2 (i.e. t1 <= t2), not commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be compared as argument.

    Parameters
    ----------
    t1: tensor or scalar
       The first operand to be compared less than or equal to second operand
    t2: tensor or scalar
       The second operand to be compared greater than or equal to first operand

    Returns
    -------
    result: ht.DNDarray
       A uint8-tensor holding 1 for all elements in which values of t1 are less than or equal to values of t2,
       0 for all other elements

    Examples
    -------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2],[3, 4]])
    >>> ht.le(T1, 3.0)
    tensor([[1, 1],
            [1, 0]], dtype=torch.uint8)

    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.le(T1, T2)
    tensor([[1, 1],
            [0, 0]], dtype=torch.uint8)
    """
    return operations.__binary_op(torch.le, t1, t2)


def lt(t1, t2):
    """
    Element-wise rich less than comparison between values from operand t1 with respect to values of
    operand t2 (i.e. t1 < t2), not commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be compared as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand to be compared less than second operand

    t2: tensor or scalar
        The second operand to be compared greater than first operand

    Returns
    -------
    result: ht.DNDarray
        A uint8-tensor holding 1 for all elements in which values of t1 are less than values of t2,
        0 for all other elements

    Examples
    -------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2],[3, 4]])
    >>> ht.lt(T1, 3.0)
    tensor([[1, 1],
            [0, 0]], dtype=torch.uint8)
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.lt(T1, T2)
    tensor([[1, 0],
            [0, 0]], dtype=torch.uint8)
    """
    return operations.__binary_op(torch.lt, t1, t2)


def ne(t1, t2):
    """
    Element-wise rich comparison of non-equality between values from two operands, commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be compared as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand involved in the comparison
    t2: tensor or scalar
        The second operand involved in the comparison

    Returns
    -------
    result: ht.DNDarray
        A uint8-tensor holding 1 for all elements in which values of t1 are not equal to values of t2,
        0 for all other elements

    Examples:
    ---------
    >>> import heat as ht
    >>> T1 = ht.float32([[1, 2],[3, 4]])
    >>> ht.ne(T1, 3.0)
    tensor([[1, 1],
            [0, 1]])

    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.ne(T1, T2)
    tensor([[1, 0],
            [1, 1]])
    """
    return operations.__binary_op(torch.ne, t1, t2)
