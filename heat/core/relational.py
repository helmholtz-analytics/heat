import torch
import numpy as np

from . import factories
from . import operations
from . import tensor

__all__ = [
    'eq',
    'equal',
    'ge',
    'gt',
    'le',
    'lt',
    'ne'
]


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
    result: ht.Tensor
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
    if np.isscalar(t1):
        try:
            t1 = factories.array([t1])
        except (TypeError, ValueError,):
            raise TypeError('Data type not supported, input was {}'.format(type(t1)))

        if np.isscalar(t2):
            try:
                t2 = factories.array([t2])
            except (TypeError, ValueError,):
                raise TypeError('Only numeric scalars are supported, but input was {}'.format(type(t2)))
        elif isinstance(t2, tensor.Tensor):
            pass
        else:
            raise TypeError('Only tensors and numeric scalars are supported, but input was {}'.format(type(t2)))

    elif isinstance(t1, tensor.Tensor):
        if np.isscalar(t2):
            try:
                t2 = factories.array([t2])
            except (TypeError, ValueError,):
                raise TypeError('Data type not supported, input was {}'.format(type(t2)))
        elif isinstance(t2, tensor.Tensor):
            # TODO: implement complex NUMPY rules
            if t2.split is None or t2.split == t1.split:
                pass
            else:
                # It is NOT possible to perform binary operations on tensors with different splits, e.g. split=0 and split=1
                raise NotImplementedError('Not implemented for other splittings')
        else:
            raise TypeError('Only tensors and numeric scalars are supported, but input was {}'.format(type(t2)))
    else:
        raise NotImplementedError('Not implemented for non scalar')

    result = torch.equal(t1._Tensor__array, t2._Tensor__array)
    return result


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
    result: ht.Tensor
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
    result: ht.Tensor
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
    result: ht.Tensor
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
    result: ht.Tensor
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
    result: ht.Tensor
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
