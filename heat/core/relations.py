import torch
import numpy as np


from .communication import MPI
from . import tensor
from .operations import __reduce_op as reduce_op
from .operations import __binary_op as binary_op

__all__ = [
    'eq',
    'equal',
    'ge',
    'gt',
    'le',
    'lt',
    'max',
    'min',
    'ne'
]

def eq(t1,t2):
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
         result: ht.tensor
         A uint8-tensor holding 1 for all elements in which values of t1 are equal to values of t2,
         0 for all other elements

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

    return binary_op(torch.eq, t1, t2)


def equal(t1,t2):
    """
         Overall comparison of equality between two tensors. Returns True if two tensors have the same size and elements, and False otherwise.

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
            t1 = tensor.array([t1])
        except (TypeError,ValueError,):
            raise TypeError('Data type not supported, input was {}'.format(type(t1)))

        if np.isscalar(t2):
            try:
                t2 = tensor.array([t2])
            except (TypeError,ValueError,):
                raise TypeError('Only numeric scalars are supported, but input was {}'.format(type(t2)))
        elif isinstance(t2, tensor.tensor):
            pass
        else:
            raise TypeError('Only tensors and numeric scalars are supported, but input was {}'.format(type(t2)))

    elif isinstance(t1, tensor.tensor):

        if np.isscalar(t2):
            try:
                t2 = tensor.array([t2])
            except (TypeError,ValueError,):
                raise TypeError('Data type not supported, input was {}'.format(type(t2)))
        elif isinstance(t2, tensor.tensor):
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

    result = torch.equal(t1._tensor__array, t2._tensor__array)

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
         result: ht.tensor
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

    return binary_op(torch.ge, t1, t2)


def gt(t1,t2):
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
         result: ht.tensor
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

    return binary_op(torch.gt, t1, t2)


def le(t1,t2):
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
         result: ht.tensor
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
    return binary_op(torch.le, t1, t2)


def lt(t1,t2):
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
         result: ht.tensor
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

    return binary_op(torch.lt, t1, t2)

def max(x, axis=None, out=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the maximum along a given axis.

    Parameters
    ----------
    a : ht.tensor
        Input data.
    axis : None or int, optional
        Axis or axes along which to operate. By default, flattened input is used.
    out : ht.tensor, optional
        Tuple of two output tensors (max, max_indices). Must be of the same shape and buffer length as the expected
        output. The minimum value of an output element. Must be present to allow computation on empty slice.

    Examples
    --------
    >>> a = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
    >>> ht.max(a)
    tensor([12.])
    >>> ht.min(a, axis=0)
    tensor([[10., 11., 12.]])
    >>> ht.min(a, axis=1)
    tensor([[ 3.],
        [ 6.],
        [ 9.],
        [12.]])
    """
    def local_max(*args, **kwargs):
        result = torch.max(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    return reduce_op(x, local_max, MPI.MAX, axis, out)


def min(x, axis=None, out=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    a : ht.tensor
        Input data.
    axis : None or int
        Axis or axes along which to operate. By default, flattened input is used.
    out : ht.tensor, optional
        Tuple of two output tensors (min, min_indices). Must be of the same shape and buffer length as the expected
        output.The maximum value of an output element. Must be present to allow computation on empty slice.

    Examples
    --------
    >>> a = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
    >>> ht.min(a)
    tensor([1.])
    >>> ht.min(a, axis=0)
    tensor([[1., 2., 3.]])
    >>> ht.min(a, axis=1)
    tensor([[ 1.],
        [ 4.],
        [ 7.],
        [10.]])
    """
    def local_min(*args, **kwargs):
        result = torch.min(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    return reduce_op(x, local_min, MPI.MIN, axis, out)



def ne(t1,t2):
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
         result: ht.tensor
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

    return binary_op(torch.ne, t1, t2)

