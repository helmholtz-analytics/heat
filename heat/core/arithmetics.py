import torch

from . import tensor
from .operations import __binary_op as binary_op


__all__ = [
    'add',
    'div',
    'mul',
    'pow',
    'sub'
]

def add(t1, t2):
    """
    Element-wise addition of values from two operands, commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be added as argument.

    Parameters
    ----------
    t1: tensor or scalar
    The first operand involved in the addition

    t2: tensor or scalar
    The second operand involved in the addition


    Returns:
    -------
    result: ht.tensor
    A tensor containing the results of element-wise addition of t1 and t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.add ( 1.0, 4.0)
    tensor([5.])

    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.add(T1, T2)
    tensor([[3., 4.],
            [5., 6.]])

    >>> s = 2.0
    >>> ht.add(T1, s)
    tensor([[3., 4.],
            [5., 6.]])

    """

    return binary_op(torch.add, t1, t2)

def div(t1, t2):
    """
        Element-wise true division of values of operand t1 by values of operands t2 (i.e t1 / t2), not commutative.
        Takes the two operands (scalar or tensor) whose elements are to be divided (operand 1 by operand 2)
        as argument.

        Parameters
        ----------
        t1: tensor or scalar
        The first operand whose values are divided

        t2: tensor or scalar
        The second operand by whose values is divided


        Returns
        -------
        result: ht.tensor
        A tensor containing the results of element-wise true division (i.e. floating point values) of t1 by t2.


        Examples:
        ---------
        >>> import heat as ht
        >>> ht.div(2.0, 2.0)
        tensor([1.])

        >>> T1 = ht.float32([[1, 2],[3, 4]])
        >>> T2 = ht.float32([[2, 2], [2, 2]])
        >>> ht.div(T1, T2)
        tensor([[0.5000, 1.0000],
                [1.5000, 2.0000]])

        >>> s = 2.0
        >>> ht.div(s, T1)
        tensor([[2.0000, 1.0000],
        [0.6667, 0.5000]])
        """

    return binary_op(torch.div, t1, t2)

def mul(t1,t2):
    """
      Element-wise multiplication (NOT matrix multiplication) of values from two operands, commutative.
      Takes the first and second operand (scalar or tensor) whose elements are to be multiplied as argument.

      Parameters
      ----------
      t1: tensor or scalar
      The first operand involved in the multiplication

      t2: tensor or scalar
      The second operand involved in the multiplication


      Returns
      -------
      result: ht.tensor
      A tensor containing the results of element-wise multiplication of t1 and t2.

     Examples:
     ---------
     >>> import heat as ht
     >>> ht.mul ( 2.0, 4.0)
     tensor([8.])

     >>> T1 = ht.float32([[1, 2], [3, 4]])
     >>> s = 3.0
     >>> ht.mul(T1, s)
     tensor([[3., 6.],
             [9., 12.]])

     >>> T2 = ht.float32([[2, 2], [2, 2]])
     >>> ht.mul(T1, T2)
     tensor([[2., 4.],
             [6., 8.]])

    """

    return binary_op(torch.mul, t1, t2)

def pow(t1,t2):
    """
        Element-wise exponential function of values of operand t1 to the power of values of operand t2 (i.e t1 ** t2),
        not commutative. Takes the two operands (scalar or tensor) whose elements are to be involved in the exponential
        function(operand 1 to the power of operand 2)
        as argument.

        Parameters
        ----------
        t1: tensor or scalar
        The first operand whose values represent the base

        t2: tensor or scalar
        The second operand by whose values represent the exponent


        Returns
        -------
        result: ht.tensor
        A tensor containing the results of element-wise exponential function.

        Examples:
        ---------
        >>> import heat as ht
        >>> ht.pow ( 3.0, 2.0)
        tensor([9.])

        >>> T1 = ht.float32([[1, 2], [3, 4]])
        >>> T2 = ht.float32([[3, 3], [2, 2]])
        >>> ht.pow(T1, T2)
        tensor([[1., 8.],
                [9., 16.]])
        >>> s = 3.0
        >>> ht.pow(T1, s)
        tensor([[1., 8.],
                [27., 64.]])

        """

    return binary_op(torch.pow, t1, t2)


def sub(t1, t2):
    """
      Element-wise subtraction of values of operand t2 from values of operands t1 (i.e t1 - t2), not commutative.
      Takes the two operands (scalar or tensor) whose elements are to be subtracted (operand 2 from operand 1)
      as argument.

      Parameters
      ----------
      t1: tensor or scalar
      The first operand from which values are subtracted

      t2: tensor or scalar
      The second operand whose values are subtracted

      Returns
      -------
      result: ht.tensor
      A tensor containing the results of element-wise subtraction of t1 and t2.

      Examples:
      ---------
      >>> import heat as ht
      >>> ht.sub ( 4.0, 1.0)
      tensor([3.])

      >>> T1 = ht.float32([[1, 2], [3, 4]])
      >>> T2 = ht.float32([[2, 2], [2, 2]])
      >>> ht.sub(T1, T2)
      tensor([[-1., 0.],
              [1., 2.]])

      >>> s = 2.0
      >>> ht.sub(s, T1)
      tensor([[ 1.,  0.],
              [-1., -2.]])

      """

    return binary_op(torch.sub, t1, t2)
