import torch
import numpy as np

from . import dndarray
from . import factories
from . import operations
from . import stride_tricks
from . import types

__all__ = [
    'nonzero',
    'where'
]


def nonzero(a):
    """
    Return the indices of the elements that are non-zero. (using torch.nonzero)

    Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.
    The values in a are always tested and returned in row-major, C-style order. The corresponding non-zero values can be obtained with: a[nonzero(a)].

    Parameters
    ----------
    a: ht.DNDarray

    Returns
    -------
    result: ht.DNDarray
        Indices of elements that are non-zero.
        If 'a' is split then the result is split in the 0th dimension. However, this DNDarray can be UNBALANCED as it contains the indices of the
        non-zero elements on each node.

    Examples
    --------
    >>> x = ht.array([[3, 0, 0], [0, 4, 1], [0, 6, 0]], split=0)
    [0/2] tensor([[3, 0, 0]])
    [1/2] tensor([[0, 4, 1]])
    [2/2] tensor([[0, 6, 0]])
    >>> ht.nonzero(x)
    [0/2] tensor([[0, 0]])
    [1/2] tensor([[1, 1],
    [1/2]         [1, 2]])
    [2/2] tensor([[2, 1]])

    >>> a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=0)
    >>> a > 3
    [0/1] tensor([[0, 0, 0],
    [0/1]         [1, 1, 1]], dtype=torch.uint8)
    [1/1] tensor([[1, 1, 1]], dtype=torch.uint8)
    >>> ht.nonzero(a > 3)
    [0/1] tensor([[1, 0],
    [0/1]         [1, 1],
    [0/1]         [1, 2]])
    [1/1] tensor([[2, 0],
    [1/1]         [2, 1],
    [1/1]         [2, 2]])

    """

    if a.split is None:
        # if there is no split then just return the values from torch
        return operations.__local_op(torch.nonzero, a, out=None)
    else:
        # a is split
        lcl_nonzero = torch.nonzero(a._DNDarray__array)
        _, _, slices = a.comm.chunk(a.shape, a.split)
        lcl_nonzero[..., a.split] += slices[a.split].start

        return factories.array(lcl_nonzero, is_split=0, dtype=types.int)


def where(cond, x=None, y=None):
    """
    mirror of the numpy where function:
    Return elements chosen from x or y depending on condition.
    **NOTE** When only condition is provided, this function is a shorthand for np.asarray(condition).nonzero(). Using nonzero directly should be preferred

    Parameters
    ----------
    cond:
    x, y:

    Returns
    -------

    """
    if isinstance(x, (dndarray.DNDarray, int, float)) and isinstance(y, (dndarray.DNDarray, int, float)):
        return cond * x + (cond == 0) * y
    elif x is None and y is None:
        return nonzero(cond)
    else:
        raise TypeError("either both or neither x and y must be given and both must be DNDarrays or ints({}, {})".format(type(x), type(y)))
