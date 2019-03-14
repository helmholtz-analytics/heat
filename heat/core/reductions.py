import itertools
import torch
import warnings
import numpy as np

from .communication import MPI
from . import stride_tricks
from . import types
from . import tensor
from .operations import __local_operation as local_op
from .operations import __reduce_op as reduce_op


__all__ = [
    'sum',
]


def sum(x, axis=None, out=None):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    x : ht.tensor
        Input data.

    axis : None or int, optional
        Axis along which a sum is performed. The default, axis=None, will sum
        all of the elements of the input array. If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    sum_along_axis : ht.tensor
        An array with the same shape as self.__array except for the specified axis which
        becomes one, e.g. a.shape = (1, 2, 3) => ht.ones((1, 2, 3)).sum(axis=1).shape = (1, 1, 3)

    Examples
    --------
    >>> ht.sum(ht.ones(2))
    tensor([2.])

    >>> ht.sum(ht.ones((3,3)))
    tensor([9.])

    >>> ht.sum(ht.ones((3,3)).astype(ht.int))
    tensor([9])

    >>> ht.sum(ht.ones((3,2,1)), axis=-3)
    tensor([[[3.],
            [3.]]])
    """
    # TODO: make me more numpy API complete Issue #101
    return reduce_op(x, torch.sum, MPI.SUM, axis, out)
