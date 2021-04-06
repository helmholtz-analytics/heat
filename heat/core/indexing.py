"""
functions related to finding indexes of values, etc.
"""

import torch
from typing import List, Dict, Any, TypeVar, Union, Tuple, Sequence

from .communication import MPI
from .dndarray import DNDarray
from . import factories
from . import sanitation
from . import types

__all__ = ["nonzero", "where"]


def nonzero(a: DNDarray) -> DNDarray:
    """
    Return the indices of the elements that are non-zero. (using ``torch.nonzero``)
    If ``a`` is split then the result is split in the 0th dimension. However, this :class:`~heat.core.dndarray.DNDarray`
    can be UNBALANCED as it contains the indices of the non-zero elements on each node.
    Returns an array with one entry for each dimension of ``a``, containing the indices of the non-zero elements in that dimension.
    The values in ``a`` are always tested and returned in row-major, C-style order.
    The corresponding non-zero values can be obtained with: ``a[nonzero(a)]``.

    Parameters
    ----------
    a: DNDarray
        Input array

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
    [0/1] tensor([[1, 2, 3],
    [0/1]         [4, 5, 6]])
    [1/1] tensor([[7, 8, 9]])
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
    >>> a[ht.nonzero(a > 3)]
    [0/1] tensor([[4, 5, 6]])
    [1/1] tensor([[7, 8, 9]])
    """
    sanitation.sanitize_in(a)

    if a.dtype == types.bool:
        a.larray = a.larray.float()
    if a.split is None:
        # if there is no split then just return the values from torch
        # print(a.larray)
        lcl_nonzero = torch.nonzero(input=a.larray, as_tuple=False)
        gout = list(lcl_nonzero.size())
        is_split = None
    else:
        # a is split
        lcl_nonzero = torch.nonzero(input=a.larray, as_tuple=False)
        _, _, slices = a.comm.chunk(a.shape, a.split)
        lcl_nonzero[..., a.split] += slices[a.split].start
        gout = list(lcl_nonzero.size())
        gout[0] = a.comm.allreduce(gout[0], MPI.SUM)
        is_split = 0

    if a.ndim == 1:
        lcl_nonzero = lcl_nonzero.squeeze(dim=1)
    for g in range(len(gout) - 1, -1, -1):
        if gout[g] == 1:
            del gout[g]

    return DNDarray(
        lcl_nonzero,
        gshape=tuple(gout),
        dtype=types.canonical_heat_type(lcl_nonzero.dtype),
        split=is_split,
        device=a.device,
        comm=a.comm,
        balanced=False,
    )


DNDarray.nonzero = lambda self: nonzero(self)
DNDarray.nonzero.__doc__ = nonzero.__doc__


def where(
    cond: DNDarray,
    x: Union[None, int, float, DNDarray] = None,
    y: Union[None, int, float, DNDarray] = None,
) -> DNDarray:
    """
    Return elements chosen from ``x`` or ``y`` depending on condition.
    Result is a :class:`~heat.core.dndarray.DNDarray` with elements from ``x`` where cond is ``True``,
    and elements from ``y`` elsewhere (``False``).

    Parameters
    ----------
    cond : DNDarray
        Condition of interest, where true yield ``x`` otherwise yield ``y``
    x : DNDarray or int or float, optional
        Values from which to choose. ``x``, ``y`` and condition need to be broadcastable to some shape.
    y : DNDarray or int or float, optional
        Values from which to choose. ``x``, ``y`` and condition need to be broadcastable to some shape.

    Notes
    -------
    When only condition is provided, this function is a shorthand for :func:`nonzero`.

    Examples
    --------
    >>> a = ht.arange(10, split=0)
    [0/1] tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    [1/1] tensor([5, 6, 7, 8, 9], dtype=torch.int32)
    >>> ht.where(a < 5, a, 10*a)
    [0/1] tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    [1/1] tensor([50, 60, 70, 80, 90], dtype=torch.int32)
    >>> a = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    [0/1] tensor([[ 0.,  1.,  2.],
    [0/1]         [ 0.,  2.,  4.]])
    [1/1] tensor([[ 0.,  3.,  6.]])
    >>> ht.where(a < 4, a, -1)
    [0/1] tensor([[ 0.,  1.,  2.],
    [0/1]         [ 0.,  2., -1.]])
    [1/1] tensor([[ 0.,  3., -1.]])

    """
    if cond.split is not None and (isinstance(x, DNDarray) or isinstance(y, DNDarray)):
        if (isinstance(x, DNDarray) and cond.split != x.split) or (
            isinstance(y, DNDarray) and cond.split != y.split
        ):
            if len(y.shape) >= 1 and y.shape[0] > 1:
                raise NotImplementedError("binary op not implemented for different split axes")
    if isinstance(x, (DNDarray, int, float)) and isinstance(y, (DNDarray, int, float)):
        for var in [x, y]:
            if isinstance(var, int):
                var = float(var)
        return cond.dtype(cond == 0) * y + cond * x
    elif x is None and y is None:
        return nonzero(cond)
    else:
        raise TypeError(
            "either both or neither x and y must be given and both must be DNDarrays or ints({}, {})".format(
                type(x), type(y)
            )
        )
