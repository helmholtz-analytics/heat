"""
Functions relating to indices of items within DNDarrays, i.e. `where()`
"""

import torch
from typing import List, Dict, Any, TypeVar, Union, Tuple, Sequence

from .communication import MPI
from .dndarray import DNDarray
from . import sanitation
from . import types

__all__ = ["nonzero", "where"]


def nonzero(x: DNDarray) -> DNDarray:
    """
    Return a :class:`~heat.core.dndarray.DNDarray` containing the indices of the elements that are non-zero.. (using ``torch.nonzero``)
    If ``x`` is split then the result is split in the 0th dimension. However, this :class:`~heat.core.dndarray.DNDarray`
    can be UNBALANCED as it contains the indices of the non-zero elements on each node.
    Returns an array with one entry for each dimension of ``x``, containing the indices of the non-zero elements in that dimension.
    The values in ``x`` are always tested and returned in row-major, C-style order.
    The corresponding non-zero values can be obtained with: ``x[nonzero(x)]``.

    Parameters
    ----------
    x: DNDarray
        Input array

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.array([[3, 0, 0], [0, 4, 1], [0, 6, 0]], split=0)
    >>> ht.nonzero(x)
    DNDarray([[0, 0],
              [1, 1],
              [1, 2],
              [2, 1]], dtype=ht.int64, device=cpu:0, split=0)
    >>> y = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=0)
    >>> y > 3
    DNDarray([[False, False, False],
              [ True,  True,  True],
              [ True,  True,  True]], dtype=ht.bool, device=cpu:0, split=0)
    >>> ht.nonzero(y > 3)
    DNDarray([[1, 0],
              [1, 1],
              [1, 2],
              [2, 0],
              [2, 1],
              [2, 2]], dtype=ht.int64, device=cpu:0, split=0)
    >>> y[ht.nonzero(y > 3)]
    DNDarray([4, 5, 6, 7, 8, 9], dtype=ht.int64, device=cpu:0, split=0)
    """
    sanitation.sanitize_in(x)

    if x.split is None:
        # if there is no split then just return the values from torch
        lcl_nonzero = torch.nonzero(input=x.larray, as_tuple=False)
        gout = list(lcl_nonzero.size())
        is_split = None
    else:
        # a is split
        lcl_nonzero = torch.nonzero(input=x.larray, as_tuple=False)
        _, _, slices = x.comm.chunk(x.shape, x.split)
        lcl_nonzero[..., x.split] += slices[x.split].start
        gout = list(lcl_nonzero.size())
        gout[0] = x.comm.allreduce(gout[0], MPI.SUM)
        is_split = 0

    if x.ndim == 1:
        lcl_nonzero = lcl_nonzero.squeeze(dim=1)
    for g in range(len(gout) - 1, -1, -1):
        if gout[g] == 1:
            del gout[g]

    return DNDarray(
        lcl_nonzero,
        gshape=tuple(gout),
        dtype=types.canonical_heat_type(lcl_nonzero.dtype),
        split=is_split,
        device=x.device,
        comm=x.comm,
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
    Return a :class:`~heat.core.dndarray.DNDarray` containing elements chosen from ``x`` or ``y`` depending on condition.
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

    Raises
    ------
    NotImplementedError
        if splits of the two input :class:`~heat.core.dndarray.DNDarray` differ
    TypeError
        if only x or y is given or both are not DNDarrays or numerical scalars

    Notes
    -------
    When only condition is provided, this function is a shorthand for :func:`nonzero`.

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.arange(10, split=0)
    >>> ht.where(x < 5, x, 10*x)
    DNDarray([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90], dtype=ht.int64, device=cpu:0, split=0)
    >>> y = ht.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    >>> ht.where(y < 4, y, -1)
    DNDarray([[ 0,  1,  2],
              [ 0,  2, -1],
              [ 0,  3, -1]], dtype=ht.int64, device=cpu:0, split=None)
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
            f"either both or neither x and y must be given and both must be DNDarrays or numerical scalars({type(x)}, {type(y)})"
        )
