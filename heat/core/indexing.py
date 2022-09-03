"""
Functions relating to indices of items within DNDarrays, i.e. `where()`
"""

import torch
from typing import List, Dict, Any, TypeVar, Union, Tuple, Sequence

from .communication import MPI
from .dndarray import DNDarray
from . import sanitation
from . import types
from . import manipulations

__all__ = ["nonzero", "where"]


def nonzero(x: DNDarray) -> Tuple[DNDarray, ...]:
    """
    TODO: UPDATE DOCS!
    Return a Tuple of :class:`~heat.core.dndarray.DNDarray`s, one for each dimension of ``x``,
    containing the indices of the non-zero elements in that dimension. If ``x`` is split then
    the result is split in the 0th dimension. However, this :class:`~heat.core.dndarray.DNDarray`
    can be UNBALANCED as it contains the indices of the non-zero elements on each node.
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
    (DNDarray([0, 1, 1, 2], dtype=ht.int64, device=cpu:0, split=None),
        DNDarray([0, 1, 2, 1], dtype=ht.int64, device=cpu:0, split=None))
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
    (DNDarray([1, 1, 1, 2, 2, 2], dtype=ht.int64, device=cpu:0, split=None),
        DNDarray([0, 1, 2, 0, 1, 2], dtype=ht.int64, device=cpu:0, split=None))
    >>> y[ht.nonzero(y > 3)]
    DNDarray([4, 5, 6, 7, 8, 9], dtype=ht.int64, device=cpu:0, split=0)
    """
    try:
        local_x = x.larray
    except AttributeError:
        raise TypeError("Input must be a DNDarray, is {}".format(type(x)))

    if not x.is_distributed():
        # nonzero indices as tuple
        lcl_nonzero = torch.nonzero(input=local_x, as_tuple=True)
        # bookkeeping for final DNDarray construct
        output_shape = (lcl_nonzero[0].shape,)
        output_split = None if x.split is None else 0
        output_balanced = True
    else:
        lcl_nonzero = torch.nonzero(input=local_x, as_tuple=False)
        nonzero_size = torch.tensor(
            lcl_nonzero.shape[0], dtype=torch.int64, device=lcl_nonzero.device
        )
        # global nonzero_size
        x.comm.Allreduce(MPI.IN_PLACE, nonzero_size, MPI.SUM)
        # correct indices along split axis
        _, displs = x.counts_displs()
        lcl_nonzero[:, x.split] += displs[x.comm.rank]

        if x.split != 0:
            # construct global 2D DNDarray of nz indices:
            shape_2d = (nonzero_size.item(), x.ndim)
            global_nonzero = DNDarray(
                lcl_nonzero,
                gshape=shape_2d,
                dtype=types.int64,
                split=0,
                device=x.device,
                comm=x.comm,
                balanced=False,
            )
            # stabilize distributed result: vectorized sorting of nz indices along axis 0
            global_nonzero.balance_()
            global_nonzero = manipulations.unique(global_nonzero, axis=0)
            # return indices as tuple of columns
            lcl_nonzero = global_nonzero.larray.split(1, dim=1)
            output_balanced = True
        else:
            # return indices as tuple of columns
            lcl_nonzero = lcl_nonzero.split(1, dim=1)
            output_balanced = False

    # return global_nonzero as tuple of DNDarrays
    global_nonzero = list(lcl_nonzero)
    output_shape = (nonzero_size.item(),)
    output_split = 0
    for i, nz_tensor in enumerate(global_nonzero):
        if nz_tensor.ndim > 1:
            # extra dimension in distributed case from usage of torch.split()
            nz_tensor = nz_tensor.squeeze()
        nz_array = DNDarray(
            nz_tensor,
            gshape=output_shape,
            dtype=types.int64,
            split=output_split,
            device=x.device,
            comm=x.comm,
            balanced=output_balanced,
        )
        global_nonzero[i] = nz_array
    global_nonzero = tuple(global_nonzero)

    return tuple(global_nonzero)


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
            "either both or neither x and y must be given and both must be DNDarrays or numerical scalars({}, {})".format(
                type(x), type(y)
            )
        )
