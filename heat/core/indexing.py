"""
Functions relating to indices of items within DNDarrays, i.e. `where()`
"""

import torch

from .communication import MPI
from .dndarray import DNDarray
from . import factories
from .sanitation import sanitize_in
from . import types
from . import manipulations

__all__ = ["nonzero", "where"]


def nonzero(x: DNDarray, as_tuple: bool = True) -> tuple[DNDarray, ...] | DNDarray:
    """
    Return a tuple of :class:`~heat.core.dndarray.DNDarray`s, one for each dimension of ``x``,
    containing the indices of the non-zero elements in that dimension. If ``x`` is split then
    the result is split in the first dimension. However, this :class:`~heat.core.dndarray.DNDarray`
    can be UNBALANCED as it contains the indices of the non-zero elements on each node.
    The values in ``x`` are always tested and returned in row-major, C-style order.
    The corresponding non-zero values can be obtained with: ``x[nonzero(x)]``.

    Parameters
    ----------
    x: DNDarray
        Input array
    as_tuple: bool, optional
        Default is True for numpy-style nonzero output. If False, the output is a torch-style single 2D ``DNDarray`` of shape `(num_nonzero, ndim)` containing the indices of the non-zero elements.

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
    sanitize_in(x)

    if not x.is_distributed():
        # nonzero indices as tuple
        nonzero = torch.nonzero(input=x.larray, as_tuple=as_tuple)
        # bookkeeping for final DNDarray construct
        if as_tuple:
            nonzero = list(nonzero)
            for i, nz_tensor in enumerate(nonzero):
                nonzero[i] = factories.array(nz_tensor, device=x.device, comm=x.comm)
            return tuple(nonzero)
        else:
            # nonzero indices as single 2D DNDarray
            return factories.array(nonzero, device=x.device, comm=x.comm)

    # distributed case
    lcl_nonzero = torch.nonzero(input=x.larray, as_tuple=False)
    nonzero_size = torch.tensor(lcl_nonzero.shape[0], dtype=torch.int64, device="cpu")
    nonzero_dtype = types.canonical_heat_type(lcl_nonzero.dtype)

    # global nonzero_size
    x.comm.Allreduce(MPI.IN_PLACE, nonzero_size, MPI.SUM)
    # correct indices along split axis
    _, displs = x.counts_displs()
    lcl_nonzero[:, x.split] += displs[x.comm.rank]

    if x.split == 0:
        # for split=0, the local nonzero indices are already globally ordered along the split axis
        if as_tuple:  # return indices as tuple of 1D DNDarrays
            lcl_nonzero = lcl_nonzero.unbind(dim=1)
            return tuple(
                DNDarray(
                    nz_tensor,
                    gshape=(nonzero_size.item(),),
                    dtype=nonzero_dtype,
                    split=0,
                    device=x.device,
                    comm=x.comm,
                    balanced=False,
                )
                for nz_tensor in lcl_nonzero
            )
        else:  # return indices as single 2D DNDarray
            return DNDarray(
                lcl_nonzero,
                gshape=(nonzero_size.item(), x.ndim),
                dtype=nonzero_dtype,
                split=0,
                device=x.device,
                comm=x.comm,
                balanced=False,
            )
    else:
        # construct global 2D DNDarray of nz indices:
        shape_2d = (nonzero_size.item(), x.ndim)
        global_nonzero = DNDarray(
            lcl_nonzero,
            gshape=shape_2d,
            dtype=nonzero_dtype,
            split=0,
            device=x.device,
            comm=x.comm,
            balanced=False,
        )
        # vectorized sorting of nz indices along axis 0
        global_nonzero.balance_()
        global_nonzero = manipulations.unique(global_nonzero, axis=0)
        if as_tuple:  # return indices as tuple of 1D DNDarrays
            lcl_nonzero = global_nonzero.larray.unbind(dim=1)
            return tuple(
                DNDarray(
                    nz_tensor,
                    gshape=(nonzero_size.item(),),
                    dtype=nonzero_dtype,
                    split=0,
                    device=x.device,
                    comm=x.comm,
                    balanced=True,
                )
                for nz_tensor in lcl_nonzero
            )
        else:  # return indices as single 2D DNDarray
            return global_nonzero


DNDarray.nonzero = lambda self: nonzero(self, as_tuple=True)
DNDarray.nonzero.__doc__ = nonzero.__doc__


def where(
    cond: DNDarray,
    x: None | int | float | DNDarray = None,
    y: None | int | float | DNDarray = None,
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
    -----
    When only condition is provided, this function is a shorthand for :func:`nonzero` and the function returns a tuple
    of :class:`~heat.core.dndarray.DNDarray`, analogously to ``numpy.where``.

    Examples
    --------
    >>> import heat as ht
    >>> x = ht.arange(10, split=0)
    >>> ht.where(x < 5, x, 10 * x)
    DNDarray(MPI-rank: 0, Shape: (10,), Split: 0, Local Shape: (10,), Device: cpu:0, Dtype: int32, Data:
         [ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
    >>> y = ht.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    >>> ht.where(y < 4, y, -1)
    DNDarray(MPI-rank: 0, Shape: (3, 3), Split: None, Local Shape: (3, 3), Device: cpu:0, Dtype: int64, Data:
         [[ 0,  1,  2],
          [ 0,  2, -1],
          [ 0,  3, -1]])
    """
    # binary where(cond, x, y) branch
    if cond.split is not None and isinstance(y, DNDarray) and len(y.shape) >= 1 and y.shape[0] > 1:
        if (isinstance(x, DNDarray) and cond.split != x.split) or cond.split != y.split:
            raise NotImplementedError("binary op not implemented for different split axes")

    if isinstance(x, (DNDarray, int, float)) and isinstance(y, (DNDarray, int, float)):
        # Simple elementwise selection using arithmetic:
        # cond == 0 -> take y, cond == 1 -> take x
        for var in [x, y]:
            if isinstance(var, int):
                var = float(var)
        return cond.dtype(cond == 0) * y + cond * x

    # where(cond) "indices only" branch
    elif x is None and y is None:  # delegate to nonzero(cond)
        return nonzero(cond)  # tuple of DNDarrays, one per dimension

    else:
        raise TypeError(
            "either both or neither x and y must be given and both must be "
            f"DNDarrays or numerical scalars (got {type(x)}, {type(y)})"
        )
