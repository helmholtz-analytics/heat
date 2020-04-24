import torch

from .communication import MPI
from . import dndarray
from . import factories
from . import types

__all__ = ["nonzero", "where"]


def nonzero(a):
    """
    Return the indices of the elements that are non-zero. (using torch.nonzero)

    Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.
    The values in a are always tested and returned in row-major, C-style order. The corresponding non-zero values can be obtained with: a[nonzero(a)].

    Parameters
    ----------
    a: ht.DNDarray
        Input array

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
    if a.dtype == types.bool:
        a._DNDarray__array = a._DNDarray__array.float()
    if a.split is None:
        # if there is no split then just return the values from torch
        # print(a._DNDarray__array)
        lcl_nonzero = torch.nonzero(input=a._DNDarray__array, as_tuple=False)
        gout = list(lcl_nonzero.size())
        is_split = None
    else:
        # a is split
        lcl_nonzero = torch.nonzero(input=a._DNDarray__array, as_tuple=False)
        _, _, slices = a.comm.chunk(a.shape, a.split)
        lcl_nonzero[..., a.split] += slices[a.split].start
        gout = list(lcl_nonzero.size())
        gout[0] = a.comm.allreduce(gout[0], MPI.SUM)
        is_split = 0

    if a.numdims == 1:
        lcl_nonzero = lcl_nonzero.squeeze(dim=1)

    return dndarray.DNDarray(
        lcl_nonzero,
        gshape=tuple(gout),
        dtype=types.canonical_heat_type(lcl_nonzero.dtype),
        split=is_split,
        device=a.device,
        comm=a.comm,
    )


def where(cond, x=None, y=None):
    """
    Return elements chosen from x or y depending on condition.
    **NOTE** When only condition is provided, this function is a shorthand for ht.nonzero(cond).

    Parameters
    ----------
    cond: DNDarray
        condition of interest, where true yield x otherwise yield y
    x, y: DNDarray, int, or float
        Values from which to choose. x, y and condition need to be broadcastable to some shape.

    Returns
    -------
    out: DNDarray
        A DNDarray with elements from x where cond is True(1), and elements from y elsewhere (False/0).

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
    if cond.split is not None and (
        isinstance(x, dndarray.DNDarray) or isinstance(y, dndarray.DNDarray)
    ):
        if (isinstance(x, dndarray.DNDarray) and cond.split != x.split) or (
            isinstance(y, dndarray.DNDarray) and cond.split != y.split
        ):
            if len(y.shape) >= 1 and y.shape[0] > 1:
                raise NotImplementedError("binary op not implemented for different split axes")
    if isinstance(x, (dndarray.DNDarray, int, float)) and isinstance(
        y, (dndarray.DNDarray, int, float)
    ):
        cond = types.float(cond, device=cond.device)
        return types.float(cond == 0, device=cond.device) * y + cond * x
    elif x is None and y is None:
        return nonzero(cond)
    else:
        raise TypeError(
            "either both or neither x and y must be given and both must be DNDarrays or ints({}, {})".format(
                type(x), type(y)
            )
        )
