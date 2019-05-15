import torch

from . import dndarray
from . import stride_tricks

__all__ = [
    'expand_dims'
]


def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the axis position in the expanded array shape.

    Parameters
    ----------
    a : ht.DNDarray
        Input array to be expanded.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : ht.DNDarray
        Output array. The number of dimensions is one greater than that of the input array.

    Raises
    ------
    ValueError
        If the axis is not in range of the axes.

    Examples
    --------
    >>> x = ht.array([1,2])
    >>> x.shape
    (2,)

    >>> y = ht.expand_dims(x, axis=0)
    >>> y
    array([[1, 2]])
    >>> y.shape
    (1, 2)

    y = ht.expand_dims(x, axis=1)
    >>> y
    array([[1],
           [2]])
    >>> y.shape
    (2, 1)
    """
    # ensure type consistency
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError('expected ht.DNDarray, but was {}'.format(type(a)))

    # sanitize axis, introduce arbitrary dummy dimension to model expansion
    axis = stride_tricks.sanitize_axis(a.shape + (1,), axis)

    return dndarray.DNDarray(
        a._DNDarray__array.unsqueeze(dim=axis), a.shape[:axis] + (1,) + a.shape[axis:],
        a.dtype,
        a.split if a.split is None or a.split < axis else a.split + 1,
        a.device,
        a.comm
    )


def sort(a, axis=None, descending=False, out=None):
    # default: using last axis
    if axis is None:
        axis = len(a.shape) - 1

    if not a.split or axis != a.split:
        partial = torch.sort(a._DNDarray__array, dim=axis, descending=descending)

    else:
        transposed = a._DNDarray__array.transpose(0, axis)
        local_sorted = torch.sort(a._DNDarray__array, dim=0, descending=descending)

        size = a.comm.Get_size()
        rank = a.comm.Get_rank()
        length = local_sorted.size()[axis]
        # Separate the sorted tensor into size + 1 equal length partitions
        partitions = [x * length // (size + 1) for x in range(1, size + 1)]
        local_pivots = local_sorted

        # Share pivot elements with root process
        pivot_buffer = None
        if rank is 0:
            pivot_buffer = torch.empty((size * size, ), dtype=a.dtype.torch_type())
        a.comm.Gather(local_pivots, pivot_buffer, root=0)

        global_pivots = torch.empty((size - 1,), dtype=a.dtype.torch_type())
        if rank is 0:
            # sorting the random pivots for the global partitioning
            sorted_pivots = torch.sort(pivot_buffer, dim=0, descending=descending)
            length = sorted_pivots.size()[axis]
            global_partitions = [x * length // size for x in range(1, size)]
            global_pivots = sorted_pivots[global_partitions]

        a.comm.Bcast(global_pivots, root=0)

        # compute starting locations
        partition_list = [[0] * size]
        index = 0
        # for val in local_sorted:



    if out:
        out._DND__array = partial
    else:
        return dndarray.DNDarray(
            partial,
            a.dtype,
            a.split,
            a.device,
            a.comm
        )