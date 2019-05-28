import torch

from . import dndarray
from . import stride_tricks

__all__ = [
    'expand_dims',
    'sort'
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
    partial = a._DNDarray__array
    if not a.split or axis != a.split:
        # sorting is not affected by split -> we can just sort along the axis
        partial, index = torch.sort(a._DNDarray__array, dim=axis, descending=descending)

    else:
        # sorting is affected by split, process need to communicate results
        # transpose so we can work along the 0 axis
        transposed = a._DNDarray__array.transpose(axis, 0)
        print("transposed", transposed)
        local_sorted, _ = torch.sort(transposed, dim=0, descending=descending)
        print("local_sorted", local_sorted)

        size = a.comm.Get_size()
        rank = a.comm.Get_rank()
        length = local_sorted.size()[0]
        print("length", length)

        # Separate the sorted tensor into size + 1 equal length partitions
        partitions = [x * length // (size + 1) for x in range(1, size + 1)]
        print("partitions", partitions)
        local_pivots = local_sorted[partitions]
        print("local_pivots", local_pivots)
        pivot_dim = list(transposed.size())
        pivot_dim[0] = size * size
        print("pivot_dim", pivot_dim)

        # share the local pivots with root process
        pivot_buffer = torch.empty(pivot_dim, dtype=a.dtype.torch_type())
        a.comm.Gather(local_pivots, pivot_buffer, root=0)
        print("Gathered pivot_buffer", pivot_buffer)

        pivot_dim[0] = size - 1
        global_pivots = torch.empty(pivot_dim, dtype=a.dtype.torch_type())

        # root process creates new pivots and shares them with other processes
        if rank is 0:
            sorted_pivots, _ = torch.sort(pivot_buffer, dim=0)
            length = sorted_pivots.size()[0]
            global_partitions = [x * length // size for x in range(1, size)]
            print("global_partitions", global_partitions)
            global_pivots = sorted_pivots[global_partitions]
        print("global_pivots", global_pivots)

        a.comm.Bcast(global_pivots, root=0)

        print("Bcas global_pivots", global_pivots)

        # TODO: wirte algorithm that creates a 2D list with the indices for each column (is it also 2D for higher dimensions?)
        # # compute starting locations
        # partition_list = [[0] * size]
        # index = 0
        # for num, val in enumerate(local_sorted):
        #     if val > global_pivots[index]:

    if out:
        out._DND__array = partial
    else:
        return dndarray.DNDarray(
            partial,
            a.gshape,
            a.dtype,
            a.split,
            a.device,
            a.comm
        )