import functools
import operator

import numpy as np
import torch

from . import dndarray
from . import factories
from . import stride_tricks
from . import types
from .communication import MPI

__all__ = [
    'expand_dims',
    'sort',
    'squeeze'
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
    if a.split is None or axis != a.split:
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
            print('sorted_pivots', sorted_pivots)
            length = sorted_pivots.size()[0]
            global_partitions = [x * length // size for x in range(1, size)]
            print("global_partitions", global_partitions)
            global_pivots = sorted_pivots[global_partitions]
        print("global_pivots", global_pivots)

        a.comm.Bcast(global_pivots, root=0)

        print("Bcas global_pivots", global_pivots)

        # Create matrix that holds information which process gets how many values at which position
        zeroes_dim = [size] + list(transposed.size())[1:]
        print('zeros_dim', zeroes_dim)
        partition_matrix = torch.zeros(zeroes_dim, dtype=torch.int64)

        # Iterate along the split axis which is now 0 due to transpose
        for x in local_sorted:
            print('x', x)
            # Enumerate over all values with correct index
            for idx, val in np.ndenumerate(x.numpy()):
                print('index', idx, 'val', val)
                cur = next(i for i in range(len(global_pivots) + 1) if (i == len(global_pivots) or (val < global_pivots[i][idx])))
                print('cur', cur)
                partition_matrix[cur][idx] += 1
        print('partition_matrix', partition_matrix)
        # Tested with 2-4 parallel processes to this point

        # Share and sum the local partition_matrix
        g_partition_matrix = torch.empty(zeroes_dim, dtype=torch.int64)
        a.comm.Allreduce(partition_matrix, g_partition_matrix, op=MPI.SUM)
        print('sum_buf', g_partition_matrix)

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


def squeeze(x, axis=None):
    """
    Remove single-dimensional entries from the shape of a tensor.

    Parameters:
    -----------
    x : ht.DNDarray
        Input data.

    axis : None or int or tuple of ints, optional
           Selects a subset of the single-dimensional entries in the shape.
           If axis is None, all single-dimensional entries will be removed from the shape.
           If an axis is selected with shape entry greater than one, a ValueError is raised.


    Returns:
    --------
    squeezed : ht.DNDarray
               The input tensor, but with all or a subset of the dimensions of length 1 removed.


    Examples:
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    <torch._C.Generator object at 0x115704ad0>
    >>> a = ht.random.randn(1,3,1,5)
    >>> a
    tensor([[[[ 0.2673, -0.4212, -0.5107, -1.5727, -0.1232]],

            [[ 3.5870, -1.8313,  1.5987, -1.2770,  0.3255]],

            [[-0.4791,  1.3790,  2.5286,  0.4107, -0.9880]]]])
    >>> a.shape
    (1, 3, 1, 5)
    >>> ht.squeeze(a).shape
    (3, 5)
    >>> ht.squeeze(a)
    tensor([[ 0.2673, -0.4212, -0.5107, -1.5727, -0.1232],
            [ 3.5870, -1.8313,  1.5987, -1.2770,  0.3255],
            [-0.4791,  1.3790,  2.5286,  0.4107, -0.9880]])
    >>> ht.squeeze(a,axis=0).shape
    (3, 1, 5)
    >>> ht.squeeze(a,axis=-2).shape
    (1, 3, 5)
    >>> ht.squeeze(a,axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: Dimension along axis 1 is not 1 for shape (1, 3, 1, 5)
    """

    # Sanitize input
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError('expected x to be a ht.DNDarray, but was {}'.format(type(x)))
    # Sanitize axis
    axis = stride_tricks.sanitize_axis(x.shape, axis)
    if axis is not None:
        if isinstance(axis, int):
            dim_is_one = (x.shape[axis] == 1)
        if isinstance(axis, tuple):
            dim_is_one = bool(factories.array(list(x.shape[dim] == 1 for dim in axis)).all()._DNDarray__array)
        if not dim_is_one:
            raise ValueError('Dimension along axis {} is not 1 for shape {}'.format(axis, x.shape))

    # Local squeeze
    if axis is None:
        axis = tuple(i for i, dim in enumerate(x.shape) if dim == 1)
    if isinstance(axis, int):
        axis = (axis,)
    out_lshape = tuple(x.lshape[dim] for dim in range(len(x.lshape)) if not dim in axis)
    x_lsqueezed = x._DNDarray__array.reshape(out_lshape)

    # Calculate split axis according to squeezed shape
    if x.split is not None:
        split = x.split - len(list(dim for dim in axis if dim < x.split))
    else:
        split = x.split

    # Distributed squeeze
    if x.split is not None:
        if x.comm.is_distributed():
            if x.split in axis:
                raise ValueError('Cannot split AND squeeze along same axis. Split is {}, axis is {} for shape {}'.format(
                    x.split, axis, x.shape))
            out_gshape = tuple(x.gshape[dim] for dim in range(len(x.gshape)) if not dim in axis)
            x_gsqueezed = factories.empty(out_gshape, dtype=x.dtype)
            loffset = factories.zeros(1, dtype=types.int64)
            loffset.__setitem__(0, x.comm.chunk(x.gshape, x.split)[0])
            displs = factories.zeros(x.comm.size, dtype=types.int64)
            x.comm.Allgather(loffset, displs)

            # TODO: address uneven distribution of dimensions (Allgatherv). Issue #273, #233
            x.comm.Allgather(x_lsqueezed, x_gsqueezed)  # works with evenly distributed dimensions only
            return dndarray.DNDarray(
                x_gsqueezed,
                out_gshape,
                x_lsqueezed.dtype,
                split=split,
                device=x.device,
                comm=x.comm)

    return dndarray.DNDarray(
        x_lsqueezed,
        out_lshape,
        x.dtype,
        split=split,
        device=x.device,
        comm=x.comm)
