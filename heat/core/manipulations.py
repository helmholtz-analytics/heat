import torch
import numpy as np

from . import factories
from . import dndarray
from . import factories
from . import stride_tricks
from . import types

__all__ = [
    'expand_dims',
    'squeeze',
    'unique'
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


def unique(a, sorted=False, return_inverse=False, axis=None):
    """
    Finds and returns the unique elements of an array.

    Works most effective if axis != a.split.

    Parameters
    ----------
    a : ht.DNDarray
        Input array where unique elements should be found.
    sorted : bool
        Whether the found elements should be sorted before returning as output.
    return_inverse:
        Whether to also return the indices for where elements in the original input ended up in the returned
        unique list.
    axis : int
        Axis along which unique elements should be found. Default to None, which will return a one dimensional list of
        unique values.

    Returns
    -------
    res : ht.DNDarray
        Output array. The unique elements. Elements are distributed the same way as the input tensor.
    inverse_indices : torch.tensor (optional)
        If return_inverse is True, this tensor will hold the list of inverse indices

    Examples
    --------
    >>> x = ht.array([[3, 2], [1, 3]])
    >>> ht.unique(x, sorted=True)
    array([1, 2, 3])

    >>> ht.unique(x, sorted=True, axis=0)
    array([[1, 3],
           [2, 3]])

    >>> ht.unique(x, sorted=True, axis=1)
    array([[2, 3],
           [3, 1]])
    """
    local_data = a._DNDarray__array
    unique_axis = None
    inverse_indices = None

    if axis is not None:
        # transpose so we can work along the 0 axis
        local_data = local_data.transpose(0, axis)
        unique_axis = 0

    # Calculate the unique on the local values
    if a.lshape[a.split] == 0:
        # Passing an empty vector to torch throws exception
        if axis is None:
            res_shape = [0]
        else:
            res_shape = list(local_data.shape)
            res_shape[0] = 0
        lres = torch.empty(res_shape, dtype=a.dtype.torch_type())
        inverse_pos = torch.tensor([])
    else:
        lres, inverse_pos = torch.unique(local_data, sorted=sorted, return_inverse=True, dim=unique_axis)

    # Share and gather the results with the other processes
    uniques = torch.tensor([lres.shape[0]]).to(torch.int32)
    uniques_buf = torch.empty((a.comm.Get_size(), ), dtype=torch.int32)
    a.comm.Allgather(uniques, uniques_buf)

    split = None
    is_split = None

    if axis is None or axis == a.split:
        # Local results can now just be added together
        if axis is None:
            # One dimensional vectors can't be distributed -> no split
            output_dim = [uniques_buf.sum().item()]
            recv_axis = 0
        else:
            output_dim = list(lres.shape)
            output_dim[0] = uniques_buf.sum().item()
            recv_axis = a.split

            # Result will be split along the same axis as a
            split = a.split

        # Gather all unique vectors
        counts = list(uniques_buf.tolist())
        displs = list([0] + uniques_buf.cumsum(0).tolist()[:-1])
        gres_buf = torch.empty(output_dim, dtype=a.dtype.torch_type())
        a.comm.Allgatherv(lres, (gres_buf, counts, displs,), axis=recv_axis, recv_axis=0)

        if return_inverse:
            # Prepare some information to generated the inverse indices list
            avg_len = a.gshape[a.split] // a.comm.Get_size()
            rem = a.gshape[a.split] % a.comm.Get_size()

            # Share the local reverse indices with other processes
            counts = [avg_len] * a.comm.Get_size()
            add_vec = [1] * rem + [0] * (a.comm.Get_size() - rem)
            inverse_counts = [sum(x) for x in zip(counts, add_vec)]
            inverse_displs = [0] + list(np.cumsum(inverse_counts[:-1]))
            inverse_dim = list(inverse_pos.shape)
            inverse_dim[0] = a.gshape[0]
            inverse_buf = torch.empty(inverse_dim, dtype=inverse_pos.dtype)
            a.comm.Allgatherv(inverse_pos, (inverse_buf, inverse_counts, inverse_displs))

        # Run unique a second time
        gres = torch.unique(gres_buf, sorted=sorted, return_inverse=return_inverse, dim=unique_axis)
        print('gres_buf', gres_buf)
        if return_inverse:
            # Use the previously gathered information to generate global inverse_indices
            g_inverse = gres[1]
            gres = gres[0]
            if axis is None:
                print('lres', lres, 'inverse_pos', inverse_pos, 'gres', gres, 'g_inverse', g_inverse, 'inverse_buf', inverse_buf, 'inverse_displs', inverse_displs)
                inverse_indices = g_inverse.reshape(a.gshape)
            else:
                inverse_indices = torch.zeros_like(inverse_buf)
                steps = displs + [None]

                # Algorithm that creates the correct list for the reverse_indices
                for i in range(len(steps) - 1):
                    begin = steps[i]
                    end = steps[i + 1]
                    for num, x in enumerate(inverse_buf[begin: end]):
                        inverse_indices[begin + num] = g_inverse[begin + x]

    else:
        max_uniques, max_pos = uniques_buf.max(0)

        # find indices of vectors
        if a.comm.Get_rank() == max_pos.item():
            # Get the indices of the vectors we need from each process
            indices = []
            found = []
            pos_list = inverse_pos.tolist()
            for p in pos_list:
                if p not in found:
                    found += [p]
                    indices += [pos_list.index(p)]
                if len(indices) is max_uniques.item():
                    break
            indices = torch.tensor(indices, dtype=a.dtype.torch_type())
        else:
            indices = torch.empty((max_uniques.item(),), dtype=a.dtype.torch_type())

        a.comm.Bcast(indices, root=max_pos)
        gres = local_data[indices.tolist()]

        is_split = a.split
        inverse_indices = indices

    if axis is not None:
        # transpose matrix back
        gres = gres.transpose(0, axis)
    print('gres', gres, 'is_split', is_split)
    result = factories.array(gres, dtype=a.dtype, device=a.device, comm=a.comm, is_split=is_split)

    if split is not None:
        result.resplit(a.split)

    return_value = result
    if return_inverse:
        return_value = [return_value, inverse_indices]

    return return_value
