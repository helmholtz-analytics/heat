import torch
import numpy as np

import heat
from . import dndarray
from . import stride_tricks

__all__ = [
    'expand_dims',
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
    if a.lshape[a.split] is 0:
        # Passing an empty vector to torch throws exception
        if axis is None:
            res_shape = [0]
        else:
            res_shape = list(local_data.shape)
            res_shape[0] = 0
        lres = torch.empty(res_shape, dtype=a.dtype.torch_type())
        inverse_pos = []
    else:
        lres, inverse_pos = torch.unique(local_data, sorted=sorted, return_inverse=True, dim=unique_axis)

    # Share and gather the results with the other processes
    uniques = torch.tensor([lres.shape[0]]).to(torch.int32)
    uniques_buf = torch.empty((a.comm.Get_size(), ), dtype=torch.int32)
    a.comm.Allgather(uniques, uniques_buf)

    split = None
    is_split = None

    if axis is None or axis is a.split:
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
            if axis is None:
                inverse_indices = inverse_pos
            else:
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

        # Run unique a second time for
        gres = torch.unique(gres_buf, sorted=sorted, return_inverse=return_inverse, dim=unique_axis)

        if return_inverse and axis is not None:
            # Use the previously gathered information to generate global inverse_indices
            g_inverse = gres[1]
            gres = gres[0]
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
        if a.comm.Get_rank() is max_pos.item():
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

    result = heat.array(gres, dtype=a.dtype, device=a.device, comm=a.comm, is_split=is_split)

    if split is not None:
        result.resplit(a.split)

    return_value = result
    if return_inverse:
        return_value = [return_value, inverse_indices]

    return return_value

