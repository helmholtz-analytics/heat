import torch

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

    local_data = a._DNDarray__array
    unique_axis = None
    if axis is not None:
        # transpose so we can work along the 0 axis
        local_data = local_data.transpose(0, axis)
        unique_axis = 0

    print("local_data", local_data, "unqiue_axis", unique_axis)

    # Calculate the unique on the local values
    if a.lshape[a.split] is 0:
        # Passing an empty vector to torch throws exception
        if axis is None:
            res_shape = [0]
        else:
            res_shape = list(local_data.shape)
            res_shape[0] = 0
        lres = torch.empty(res_shape, dtype=a.dtype.torch_type())
    else:
        lres, inverse_pos = torch.unique(local_data, sorted=sorted, return_inverse=True, dim=unique_axis)

    print("lres", lres)

    # Share and gather the results with the other processes
    uniques = torch.tensor([lres.shape[0]]).to(torch.int32)
    uniques_buf = torch.empty((a.comm.Get_size(), ), dtype=torch.int32)
    a.comm.Allgather(uniques, uniques_buf)

    split = None
    is_split = None

    if axis is None or axis is a.split:
        # Local results can now just be added together
        if axis is None:
            output_dim = [uniques_buf.sum().item()]
            recv_axis = 0
        else:
            output_dim = list(lres.shape)
            output_dim[0] = uniques_buf.sum().item()
            recv_axis = a.split
            split = a.split

        print("output_dim", output_dim)

        counts = tuple(uniques_buf.tolist())
        displs = tuple([0] + uniques_buf.cumsum(0).tolist()[:-1])
        gres_buf = torch.empty(output_dim, dtype=a.dtype.torch_type())
        a.comm.Allgatherv(lres, (gres_buf, counts, displs,), axis=recv_axis, recv_axis=0)
        gres = torch.unique(gres_buf, sorted=sorted, return_inverse=return_inverse, dim=unique_axis)

        print("gres", gres)

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
            indices = torch.tensor(indices, dtype=torch.int32)
        else:
            indices = torch.empty((max_uniques.item(),), dtype=torch.int32)

        a.comm.Bcast(indices, root=max_pos)

        gres = local_data[indices.tolist()]

        is_split = a.split

    if axis is not None:
        # transpose matrix back
        gres = gres.transpose(0, axis)

    print("gres after transpose", gres)

    result = heat.array(gres, dtype=a.dtype, device=a.device, comm=a.comm, split=split, is_split=is_split)

    print("result", result)

    return result

