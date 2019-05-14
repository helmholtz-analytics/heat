import numpy as np
import torch

from heat.core import factories
from .communication import MPI
from . import operations
from . import types


__all__ = [
    'argmax',
    'argmin',
    'max',
    'min',
    'unique'
]


def argmax(x, axis=None, out=None, **kwargs):
    """
    Returns the indices of the maximum values along an axis.

    Parameters:
    ----------
    x : ht.DNDarray
        Input array.
    axis : int, optional
        By default, the index is into the flattened tensor, otherwise along the specified axis.
    out : ht.DNDarray, optional.
        If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

    Returns:
    -------
    index_tensor : ht.DNDarray of ints
        Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

    Examples:
    --------
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    >>> a = ht.random.randn(3,3)
    >>> a
    tensor([[-0.5631, -0.8923, -0.0583],
            [-0.1955, -0.9656,  0.4224],
            [ 0.2673, -0.4212, -0.5107]])
    >>> ht.argmax(a)
    tensor([5])
    >>> ht.argmax(a, axis=0)
    tensor([[2, 2, 1]])
    >>> ht.argmax(a, axis=1)
    tensor([[2],
            [2],
            [0]])
    """
    def local_argmax(*args, **kwargs):
        axis = kwargs.get('dim', -1)
        shape = x.shape

        # case where the argmin axis is set to None
        # argmin will be the flattened index, computed standalone and the actual minimum value obtain separately
        if len(args) <= 1 and axis < 0:
            indices = torch.argmax(*args, **kwargs).reshape(1)
            maxima = args[0].flatten()[indices]

            # artificially flatten the input tensor shape to correct the offset computation
            axis = x.split
            shape = [np.prod(shape)]
        # usual case where indices and maximum values are both returned. Axis is not equal to None
        else:
            maxima, indices = torch.max(*args, **kwargs)

        # add offset of data chunks if reduction is computed across split axis
        if axis == x.split:
            offset, _, _ = x.comm.chunk(shape, x.split)
            indices += offset

        return torch.cat([maxima.double(), indices.double()])

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError('axis must be None or int, but was {}'.format(type(axis)))

    # perform the global reduction
    reduced_result = operations.__reduce_op(x, local_argmax, MPI_ARGMAX, axis=axis, out=out, **kwargs)

    # correct the tensor
    reduced_result._DNDarray__array = reduced_result._DNDarray__array.chunk(2)[-1].type(torch.int64)
    reduced_result._DNDarray__dtype = types.int64

    # set out parameter correctly, i.e. set the storage correctly
    if out is not None:
        out._DNDarray__array.storage().copy_(reduced_result._DNDarray__array.storage())

    return reduced_result


def argmin(x, axis=None, out=None, **kwargs):
    """
    Returns the indices of the minimum values along an axis.

    Parameters:
    ----------
    x : ht.DNDarray
        Input array.
    axis : int, optional
        By default, the index is into the flattened tensor, otherwise along the specified axis.
    # out : ht.DNDarray, optional. Issue #100
        If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

    Returns:
    -------
    index_tensor : ht.DNDarray of ints
        Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

    Examples:
    --------
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    >>> a = ht.random.randn(3,3)
    >>> a
    tensor([[-0.5631, -0.8923, -0.0583],
    [-0.1955, -0.9656,  0.4224],
    [ 0.2673, -0.4212, -0.5107]])
    >>> ht.argmin(a)
    tensor([4])
    >>> ht.argmin(a, axis=0)
    tensor([[0, 1, 2]])
    >>> ht.argmin(a, axis=1)
    tensor([[1],
            [1],
            [2]])
    """
    def local_argmin(*args, **kwargs):
        axis = kwargs.get('dim', -1)
        shape = x.shape

        # case where the argmin axis is set to None
        # argmin will be the flattened index, computed standalone and the actual minimum value obtain separately
        if len(args) <= 1 and axis < 0:
            indices = torch.argmin(*args, **kwargs).reshape(1)
            minimums = args[0].flatten()[indices]

            # artificially flatten the input tensor shape to correct the offset computation
            axis = x.split
            shape = [np.prod(shape)]
        # usual case where indices and minimum values are both returned. Axis is not equal to None
        else:
            minimums, indices = torch.min(*args, **kwargs)

        # add offset of data chunks if reduction is computed across split axis
        if axis == x.split:
            offset, _, _ = x.comm.chunk(shape, x.split)
            indices += offset

        return torch.cat([minimums.double(), indices.double()])

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError('axis must be None or int, but was {}'.format(type(axis)))

    # perform the global reduction
    reduced_result = operations.__reduce_op(x, local_argmin, MPI_ARGMIN, axis=axis, out=out, **kwargs)

    # correct the tensor
    reduced_result._DNDarray__array = reduced_result._DNDarray__array.chunk(2)[-1].type(torch.int64)
    reduced_result._DNDarray__dtype = types.int64

    # set out parameter correctly, i.e. set the storage correctly
    if out is not None:
        out._DNDarray__array.storage().copy_(reduced_result._DNDarray__array.storage())

    return reduced_result


def max(x, axis=None, out=None, keepdim=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the maximum along a given axis.

    Parameters
    ----------
    a : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ht.DNDarray, optional
        Tuple of two output tensors (max, max_indices). Must be of the same shape and buffer length as the expected
        output. The minimum value of an output element. Must be present to allow computation on empty slice.

    Examples
    --------
    >>> a = ht.float32([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    >>> ht.max(a)
    tensor([12.])
    >>> ht.min(a, axis=0)
    tensor([[10., 11., 12.]])
    >>> ht.min(a, axis=1)
    tensor([[ 3.],
            [ 6.],
            [ 9.],
            [12.]])
    """
    def local_max(*args, **kwargs):
        result = torch.max(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    return operations.__reduce_op(x, local_max, MPI.MAX, axis=axis, out=out, keepdim=keepdim)


def min(x, axis=None, out=None, keepdim=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    a : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ht.DNDarray, optional
        Tuple of two output tensors (min, min_indices). Must be of the same shape and buffer length as the expected
        output.The maximum value of an output element. Must be present to allow computation on empty slice.

    Examples
    --------
    >>> a = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
    >>> ht.min(a)
    tensor([1.])
    >>> ht.min(a, axis=0)
    tensor([[1., 2., 3.]])
    >>> ht.min(a, axis=1)
    tensor([[ 1.],
        [ 4.],
        [ 7.],
        [10.]])
    """
    def local_min(*args, **kwargs):
        result = torch.min(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    return operations.__reduce_op(x, local_min, MPI.MIN, axis=axis, out=out, keepdim=keepdim)


def mpi_argmax(a, b, _):
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0],), dim=1)
    indices = torch.stack((lhs.chunk(2)[1], rhs.chunk(2)[1],), dim=1)

    # determine the minimum value and select the indices accordingly
    max, max_indices = torch.max(values, dim=1)
    result = torch.cat((max, indices[torch.arange(values.shape[0]), max_indices],))

    rhs.copy_(result)


MPI_ARGMAX = MPI.Op.Create(mpi_argmax, commute=True)


def mpi_argmin(a, b, _):
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0],), dim=1)
    indices = torch.stack((lhs.chunk(2)[1], rhs.chunk(2)[1],), dim=1)

    # determine the minimum value and select the indices accordingly
    min, min_indices = torch.min(values, dim=1)
    result = torch.cat(
        (min, indices[torch.arange(values.shape[0]), min_indices],))

    rhs.copy_(result)


MPI_ARGMIN = MPI.Op.Create(mpi_argmin, commute=True)


def unique(a, sorted=False, return_inverse=False, axis=None):
    # Calculate the unique on the local values
    if a.lshape[a.split] is 0:
        # Passing empty vector to torch results in an exception
        if axis is None:
            res_shape = [0]
        else:
            res_shape = list(a.lshape)
            res_shape[axis] = 0
        lres = torch.empty(res_shape, dtype=a.dtype.torch_type())
    else:
        lres, inverse_pos = torch.unique(a._DNDarray__array, sorted=sorted, return_inverse=True, dim=axis)

    # Share and gather the results with the other processes
    uniques = torch.tensor([lres.shape[0 if axis is None else axis]]).to(torch.int32)
    uniques_buf = torch.empty((a.comm.Get_size(), ), dtype=torch.int32)
    a.comm.Allgather(uniques, uniques_buf)

    if axis is None or axis is a.split:
        # Local results can now just be added together
        if axis is None:
            output_dim = [uniques_buf.sum().item()]
            axis = 0
        else:
            output_dim = [None, None]
            output_dim[(axis + 1) % 2] = lres.shape[(axis + 1) % 2]
            output_dim[axis] = uniques_buf.sum().item()
            axis = a.split

        counts = tuple(uniques_buf.tolist())
        displs = tuple([0] + uniques_buf.cumsum(0).tolist()[:-1])
        gres_buf = torch.empty(output_dim, dtype=a.dtype.torch_type())

        a.comm.Allgatherv(lres, (gres_buf, counts, displs,), axis=axis, recv_axis=axis)

        return torch.unique(gres_buf, sorted=sorted, return_inverse=return_inverse, dim=axis)

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

    # Creates a list of slices to select the correct tensors of the matrix
    local_slice = [slice(None)] * axis + [indices.tolist()]
    lres = a._DNDarray__array[local_slice]

    output_dim = [None, None]
    output_dim[a.split] = a.gshape[a.split]
    output_dim[(a.split + 1) % 2] = max_uniques.item()
    counts_buf = torch.empty(a.comm.Get_size(), dtype=torch.int32)
    local_count = torch.tensor([a.lshape[a.split]]).to(torch.int32)

    a.comm.Allgather(local_count, counts_buf)

    result_buf = torch.empty(output_dim, dtype=a.dtype.torch_type())

    counts = tuple(counts_buf.tolist())
    displs = tuple([0] + counts_buf.cumsum(dim=0).tolist())[:-1]
    print("lres", lres)

    a.comm.Allgatherv(lres, (result_buf, counts, displs), axis=a.split, recv_axis=a.split)

    return result_buf
