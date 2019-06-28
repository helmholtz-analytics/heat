import operator

import numpy as np
import torch
from mpi4py import MPI

from . import dndarray
from . import factories
from . import stride_tricks
from . import types

__all__ = [
    'expand_dims',
    'sort',
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

    >>> y = ht.expand_dims(x, axis=1)
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
    """
    Sorts the elements of the DNDarray a along the given dimension (by default in ascending order) by their value.

    The sorting is not stable which means that equal elements in the result may have a different ordering than in the
    original array.

    Sorting where `axis == a.split` needs a lot of communication between the processes of MPI.

    Parameters
    ----------
    a : ht.DNDarray
        Input array to be sorted.
    axis : int, optional
        The dimension to sort along.
        Default is the last axis.
    descending : bool, optiona
        If set to true values are sorted in descending order
        Default is false
    out : ht.DNDarray or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    values : ht.DNDarray
        The sorted local results.
    indices
        The indices of the elements in the original data

    Raises
    ------
    ValueError
        If the axis is not in range of the axes.

    Examples
    --------
    >>> x = ht.array([[4, 1], [2, 3]], split=0)
    >>> x.shape
    (1, 2)
    (1, 2)

    >>> y = ht.sort(x, axis=0)
    >>> y
    (array([[2, 1]], array([[1, 0]]))
    (array([[4, 3]], array([[0, 1]]))

    >>> ht.sort(x, descending=True)
    (array([[4, 1]], array([[0, 1]]))
    (array([[3, 2]], array([[1, 0]]))
    """
    # default: using last axis
    if axis is None:
        axis = len(a.shape) - 1

    stride_tricks.sanitize_axis(a.shape, axis)

    if a.split is None or axis != a.split:
        # sorting is not affected by split -> we can just sort along the axis
        final_result, final_indices = torch.sort(a._DNDarray__array, dim=axis, descending=descending)

    else:
        # sorting is affected by split, processes need to communicate results
        # transpose so we can work along the 0 axis
        transposed = a._DNDarray__array.transpose(axis, 0)
        local_sorted, local_indices = torch.sort(transposed, dim=0, descending=descending)
        # print('local_sorted', local_sorted)

        size = a.comm.Get_size()
        rank = a.comm.Get_rank()
        counts, disp, _ = a.comm.counts_displs_shape(a.gshape, axis=axis)

        actual_indices = local_indices.to(dtype=local_sorted.dtype) + disp[rank]

        length = local_sorted.size()[0]

        # Separate the sorted tensor into size + 1 equal length partitions
        partitions = [x * length // (size + 1) for x in range(1, size + 1)]
        local_pivots = local_sorted[partitions] if counts[rank] else torch.empty(
            (0, ) + local_sorted.size()[1:], dtype=local_sorted.dtype)

        # Only processes with elements should share their pivots
        gather_counts = [int(x > 0) * size for x in counts]
        gather_displs = (0, ) + tuple(np.cumsum(gather_counts[:-1]))

        pivot_dim = list(transposed.size())
        pivot_dim[0] = size * sum([1 for x in counts if x > 0])

        # share the local pivots with root process
        pivot_buffer = torch.empty(pivot_dim, dtype=a.dtype.torch_type())
        a.comm.Gatherv(local_pivots, (pivot_buffer, gather_counts, gather_displs), root=0)

        pivot_dim[0] = size - 1
        global_pivots = torch.empty(pivot_dim, dtype=a.dtype.torch_type())

        # root process creates new pivots and shares them with other processes
        if rank is 0:
            sorted_pivots, _ = torch.sort(pivot_buffer, descending=descending, dim=0)
            length = sorted_pivots.size()[0]
            global_partitions = [x * length // size for x in range(1, size)]
            global_pivots = sorted_pivots[global_partitions]

        a.comm.Bcast(global_pivots, root=0)
        # print('global_pivots', global_pivots)

        # Create matrix that holds information which process gets how many values at which position
        zeroes_dim = (size, ) + transposed.size()[1:]
        partition_matrix = torch.zeros(zeroes_dim, dtype=torch.int64)

        # Create matrix that holds information, which value is shipped to which process
        index_matrix = torch.empty_like(local_sorted, dtype=torch.int64)

        # Iterate along the split axis which is now 0 due to transpose
        for i, x in enumerate(local_sorted):
            # Enumerate over all elements with correct index
            for idx, val in np.ndenumerate(x.numpy()):
                op_func = operator.gt if descending else operator.lt
                # Calculate position where element must be sent to
                cur = next(i for i in range(len(global_pivots) + 1)
                           if (i == len(global_pivots) or op_func(val, global_pivots[i][idx])))

                partition_matrix[cur][idx] += 1
                index_matrix[i][idx] = cur

        # Share and sum the local partition_matrix
        a.comm.Allreduce(MPI.IN_PLACE, partition_matrix, op=MPI.SUM)
        # print('Partitions', partition_matrix)

        # Create matrix that holds information where and how many elements will be received from each process
        shape = (size, ) + transposed.size()[1:]
        send_recv_matrix = torch.zeros(shape, dtype=partition_matrix.dtype)

        for idx, val in np.ndenumerate(index_matrix.numpy()):
            pos = (val, ) + idx[1:]
            send_recv_matrix[pos] += 1

        a.comm.Alltoall(MPI.IN_PLACE, send_recv_matrix)

        shape = (partition_matrix[rank].max(), ) + transposed.size()[1:]

        # create matrix whose elements are ranks of processes where the value will come from
        recv_indices = torch.empty(shape, dtype=local_sorted.dtype)
        fill_level = torch.zeros(shape[1:], dtype=torch.int32)

        for i, x in enumerate(send_recv_matrix):
            for idx, val in np.ndenumerate(x.numpy()):
                for k in range(val):
                    recv_indices[fill_level[idx]][idx] = i
                    fill_level[idx] += 1

        # Finally send and receive the values with the correct processes according to the global pivots
        for idx, val in np.ndenumerate(index_matrix.numpy()):
            send_buf = torch.tensor([local_sorted[idx], actual_indices[idx]])
            # Add tag to identify correct value we want to receive later
            tag = int(''.join([str(el) for el in idx[1:]]))
            a.comm.Send(send_buf, dest=val, tag=tag)

        recv_amount = sum(send_recv_matrix)
        fill_level = torch.zeros(shape[1:], dtype=torch.int32)
        first_result = torch.empty(shape, dtype=local_sorted.dtype)
        first_indices = torch.empty_like(first_result)

        for idx, val in np.ndenumerate(recv_amount.numpy()):
            for i in range(val):
                source = recv_indices[fill_level[idx]][idx]
                tag = int(''.join([str(el) for el in idx]))
                recv_buf = torch.empty(2, dtype=local_sorted.dtype)
                a.comm.Recv(recv_buf, source=source, tag=tag)

                first_result[fill_level[idx]][idx] = recv_buf[0]
                first_indices[fill_level[idx]][idx] = recv_buf[1]

                fill_level[idx] += 1

        # print('first_result', first_result)

        # Create a matrix which holds information about the 'unbalancedness' of the local result
        problem_idx = torch.zeros((size, ) + first_result.shape[1:], dtype=partition_matrix.dtype)
        for i, x in enumerate(partition_matrix):
            for idx, val in np.ndenumerate(x.numpy()):
                problem_idx[i][idx] = x[idx] - counts[i]

        print('problem_idx', problem_idx)

        # create final result tensor by iteratively redistributing with the neighbour processes
        second_result = torch.empty(transposed.size(), dtype=a.dtype.torch_type())
        second_indices = torch.empty_like(second_result)
        copy_size = min(a.lshape[axis], partition_matrix[rank].max())
        second_result[0: copy_size] = first_result[0: copy_size]
        second_indices[0: copy_size] = first_indices[0: copy_size]
        for i in range(size):
            for idx, val in np.ndenumerate(problem_idx[i].numpy()):
                while val != 0:
                    if val < 0:
                        # Not enough elements yet -> find next process to receive values from
                        receiver = i
                        sender = next(ind + i + 1 for ind, pr in enumerate(partition_matrix[i + 1:]) if pr[idx] > 0)
                        receiver_idx = (val, ) + idx

                        if rank == sender:
                            end = partition_matrix[sender][idx]
                            enumerate_index = [slice(None)] + [slice(ind, ind + 1) for ind in idx]
                            values = first_result[0: end][enumerate_index]
                            sender_idx = (values.argmax() if descending else values.argmin(), ) + idx

                            send_buf = torch.tensor([first_result[sender_idx], first_indices[sender_idx]])
                            a.comm.Send(send_buf, dest=receiver)
                            # Swap last element along axis at the now freed location
                            last_idx = (a.lshape[axis] + problem_idx[sender][idx] - 1, ) + sender_idx[1:]
                            first_result[sender_idx] = first_result[last_idx]
                            if sender_idx[0] < second_result.shape[0]:
                                second_result[sender_idx] = first_result[last_idx]
                                second_indices[sender_idx] = first_indices[last_idx]

                        if rank == receiver:
                            recv_buf = torch.empty(2, dtype=first_result.dtype)
                            a.comm.Recv(recv_buf, source=sender)
                            second_result[receiver_idx] = recv_buf[0]
                            second_indices[receiver_idx] = recv_buf[1]

                        val += 1
                        problem_idx[receiver][idx] += 1
                        partition_matrix[receiver][idx] += 1
                        problem_idx[sender][idx] -= 1
                        partition_matrix[sender][idx] -= 1

                    if val > 0:
                        # Too many values -> send all to the next process
                        sender = i
                        receiver = i + 1
                        if rank == sender:
                            end = partition_matrix[sender][idx]
                            enumerate_index = [slice(None)] + [slice(ind, ind + 1) for ind in idx]
                            values = first_result[0: end][enumerate_index]
                            print('idx', idx, 'values', values)
                            kth = val if descending else -val
                            send_part = slice(None, kth) if descending else slice(kth, None)
                            rem_part = slice(kth, None) if descending else slice(None, kth)
                            partition_indices = np.array(values).argpartition(axis=0, kth=kth).reshape(-1)
                            sender_indices = partition_indices[send_part]
                            print('kth', kth, 'part', send_part, 'indices', sender_indices, 'values', values[sender_indices, idx], 'rem', rem_part, partition_indices)

                            send_buf = torch.tensor([list(first_result[sender_indices, idx]), list(first_indices[sender_indices, idx])])
                            print('sender', sender, 'receiver', receiver, 'value', send_buf)
                            a.comm.Send(send_buf, dest=receiver)
                            # Swap the not sent values to the correct location
                            print('comp', sender_indices[0])
                            free_spots = [sent_loc for sent_loc in sender_indices if sent_loc < a.lshape[a.split]]
                            unsent = [loc for loc in partition_indices[rem_part] if loc >= a.lshape[a.split]]
                            print('free_spots', free_spots, 'unsent', unsent)
                            second_result[free_spots] = first_result[unsent]
                            second_indices[free_spots] = first_indices[unsent]
                            print('updated', second_result)

                        if rank == receiver:
                            recv_buf = torch.empty((2, val), dtype=first_result.dtype)
                            a.comm.Recv(recv_buf, source=sender)
                            recv_start = partition_matrix[receiver][idx]
                            print('received', recv_buf)
                            if recv_start + val > first_result.shape[0]:
                                # The temporary buffer is to small to store the received values
                                new_shape = list(first_result.shape)
                                new_shape[0] = new_shape[0] + val
                                print('shape', new_shape)
                                tmp = torch.empty(new_shape, dtype=first_result.dtype)
                                tmp_indices = torch.empty_like(tmp)
                                tmp[0: first_result.shape[axis]] = first_result
                                tmp_indices[0: first_result.shape[axis]] = first_indices
                                first_result = tmp
                                first_indices = tmp_indices
                            print('start', recv_start, 'first_end', val, 'second_end', a.lshape[axis])
                            reshape_dim = list(local_sorted.shape)
                            reshape_dim[0] = -1
                            recv_values, recv_indices = recv_buf[0].reshape(reshape_dim), recv_buf[1].reshape(reshape_dim)
                            print('recv_values', recv_values, 'recv_indices', recv_indices)
                            print('first_result', first_result.shape)
                            end_second = min(a.lshape[axis] - recv_start, val)
                            first_result[recv_start: recv_start + val, idx] = recv_values
                            first_indices[recv_start: recv_start + val, idx] = recv_indices
                            second_result[recv_start: recv_start + end_second, idx] = recv_values[: end_second]
                            second_indices[recv_start: recv_start + end_second, idx] = recv_indices[: end_second]

                        problem_idx[receiver][idx] += val
                        partition_matrix[receiver][idx] += val
                        problem_idx[sender][idx] -= val
                        partition_matrix[sender][idx] -= val
                        val -= val
                        print('Sender', sender, 'receiver', receiver, 'idx', idx, 'val', val)

                    print('problem_idx', problem_idx)

        second_result, tmp_indices = second_result.sort(dim=0, descending=descending)
        final_result = second_result.transpose(0, axis)
        final_indices = torch.empty_like(second_indices)
        # Update the indices in case the ordering changed during the last sort
        for idx, val in np.ndenumerate(tmp_indices.numpy()):
            final_indices[idx] = second_indices[val][idx[1:]]
        final_indices = final_indices.to(dtype=torch.int64).transpose(0, axis)

    if out is not None:
        out._DNDarray__array = final_result
        return final_indices
    else:
        tensor = dndarray.DNDarray(
            final_result,
            a.gshape,
            a.dtype,
            a.split,
            a.device,
            a.comm
        )
        return_indices = dndarray.DNDarray(
            final_indices,
            a.gshape,
            dndarray.types.int32,
            a.split,
            a.device,
            a.comm
        )
        return tensor, return_indices


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
    if a.split is None:
        # Trivial case, result can just be forwarded
        return torch.unique(a._DNDarray__array, sorted=sorted, return_inverse=return_inverse, dim=axis)

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
            inv_shape = list(a.gshape)
            inv_shape[a.split] = 0
        else:
            res_shape = list(local_data.shape)
            res_shape[0] = 0
            inv_shape = [0]
        lres = torch.empty(res_shape, dtype=a.dtype.torch_type())
        inverse_pos = torch.empty(inv_shape, dtype=torch.int64)

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
            inverse_dim[a.split] = a.gshape[a.split]
            inverse_buf = torch.empty(inverse_dim, dtype=inverse_pos.dtype)

            # Transpose data and buffer so we can use Allgatherv along axis=0 (axis=1 does not work properly yet)
            inverse_pos = inverse_pos.transpose(0, a.split)
            inverse_buf = inverse_buf.transpose(0, a.split)
            a.comm.Allgatherv(inverse_pos, (inverse_buf, inverse_counts, inverse_displs), axis=0)
            inverse_buf = inverse_buf.transpose(0, a.split)

        # Run unique a second time
        gres = torch.unique(gres_buf, sorted=sorted, return_inverse=return_inverse, dim=unique_axis)
        if return_inverse:
            # Use the previously gathered information to generate global inverse_indices
            g_inverse = gres[1]
            gres = gres[0]
            if axis is None:
                # Calculate how many elements we have in each layer along the split axis
                elements_per_layer = 1
                for num, val in enumerate(a.gshape):
                    if not num == a.split:
                        elements_per_layer *= val

                # Create the displacements for the flattened inverse indices array
                local_elements = [displ * elements_per_layer for displ in inverse_displs][1:] + [float('inf')]

                # Flatten the inverse indices array every element can be updated to represent a global index
                transposed = inverse_buf.transpose(0, a.split)
                transposed_shape = transposed.shape
                flatten_inverse = transposed.flatten()

                # Update the index elements iteratively
                cur_displ = 0
                inverse_indices = [0] * len(flatten_inverse)
                for num in range(len(inverse_indices)):
                    if num >= local_elements[cur_displ]:
                        cur_displ += 1
                    index = flatten_inverse[num] + displs[cur_displ]
                    inverse_indices[num] = g_inverse[index].tolist()

                # Convert the flattened array back to the correct global shape of a
                inverse_indices = torch.tensor(inverse_indices).reshape(transposed_shape)
                inverse_indices = inverse_indices.transpose(0, a.split)

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
    result = factories.array(gres, dtype=a.dtype, device=a.device, comm=a.comm, is_split=is_split)

    if split is not None:
        result.resplit(a.split)

    return_value = result
    if return_inverse:
        return_value = [return_value, inverse_indices]

    return return_value
