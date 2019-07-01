import torch
import numpy as np

from .communication import MPI

from . import dndarray
from . import factories
from . import stride_tricks
from . import types

__all__ = [
    'concatenate',
    'expand_dims',
    'squeeze',
    'unique'
]


def concatenate(arrays, axis=0):
    """
    todo: this
    :param arrays:
    :param axis:
    :return:
    """
    '''
    test if the same split axis
    test if compatible sizes in the '''

    arr0, arr1 = arrays[0], arrays[1]
    if not isinstance(arr0, dndarray.DNDarray) and not isinstance(arr1, dndarray.DNDarray):
        raise TypeError('Both arrays must be DNDarrays')
    # todo: raise an exception if the arrays cannot be concatenated

    s0, s1 = arr0.split, arr1.split
    if s0 is None and s1 is None:
        return factories.array(torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis))
    elif s0 == s1:
        if s0 != axis:  # case 1
            # the axis is different than the split axis, this case can be easily implemented
            # torch cat arrays together and return a new array that is_split
            out_shape = tuple(arr1.gshape[x] if x != axis else arr0.gshape[x] + arr1.gshape[x]
                              for x in range(len(arr1.gshape)))
            out = factories.empty(out_shape, split=s0)
            out._DNDarray__array = torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis)
            return out
            # return factories.array(torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis), is_split=s0)
        else:
            arr0 = arr0.copy()
            arr1 = arr1.copy()
            # in this case the data needs to be pushed up on the first
            '''
            get data from arr0 and compress it to half of the processes 0 -> size/2
            for arr1 do the same but put it on processes size/2 -> size'''
            lshape_map = factories.zeros((arr0.comm.size, len(arr0.gshape)), dtype=int)
            lshape_map[arr0.comm.rank, :] = torch.Tensor(arr0.lshape)
            lshape_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

            arr0_shape, arr1_shape = list(arr0.shape), list(arr1.shape)
            arr0_shape[s0] += arr1_shape[s1]
            out_shape = tuple(arr0_shape)

            chunk_map = factories.zeros((arr0.comm.size, len(arr0.gshape)), dtype=int)
            _, _, chk = arr0.comm.chunk(out_shape, arr0.split)
            for i in range(len(out_shape)):
                chunk_map[arr0.comm.rank, i] = chk[i].stop - chk[i].start
            chunk_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, chunk_map, MPI.SUM)

            lshape_map_comm.wait()
            chunk_map_comm.wait()

            send_slice = [slice(None), ] * arr0.numdims
            keep_slice = [slice(None), ] * arr0.numdims
            # push the data forward, making the data the proper size from arr0 on the first nodes
            for spr in range(1, arr0.comm.size):
                if arr0.comm.rank == spr:
                    for pr in range(spr):
                        send_amt = abs((chunk_map[pr, axis] - lshape_map[pr, axis]).item())
                        send_amt = send_amt if send_amt < arr0.lshape[axis] else arr0.lshape[axis]
                        if send_amt:
                            send_slice[arr0.split] = slice(0, send_amt)
                            keep_slice[arr0.split] = slice(send_amt, arr0.lshape[axis])

                            send = arr0.comm.Isend(arr0._DNDarray__array[send_slice].clone(), dest=pr, tag=pr + arr0.comm.size + spr)
                            arr0._DNDarray__array = arr0._DNDarray__array[keep_slice].clone()
                            send.wait()
                for pr in range(spr):
                    snt = abs((chunk_map[pr, arr0.split] - lshape_map[pr, arr0.split]).item())
                    snt = snt if snt < lshape_map[spr, arr0.split] else lshape_map[spr, arr0.split].item()
                    if arr0.comm.rank == pr and snt:
                        shp = list(arr0.gshape)
                        shp[arr0.split] = snt
                        data = torch.zeros(shp)

                        arr0.comm.Recv(data, source=spr, tag=pr + arr0.comm.size + spr)
                        arr0._DNDarray__array = torch.cat((arr0._DNDarray__array, data), dim=arr0.split)
                    lshape_map[pr, arr0.split] += snt
                    lshape_map[spr, arr0.split] -= snt

            send_slice = [slice(None), ] * arr0.numdims
            keep_slice = [slice(None), ] * arr0.numdims

            lshape_map = factories.zeros((arr1.comm.size, len(arr1.gshape)), dtype=int)
            lshape_map[arr0.comm.rank, :] = torch.Tensor(arr1.lshape)
            arr1.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)
            # push the data backwards (arr1), making the data the proper size for arr1 on the last nodes
            for spr in range(arr0.comm.size - 2, -1, -1):
                if arr0.comm.rank == spr:
                    for pr in range(arr0.comm.size - 1, spr, -1):
                        send_amt = abs((chunk_map[pr, axis] - lshape_map[pr, axis]).item())
                        send_amt = send_amt if send_amt < arr1.lshape[axis] else arr1.lshape[axis]
                        if send_amt:
                            send_slice[axis] = slice(arr1.lshape[axis] - send_amt, arr1.lshape[axis])
                            keep_slice[axis] = slice(0, arr1.lshape[axis] - send_amt)

                            send = arr0.comm.Isend(arr1._DNDarray__array[send_slice].clone(), dest=pr, tag=pr + arr0.comm.size + spr)
                            arr1._DNDarray__array = arr1._DNDarray__array[keep_slice].clone()
                            send.wait()
                for pr in range(arr1.comm.size - 1, spr, -1):
                    snt = abs((chunk_map[pr, axis] - lshape_map[pr, axis]).item())
                    snt = snt if snt < lshape_map[spr, axis] else lshape_map[spr, axis].item()

                    if arr1.comm.rank == pr and snt:
                        shp = list(arr1.gshape)
                        shp[axis] = snt
                        data = torch.zeros(shp)
                        arr1.comm.Recv(data, source=spr, tag=pr + arr0.comm.size + spr)
                        arr1._DNDarray__array = torch.cat((data, arr1._DNDarray__array), dim=arr0.split)
                    lshape_map[pr, axis] += snt
                    lshape_map[spr, axis] -= snt

            # now that the data is in the proper shape, need to concatenate them on the nodes where they both exist for the others, just set them equal
            out = factories.empty((out_shape), split=s0)
            res = torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis)
            out._DNDarray__array = res
            return out
    elif s0 is None:
        # arb_slice = [None] * len(arr1.gshape)
        if s1 != axis:
            _, _, arb_slice = arr1.comm.chunk(arr0.shape, arr1.split)
            out_shape = tuple(arr1.gshape[x] if x != axis else arr0.gshape[x] + arr1.gshape[x]
                             for x in range(len(arr1.gshape)))
            out = factories.empty(out_shape, split=s1)
            out._DNDarray__array = torch.cat((arr0._DNDarray__array[arb_slice], arr1._DNDarray__array), dim=axis)
            return out
            # print(torch.cat((arr0._DNDarray__array[arb_slice], arr1._DNDarray__array), dim=axis))
            # return factories.array(torch.cat((arr0._DNDarray__array[arb_slice], arr1._DNDarray__array), dim=axis), is_split=s1)
        else:
            arr0 = arr0.copy()
            arr1 = arr1.copy()
            lshape_map = torch.zeros((arr1.comm.size, len(arr1.gshape)), dtype=torch.int)
            lshape_map[arr1.comm.rank, :] = torch.Tensor(arr1.lshape)
            lshape_map_comm = arr1.comm.Iallreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

            arr0_shape, arr1_shape = list(arr0.shape), list(arr1.shape)
            arr0_shape[axis] += arr1_shape[axis]
            out_shape = tuple(arr0_shape)

            chunk_map = torch.zeros((arr1.comm.size, len(arr1.gshape)), dtype=torch.int)
            _, _, chk = arr1.comm.chunk(out_shape, arr1.split)
            for i in range(len(out_shape)):
                chunk_map[arr1.comm.rank, i] = chk[i].stop - chk[i].start
            chunk_map_comm = arr1.comm.Iallreduce(MPI.IN_PLACE, chunk_map, MPI.SUM)

            send_slice = [slice(None), ] * arr1.numdims
            keep_slice = [slice(None), ] * arr1.numdims

            lshape_map_comm.wait()
            chunk_map_comm.wait()
            # push the data backwards, making the data the proper size from arr1 on the first nodes
            for spr in range(arr1.comm.size - 1, -1, -1):
                if arr1.comm.rank == spr:
                    for pr in range(arr1.comm.size - 1, spr, -1):
                        send_amt = abs((chunk_map[pr, axis] - lshape_map[pr, axis]).item())
                        send_amt = send_amt if send_amt < arr1.lshape[axis] else arr1.lshape[axis]
                        if send_amt:
                            send_slice[axis] = slice(arr1.lshape[axis] - send_amt, arr1.lshape[axis])
                            keep_slice[axis] = slice(0, arr1.lshape[axis] - send_amt)
                            # print('s', pr, send_amt)
                            arr1.comm.Isend(arr1.lloc[send_slice].clone(), dest=pr, tag=pr + arr1.comm.size + spr)
                            arr1._DNDarray__array = arr1.lloc[keep_slice]
                            # send.wait()
                for pr in range(arr1.comm.size - 1, spr, -1):
                    snt = abs((chunk_map[pr, s1] - lshape_map[pr, s1]).item())
                    snt = snt if snt < lshape_map[spr, s1] else lshape_map[spr, s1].item()
                    # print('\tlp', pr, spr, snt)

                    if arr1.comm.rank == pr and snt:
                        shp = list(arr1.gshape)
                        shp[s1] = snt
                        data = torch.zeros(shp)
                        # print('r', spr, snt)
                        arr1.comm.Recv(data, source=spr, tag=pr + arr1.comm.size + spr)
                        # print(data)
                        arr1._DNDarray__array = torch.cat((data, arr1._DNDarray__array), dim=axis)

                    lshape_map[pr, axis] += snt
                    lshape_map[spr, axis] -= snt

            arr1.comm.Barrier()
            # have split 0 for arr1 and all of arr1
            # print(arr1.gshape, arr1.lshape)

            arb_slice = [None] * len(arr1.shape)
            for c in range(len(chunk_map)):
                arb_slice[axis] = c
                chunk_map[arb_slice] -= lshape_map[arb_slice]

            # chunk_map_slice = [None] * len(arr0.shape)
            # chunk_map_slice[axis] = axis
            if arr0.comm.rank == 0:
                lcl_slice = [slice(None)] * arr0.numdims
                lcl_slice[axis] = slice(chunk_map[0, axis].item())
                # print(lcl_slice)
                arr0._DNDarray__array = arr0._DNDarray__array[lcl_slice].clone().squeeze()
                # print(arr0.lshape)
            ttl = chunk_map[0, axis].item()
            for en in range(1, arr0.comm.size):
                sz = chunk_map[en, axis]
                if arr0.comm.rank == en:
                    lcl_slice = [slice(None)] * arr0.numdims
                    # print(en, sz.item(), ttl, sz.item() + ttl, arr0.lshape[axis] - ttl, arr0.lshape[axis] - (sz.item() + ttl))
                    lcl_slice[axis] = slice(ttl, sz.item() + ttl, 1)
                    # print(arr0._DNDarray__array[lcl_slice].clone().squeeze())
                    arr0._DNDarray__array = arr0._DNDarray__array[lcl_slice].clone().squeeze()
                ttl += sz.item()

            # out = factories.empty(out_shape, split=s1)
            if len(arr0.lshape) < len(arr1.lshape):
                arr0._DNDarray__array.unsqueeze_(axis)
            # print('cat end', arr0._DNDarray__array, arr1._DNDarray__array)
            # print()
            # out._DNDarray__array = torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis)
            out = factories.array(torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis), is_split=s1)
            return out
    elif s1 is None:
        # arb_slice = [None] * len(arr1.gshape)
        if s0 != axis:
            _, _, arb_slice = arr0.comm.chunk(arr1.shape, arr0.split)
            out_shape = tuple(arr0.gshape[x] if x != axis else arr0.gshape[x] + arr1.gshape[x]
                              for x in range(len(arr0.gshape)))
            out = factories.empty(out_shape, split=s0)
            out._DNDarray__array = torch.cat((arr0._DNDarray__array, arr1._DNDarray__array[arb_slice]), dim=axis)
            return out
            # return factories.array(torch.cat((arr0._DNDarray__array, arr1._DNDarray__array[arb_slice]), dim=axis), is_split=s0)
        else:
            arr0 = arr0.copy()
            arr1 = arr1.copy()
            lshape_map = factories.zeros((arr0.comm.size, len(arr0.gshape)), dtype=int)
            lshape_map[arr0.comm.rank, :] = torch.Tensor(arr0.lshape)
            lshape_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

            arr0_shape, arr1_shape = list(arr0.shape), list(arr1.shape)
            arr0_shape[axis] += arr1_shape[axis]
            out_shape = tuple(arr0_shape)

            chunk_map = factories.zeros((arr0.comm.size, len(arr0.gshape)), dtype=int)
            _, _, chk = arr0.comm.chunk(out_shape, arr0.split)
            for i in range(len(out_shape)):
                chunk_map[arr0.comm.rank, i] = chk[i].stop - chk[i].start
            chunk_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, chunk_map, MPI.SUM)

            send_slice = [slice(None), ] * arr0.numdims
            keep_slice = [slice(None), ] * arr0.numdims

            lshape_map_comm.wait()
            chunk_map_comm.wait()
            # push the data forward, making the data the proper size from arr0 on the first nodes
            for spr in range(1, arr0.comm.size):
                if arr0.comm.rank == spr:
                    for pr in range(spr):
                        send_amt = abs((chunk_map[pr, axis] - lshape_map[pr, axis]).item())
                        if send_amt:
                            send_amt = send_amt if send_amt < arr0.lshape[axis] else arr0.lshape[axis]
                            send_slice[arr0.split] = slice(0, send_amt)
                            keep_slice[arr0.split] = slice(send_amt, arr0.lshape[axis])

                            send = arr0.comm.Isend(arr0._DNDarray__array[send_slice].clone(), dest=pr, tag=pr + arr0.comm.size + spr)
                            arr0._DNDarray__array = arr0._DNDarray__array[keep_slice]
                            send.wait()
                for pr in range(spr):
                    snt = abs((chunk_map[pr, s0] - lshape_map[pr, s0]).item())
                    snt = snt if snt < lshape_map[spr, s0] else lshape_map[spr, s0].item()

                    if arr0.comm.rank == pr and snt:
                        shp = list(arr0.gshape)
                        shp[arr0.split] = snt
                        data = torch.zeros(shp)

                        arr0.comm.Recv(data, source=spr, tag=pr + arr0.comm.size + spr)
                        arr0._DNDarray__array = torch.cat((arr0._DNDarray__array, data), dim=arr0.split)
                    lshape_map[pr, arr0.split] += snt
                    lshape_map[spr, arr0.split] -= snt
            # have split 0 for arr0 and all of arr1

            arb_slice = [None] * len(arr0.shape)
            for c in range(len(chunk_map)):
                arb_slice[axis] = c
                chunk_map[arb_slice] -= lshape_map[arb_slice]

            # chunk_map_slice = [None] * len(arr0.shape)
            # chunk_map_slice[axis] = axis
            if arr1.comm.rank == arr1.comm.size - 1:
                lcl_slice = [slice(None)] * arr1.numdims
                lcl_slice[axis] = slice(arr1.lshape[axis] - chunk_map[-1, axis].item(), arr1.lshape[axis], 1)
                # print(lcl_slice)
                arr1._DNDarray__array = arr1._DNDarray__array[lcl_slice].clone().squeeze()
                # print(arr1.lshape)
            ttl = chunk_map[-1, axis].item()
            for en in range(arr1.comm.size - 2, -1, -1):
                sz = chunk_map[en, axis]
                if arr1.comm.rank == en:
                    lcl_slice = [slice(None)] * arr1.numdims
                    # print(en, sz.item(), ttl, sz.item() + ttl, arr1.lshape[axis] - ttl, arr1.lshape[axis] - (sz.item() + ttl))
                    lcl_slice[axis] = slice(arr1.lshape[axis] - (sz.item() + ttl), arr1.lshape[axis] - ttl, 1)
                    # print(lcl_slice)
                    arr1._DNDarray__array = arr1._DNDarray__array[lcl_slice].clone().squeeze()
                ttl += sz.item()

            out = factories.empty((out_shape), split=s0)
            if len(arr1.lshape) < len(arr0.lshape):
                arr1._DNDarray__array.unsqueeze_(axis)

            out._DNDarray__array = torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis)
            return out
    else:
        # this is the case that s0 /= s1 and they are both not None
        # this case requires a raise -> usr needs to resplit one of them
        raise RuntimeError('DNDarrays given have differing numerical splits, arr0 {} arr1 {}'.format(s0, s1))


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

    # if all([g == 1 for g in gres.shape]):
        #     gres = torch.tensor((gres.squeeze()))
        # gres.squeeze_().unsqueeze_(0)
    # print(gres.shape)
    # print(a.dtype, is_split)
    result = factories.array(gres, dtype=a.dtype, device=a.device, comm=a.comm, is_split=is_split)

    if split is not None:
        result.resplit(a.split)

    return_value = result
    if return_inverse:
        return_value = [return_value, inverse_indices]

    return return_value
