import numpy as np
import torch
import warnings

from .communication import MPI

from . import constants
from . import dndarray
from . import factories
from . import linalg
from . import stride_tricks
from . import tiling
from . import types
from . import operations


__all__ = [
    "column_stack",
    "concatenate",
    "diag",
    "diagonal",
    "expand_dims",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "hstack",
    "reshape",
    "resplit",
    "rot90",
    "row_stack",
    "shape",
    "sort",
    "squeeze",
    "stack",
    "topk",
    "unique",
    "vstack",
]


def column_stack(arrays):
    """
    Stack 1-D or 2-D ``DNDarray``s as columns into a 2-D ``DNDarray``.
    If the input arrays are 1-D, they will be stacked as columns. If they are 2-D,
    they will be concatenated along the second axis.

    Parameters
    ----------
    arrays : Sequence[DNDarrays,...]

    Raises
    ------
    ValueError
        If arrays have more than 2 dimensions

    Returns
    -------
    DNDarray

    Note
    ----
    All ``DNDarray``s in the sequence must have the same number of rows.
    All ``DNDarray``s must be split along the same axis! Note that distributed
    1-D arrays (``split = 0``) by default will be transposed into distributed
    column arrays with ``split == 1``.

    Examples
    --------
    >>> # 1-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.column_stack((a, b))._DNDarray__array
    tensor([[1, 2],
        [2, 3],
        [3, 4]])
    >>> # 1-D and 2-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([[2, 5], [3, 6], [4, 7]])
    >>> c = ht.array([[7, 10], [8, 11], [9, 12]])
    >>> ht.column_stack((a, b, c))._DNDarray__array
    tensor([[ 1,  2,  5,  7, 10],
            [ 2,  3,  6,  8, 11],
            [ 3,  4,  7,  9, 12]])
    >>> # distributed DNDarrays, 3 processes
    >>> a = ht.arange(10, split=0).reshape((5, 2))
    >>> b = ht.arange(5, 20, split=0).reshape((5, 3))
    >>> c = ht.arange(20, 40, split=0).reshape((5, 4))
    >>> ht_column_stack((a, b, c))._DNDarray__array
    [0/2] tensor([[ 0,  1,  5,  6,  7, 20, 21, 22, 23],
    [0/2]         [ 2,  3,  8,  9, 10, 24, 25, 26, 27]], dtype=torch.int32)
    [1/2] tensor([[ 4,  5, 11, 12, 13, 28, 29, 30, 31],
    [1/2]         [ 6,  7, 14, 15, 16, 32, 33, 34, 35]], dtype=torch.int32)
    [2/2] tensor([[ 8,  9, 17, 18, 19, 36, 37, 38, 39]], dtype=torch.int32)
    >>> # distributed 1-D and 2-D DNDarrays, 3 processes
    >>> a = ht.arange(5, split=0)
    >>> b = ht.arange(5, 20, split=1).reshape((5, 3))
    >>> ht_column_stack((a, b))._DNDarray__array
    [0/2] tensor([[ 0,  5],
    [0/2]         [ 1,  8],
    [0/2]         [ 2, 11],
    [0/2]         [ 3, 14],
    [0/2]         [ 4, 17]], dtype=torch.int32)
    [1/2] tensor([[ 6],
    [1/2]         [ 9],
    [1/2]         [12],
    [1/2]         [15],
    [1/2]         [18]], dtype=torch.int32)
    [2/2] tensor([[ 7],
    [2/2]         [10],
    [2/2]         [13],
    [2/2]         [16],
    [2/2]         [19]], dtype=torch.int32)
    """
    arr_dims = list(array.ndim for array in arrays)
    # sanitation, arrays can be 1-d or 2-d, see sanitation module #468
    over_dims = [i for i, j in enumerate(arr_dims) if j > 2]
    if len(over_dims) > 0:
        raise ValueError("Arrays must be 1-D or 2-D")
    if arr_dims.count(1) == len(arr_dims):
        # all arrays are 1-D, stack
        return stack(arrays, axis=1)
    else:
        if arr_dims.count(1) > 0:
            arr_1d = [i for i, j in enumerate(arr_dims) if j == 1]
            # 1-D arrays must be columns
            arrays = list(arrays)
            for ind in arr_1d:
                arrays[ind] = arrays[ind].reshape((1, arrays[ind].size)).T
        return concatenate(arrays, axis=1)


def concatenate(arrays, axis=0):
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays : Sequence[DNDarrays,...]
        The arrays must have the same shape, except in the dimension corresponding to axis (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    res: DNDarray
        The concatenated DNDarray

    Raises
    ------
    RuntimeError
        If the concatted DNDarray meta information, e.g. split or comm, does not match.
    TypeError
        If the passed parameters are not of correct type (see documentation above).
    ValueError
        If the number of passed arrays is less than two or their shapes do not match.

    Examples
    --------
    >>> x = ht.zeros((3, 5), split=None)
    [0/1] tensor([[0., 0., 0., 0., 0.],
    [0/1]         [0., 0., 0., 0., 0.],
    [0/1]         [0., 0., 0., 0., 0.]])
    [1/1] tensor([[0., 0., 0., 0., 0.],
    [1/1]         [0., 0., 0., 0., 0.],
    [1/1]         [0., 0., 0., 0., 0.]])
    >>> y = ht.ones((3, 6), split=0)
    [0/1] tensor([[1., 1., 1., 1., 1., 1.],
    [0/1]         [1., 1., 1., 1., 1., 1.]])
    [1/1] tensor([[1., 1., 1., 1., 1., 1.]])
    >>> ht.concatenate((x, y), axis=1)
    [0/1] tensor([[0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
    [0/1]         [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]])
    [1/1] tensor([[0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]])

    >>> x = ht.zeros((4, 5), split=1)
    [0/1] tensor([[0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.]])
    [1/1] tensor([[0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.]])
    >>> y = ht.ones((3, 5), split=1)
    [0/1] tensor([[1., 1., 1.],
    [0/1]         [1., 1., 1.],
    [0/1]         [1., 1., 1.]])
    [1/1] tensor([[1., 1.],
    [1/1]         [1., 1.],
    [1/1]         [1., 1.]])
    >>> ht.concatenate((x, y), axis=0)
    [0/1] tensor([[0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [0., 0., 0.],
    [0/1]         [1., 1., 1.],
    [0/1]         [1., 1., 1.],
    [0/1]         [1., 1., 1.]])
    [1/1] tensor([[0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [0., 0.],
    [1/1]         [1., 1.],
    [1/1]         [1., 1.],
    [1/1]         [1., 1.]])
    """
    if not isinstance(arrays, (tuple, list)):
        raise TypeError("arrays must be a list or a tuple")

    # a single array cannot be concatenated
    if len(arrays) < 2:
        raise ValueError("concatenate requires 2 arrays")
    # concatenate multiple arrays
    elif len(arrays) > 2:
        res = concatenate((arrays[0], arrays[1]), axis=axis)
        for a in range(2, len(arrays)):
            res = concatenate((res, arrays[a]), axis=axis)
        return res

    # unpack the arrays
    arr0, arr1 = arrays[0], arrays[1]

    # input sanitation
    if not isinstance(arr0, dndarray.DNDarray) or not isinstance(arr1, dndarray.DNDarray):
        raise TypeError("Both arrays must be DNDarrays")
    if not isinstance(axis, int):
        raise TypeError("axis must be an integer, currently: {}".format(type(axis)))
    axis = stride_tricks.sanitize_axis(arr0.gshape, axis)

    if arr0.ndim != arr1.ndim:
        raise ValueError("DNDarrays must have the same number of dimensions")

    if not all([arr0.gshape[i] == arr1.gshape[i] for i in range(len(arr0.gshape)) if i != axis]):
        raise ValueError(
            "Arrays cannot be concatenated, shapes must be the same in every axis "
            "except the selected axis: {}, {}".format(arr0.gshape, arr1.gshape)
        )

    # different communicators may not be concatenated
    if arr0.comm != arr1.comm:
        raise RuntimeError("Communicators of passed arrays mismatch.")

    # identify common data type
    out_dtype = types.promote_types(arr0.dtype, arr1.dtype)
    if arr0.dtype != out_dtype:
        arr0 = out_dtype(arr0, device=arr0.device)
    if arr1.dtype != out_dtype:
        arr1 = out_dtype(arr1, device=arr1.device)

    s0, s1 = arr0.split, arr1.split
    # no splits, local concat
    if s0 is None and s1 is None:
        return factories.array(
            torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis),
            device=arr0.device,
            comm=arr0.comm,
        )

    # non-matching splits when both arrays are split
    elif s0 != s1 and all([s is not None for s in [s0, s1]]):
        raise RuntimeError(
            "DNDarrays given have differing split axes, arr0 {} arr1 {}".format(s0, s1)
        )

    # unsplit and split array
    elif (s0 is None and s1 != axis) or (s1 is None and s0 != axis):
        out_shape = tuple(
            arr1.gshape[x] if x != axis else arr0.gshape[x] + arr1.gshape[x]
            for x in range(len(arr1.gshape))
        )
        out = factories.empty(
            out_shape, split=s1 if s1 is not None else s0, device=arr1.device, comm=arr0.comm
        )

        _, _, arr0_slice = arr1.comm.chunk(arr0.shape, arr1.split)
        _, _, arr1_slice = arr0.comm.chunk(arr1.shape, arr0.split)
        out._DNDarray__array = torch.cat(
            (arr0._DNDarray__array[arr0_slice], arr1._DNDarray__array[arr1_slice]), dim=axis
        )
        out._DNDarray__comm = arr0.comm

        return out

    elif s0 == s1 or any([s is None for s in [s0, s1]]):
        if s0 != axis and all([s is not None for s in [s0, s1]]):
            # the axis is different than the split axis, this case can be easily implemented
            # torch cat arrays together and return a new array that is_split
            out_shape = tuple(
                arr1.gshape[x] if x != axis else arr0.gshape[x] + arr1.gshape[x]
                for x in range(len(arr1.gshape))
            )
            out = factories.empty(out_shape, split=s0, dtype=out_dtype, device=arr0.device)
            out._DNDarray__array = torch.cat(
                (arr0._DNDarray__array, arr1._DNDarray__array), dim=axis
            )
            out._DNDarray__comm = arr0.comm
            return out

        else:
            arr0 = arr0.copy()
            arr1 = arr1.copy()
            # maps are created for where the data is and the output shape is calculated
            lshape_map = torch.zeros((2, arr0.comm.size, len(arr0.gshape)), dtype=torch.int)
            lshape_map[0, arr0.comm.rank, :] = torch.Tensor(arr0.lshape)
            lshape_map[1, arr0.comm.rank, :] = torch.Tensor(arr1.lshape)
            lshape_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

            arr0_shape, arr1_shape = list(arr0.shape), list(arr1.shape)
            arr0_shape[axis] += arr1_shape[axis]
            out_shape = tuple(arr0_shape)

            # the chunk map is used for determine how much data should be on each process
            chunk_map = torch.zeros((arr0.comm.size, len(arr0.gshape)), dtype=torch.int)
            _, _, chk = arr0.comm.chunk(out_shape, s0 if s0 is not None else s1)
            for i in range(len(out_shape)):
                chunk_map[arr0.comm.rank, i] = chk[i].stop - chk[i].start
            chunk_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, chunk_map, MPI.SUM)

            lshape_map_comm.wait()
            chunk_map_comm.wait()

            if s0 is not None:
                send_slice = [slice(None)] * arr0.ndim
                keep_slice = [slice(None)] * arr0.ndim
                # data is first front-loaded onto the first size/2 processes
                for spr in range(1, arr0.comm.size):
                    if arr0.comm.rank == spr:
                        for pr in range(spr):
                            send_amt = abs((chunk_map[pr, axis] - lshape_map[0, pr, axis]).item())
                            send_amt = (
                                send_amt if send_amt < arr0.lshape[axis] else arr0.lshape[axis]
                            )
                            if send_amt:
                                send_slice[arr0.split] = slice(0, send_amt)
                                keep_slice[arr0.split] = slice(send_amt, arr0.lshape[axis])

                                send = arr0.comm.Isend(
                                    arr0.lloc[send_slice].clone(),
                                    dest=pr,
                                    tag=pr + arr0.comm.size + spr,
                                )
                                arr0._DNDarray__array = arr0.lloc[keep_slice].clone()
                                send.wait()
                    for pr in range(spr):
                        snt = abs((chunk_map[pr, s0] - lshape_map[0, pr, s0]).item())
                        snt = (
                            snt
                            if snt < lshape_map[0, spr, axis]
                            else lshape_map[0, spr, axis].item()
                        )
                        if arr0.comm.rank == pr and snt:
                            shp = list(arr0.gshape)
                            shp[arr0.split] = snt
                            data = torch.zeros(
                                shp, dtype=out_dtype.torch_type(), device=arr0.device.torch_device
                            )

                            arr0.comm.Recv(data, source=spr, tag=pr + arr0.comm.size + spr)
                            arr0._DNDarray__array = torch.cat(
                                (arr0._DNDarray__array, data), dim=arr0.split
                            )
                        lshape_map[0, pr, arr0.split] += snt
                        lshape_map[0, spr, arr0.split] -= snt

            if s1 is not None:
                send_slice = [slice(None)] * arr0.ndim
                keep_slice = [slice(None)] * arr0.ndim
                # push the data backwards (arr1), making the data the proper size for arr1 on the last nodes
                # the data is "compressed" on np/2 processes. data is sent from
                for spr in range(arr0.comm.size - 1, -1, -1):
                    if arr0.comm.rank == spr:
                        for pr in range(arr0.comm.size - 1, spr, -1):
                            # calculate the amount of data to send from the chunk map
                            send_amt = abs((chunk_map[pr, axis] - lshape_map[1, pr, axis]).item())
                            send_amt = (
                                send_amt if send_amt < arr1.lshape[axis] else arr1.lshape[axis]
                            )
                            if send_amt:
                                send_slice[axis] = slice(
                                    arr1.lshape[axis] - send_amt, arr1.lshape[axis]
                                )
                                keep_slice[axis] = slice(0, arr1.lshape[axis] - send_amt)

                                send = arr1.comm.Isend(
                                    arr1.lloc[send_slice].clone(),
                                    dest=pr,
                                    tag=pr + arr1.comm.size + spr,
                                )
                                arr1._DNDarray__array = arr1.lloc[keep_slice].clone()
                                send.wait()
                    for pr in range(arr1.comm.size - 1, spr, -1):
                        snt = abs((chunk_map[pr, axis] - lshape_map[1, pr, axis]).item())
                        snt = (
                            snt
                            if snt < lshape_map[1, spr, axis]
                            else lshape_map[1, spr, axis].item()
                        )

                        if arr1.comm.rank == pr and snt:
                            shp = list(arr1.gshape)
                            shp[axis] = snt
                            data = torch.zeros(
                                shp, dtype=out_dtype.torch_type(), device=arr1.device.torch_device
                            )
                            arr1.comm.Recv(data, source=spr, tag=pr + arr1.comm.size + spr)
                            arr1._DNDarray__array = torch.cat(
                                (data, arr1._DNDarray__array), dim=axis
                            )
                        lshape_map[1, pr, axis] += snt
                        lshape_map[1, spr, axis] -= snt

            if s0 is None:
                arb_slice = [None] * len(arr1.shape)
                for c in range(len(chunk_map)):
                    arb_slice[axis] = c
                    # the chunk map is adjusted by subtracting what data is already in the correct place (the data from
                    # arr1 is already correctly placed) i.e. the chunk map shows how much data is still needed on each
                    # process, the local
                    chunk_map[arb_slice] -= lshape_map[tuple([1] + arb_slice)]

                # after adjusting arr1 need to now select the target data in arr0 on each node with a local slice
                if arr0.comm.rank == 0:
                    lcl_slice = [slice(None)] * arr0.ndim
                    lcl_slice[axis] = slice(chunk_map[0, axis].item())
                    arr0._DNDarray__array = arr0._DNDarray__array[lcl_slice].clone().squeeze()
                ttl = chunk_map[0, axis].item()
                for en in range(1, arr0.comm.size):
                    sz = chunk_map[en, axis]
                    if arr0.comm.rank == en:
                        lcl_slice = [slice(None)] * arr0.ndim
                        lcl_slice[axis] = slice(ttl, sz.item() + ttl, 1)
                        arr0._DNDarray__array = arr0._DNDarray__array[lcl_slice].clone().squeeze()
                    ttl += sz.item()

                if len(arr0.lshape) < len(arr1.lshape):
                    arr0._DNDarray__array.unsqueeze_(axis)

            if s1 is None:
                arb_slice = [None] * len(arr0.shape)
                for c in range(len(chunk_map)):
                    arb_slice[axis] = c
                    chunk_map[arb_slice] -= lshape_map[tuple([0] + arb_slice)]

                # get the desired data in arr1 on each node with a local slice
                if arr1.comm.rank == arr1.comm.size - 1:
                    lcl_slice = [slice(None)] * arr1.ndim
                    lcl_slice[axis] = slice(
                        arr1.lshape[axis] - chunk_map[-1, axis].item(), arr1.lshape[axis], 1
                    )
                    arr1._DNDarray__array = arr1._DNDarray__array[lcl_slice].clone().squeeze()
                ttl = chunk_map[-1, axis].item()
                for en in range(arr1.comm.size - 2, -1, -1):
                    sz = chunk_map[en, axis]
                    if arr1.comm.rank == en:
                        lcl_slice = [slice(None)] * arr1.ndim
                        lcl_slice[axis] = slice(
                            arr1.lshape[axis] - (sz.item() + ttl), arr1.lshape[axis] - ttl, 1
                        )
                        arr1._DNDarray__array = arr1._DNDarray__array[lcl_slice].clone().squeeze()
                    ttl += sz.item()
                if len(arr1.lshape) < len(arr0.lshape):
                    arr1._DNDarray__array.unsqueeze_(axis)

            # now that the data is in the proper shape, need to concatenate them on the nodes where they both exist for
            # the others, just set them equal
            out = factories.empty(
                out_shape,
                split=s0 if s0 is not None else s1,
                dtype=out_dtype,
                device=arr0.device,
                comm=arr0.comm,
            )
            res = torch.cat((arr0._DNDarray__array, arr1._DNDarray__array), dim=axis)
            out._DNDarray__array = res

            return out


def diag(a, offset=0):
    """
    Extract a diagonal or construct a diagonal array.
    See the documentation for `heat.diagonal` for more information about extracting the diagonal.

    Parameters
    ----------
    a: ht.DNDarray
        The array holding data for creating a diagonal array or extracting a diagonal.
        If a is a 1-dimensional array a diagonal 2d-array will be returned.
        If a is a n-dimensional array with n > 1 the diagonal entries will be returned in an n-1 dimensional array.
    offset: int, optional
        The offset from the main diagonal.
        Offset greater than zero means above the main diagonal, smaller than zero is below the main diagonal.

    Returns
    -------
    res: ht.DNDarray
        The extracted diagonal or the constructed diagonal array

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.array([1, 2])
    >>> ht.diag(a)
    tensor([[1, 0],
           [0, 2]])

    >>> ht.diag(a, offset=1)
    tensor([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    >>> ht.equal(ht.diag(ht.diag(a)), a)
    True
    >>> a = ht.array([[1, 2], [3, 4]])
    >>> ht.diag(a)
    tensor([1, 4])
    """
    if len(a.shape) > 1:
        return diagonal(a, offset=offset)
    elif len(a.shape) < 1:
        raise ValueError("input array must be of dimension 1 or greater")
    if not isinstance(offset, int):
        raise ValueError("offset must be an integer, got", type(offset))
    if not isinstance(a, dndarray.DNDarray):
        raise ValueError("a must be a DNDarray, got", type(a))

    # 1-dimensional array, must be extended to a square diagonal matrix
    gshape = (a.shape[0] + abs(offset),) * 2
    off, lshape, _ = a.comm.chunk(gshape, a.split)

    # This ensures that the data is on the correct nodes
    if offset > 0:
        padding = factories.empty(
            (offset,), dtype=a.dtype, split=None, device=a.device, comm=a.comm
        )
        a = concatenate((a, padding))
        indices_x = torch.arange(0, min(lshape[0], max(gshape[0] - off - offset, 0)))
    elif offset < 0:
        padding = factories.empty(
            (abs(offset),), dtype=a.dtype, split=None, device=a.device, comm=a.comm
        )
        a = concatenate((padding, a))
        indices_x = torch.arange(max(0, min(abs(offset) - off, lshape[0])), lshape[0])
    else:
        # Offset = 0 values on main diagonal
        indices_x = torch.arange(0, lshape[0])

    indices_y = indices_x + off + offset
    a.balance_()

    local = torch.zeros(lshape, dtype=a.dtype.torch_type(), device=a.device.torch_device)
    local[indices_x, indices_y] = a._DNDarray__array[indices_x]

    return factories.array(local, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm)


def diagonal(a, offset=0, dim1=0, dim2=1):
    """
    Extract a diagonal of an n-dimensional array with n > 1.
    The returned array will be of dimension n-1.

    Parameters
    ----------
    a: ht.DNDarray
        The array of which the diagonal should be extracted.
    offset: int, optional
        The offset from the main diagonal.
        Offset greater than zero means above the main diagonal, smaller than zero is below the main diagonal.
        Default is 0 which means the main diagonal will be selected.
    dim1: int, optional
        First dimension with respect to which to take the diagonal.
        Default is 0.
    dim2: int, optional
        Second dimension with respect to which to take the diagonal.
        Default is 1.
    Returns
    -------
    res: ht.DNDarray
        An array holding the extracted diagonal.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.array([[1, 2], [3, 4]])
    >>> ht.diagonal(a)
    tensor([1, 4])

    >>> ht.diagonal(a, offset=1)
    tensor([2])

    >>> ht.diagonal(a, offset=-1)
    tensor([3])

    >>> a = ht.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    >>> ht.diagonal(a)
    tensor([[0, 6],
           [1, 7]])

    >>> ht.diagonal(a, dim2=2)
    tensor([[0, 5],
           [2, 7]])
    """
    dim1, dim2 = stride_tricks.sanitize_axis(a.shape, (dim1, dim2))

    if dim1 == dim2:
        raise ValueError("Dim1 and dim2 need to be different")
    if not isinstance(a, dndarray.DNDarray):
        raise ValueError("a must be a DNDarray, got", type(a))
    if not isinstance(offset, int):
        raise ValueError("offset must be an integer, got", type(offset))

    shape = a.gshape
    ax1 = shape[dim1]
    ax2 = shape[dim2]
    # determine the number of diagonal elements that will be retrieved
    length = min(ax1, ax2 - offset) if offset >= 0 else min(ax2, ax1 + offset)
    # Remove dim1 and dim2 from shape and append resulting length
    shape = tuple([x for ind, x in enumerate(shape) if ind not in (dim1, dim2)]) + (length,)
    x, y = min(dim1, dim2), max(dim1, dim2)

    if a.split is None:
        split = None
    elif a.split < x < y:
        split = a.split
    elif x < a.split < y:
        split = a.split - 1
    elif x < y < a.split:
        split = a.split - 2
    else:
        split = len(shape) - 1

    if a.split is None or a.split not in (dim1, dim2):
        result = torch.diagonal(a._DNDarray__array, offset=offset, dim1=dim1, dim2=dim2)
    else:
        vz = 1 if a.split == dim1 else -1
        off, _, _ = a.comm.chunk(a.shape, a.split)
        result = torch.diagonal(a._DNDarray__array, offset=offset + vz * off, dim1=dim1, dim2=dim2)
    return factories.array(result, dtype=a.dtype, is_split=split, device=a.device, comm=a.comm)


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
        raise TypeError("expected ht.DNDarray, but was {}".format(type(a)))

    # sanitize axis, introduce arbitrary dummy dimension to model expansion
    axis = stride_tricks.sanitize_axis(a.shape + (1,), axis)

    return dndarray.DNDarray(
        a._DNDarray__array.unsqueeze(dim=axis),
        a.shape[:axis] + (1,) + a.shape[axis:],
        a.dtype,
        a.split if a.split is None or a.split < axis else a.split + 1,
        a.device,
        a.comm,
    )


def flatten(a):
    """
    Flattens an array into one dimension.
    WARNING: if a.split > 0, then the array must be resplit.

    Parameters
    ----------
    a : DNDarray
        array to collapse
    Returns
    -------
    ret : DNDarray
        flattened copy
    Examples
    --------
    >>> a = ht.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    >>> ht.flatten(a)
    tensor([1,2,3,4,5,6,7,8])
    """
    if a.split is None:
        return factories.array(
            torch.flatten(a._DNDarray__array),
            dtype=a.dtype,
            is_split=None,
            device=a.device,
            comm=a.comm,
        )

    if a.split > 0:
        a = resplit(a, 0)

    a = factories.array(
        torch.flatten(a._DNDarray__array),
        dtype=a.dtype,
        is_split=a.split,
        device=a.device,
        comm=a.comm,
    )
    a.balance_()

    return a


def flip(a, axis=None):
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    a: ht.DNDarray
        Input array to be flipped
    axis: int, tuple
        A list of axes to be flipped

    Returns
    -------
    res: ht.DNDarray
        The flipped array.

    Examples
    --------
    >>> a = ht.array([[0,1],[2,3]])
    >>> ht.flip(a, [0])
    tensor([[2, 3],
        [0, 1]])

    >>> b = ht.array([[0,1,2],[3,4,5]], split=1)
    >>> ht.flip(a, [0,1])
    (1/2) tensor([5,4,3])
    (2/2) tensor([2,1,0])
    """
    # flip all dimensions
    if axis is None:
        axis = tuple(range(a.ndim))

    # torch.flip only accepts tuples
    if isinstance(axis, int):
        axis = [axis]

    flipped = torch.flip(a._DNDarray__array, axis)

    if a.split not in axis:
        return factories.array(
            flipped, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm
        )

    # Need to redistribute tensors on split axis
    # Get local shapes
    old_lshape = a.lshape
    dest_proc = a.comm.size - 1 - a.comm.rank
    new_lshape = a.comm.sendrecv(old_lshape, dest=dest_proc, source=dest_proc)

    # Exchange local tensors
    req = a.comm.Isend(flipped, dest=dest_proc)
    received = torch.empty(new_lshape, dtype=a._DNDarray__array.dtype, device=a.device.torch_device)
    a.comm.Recv(received, source=dest_proc)

    res = factories.array(received, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm)
    res.balance_()  # after swapping, first processes may be empty
    req.Wait()
    return res


def fliplr(a):
    """
        Flip array in the left/right direction. If a.ndim > 2, flip along dimension 1.

        Parameters
        ----------
        a: ht.DNDarray
            Input array to be flipped, must be at least 2-D

        Returns
        -------
        res: ht.DNDarray
            The flipped array.

        Examples
        --------
        >>> a = ht.array([[0,1],[2,3]])
        >>> ht.fliplr(a)
        tensor([[1, 0],
                [3, 2]])

        >>> b = ht.array([[0,1,2],[3,4,5]], split=0)
        >>> ht.fliplr(b)
        (1/2) tensor([[2, 1, 0]])
        (2/2) tensor([[5, 4, 3]])
    """
    return flip(a, 1)


def flipud(a):
    """
        Flip array in the up/down direction.

        Parameters
        ----------
        a: ht.DNDarray
            Input array to be flipped

        Returns
        -------
        res: ht.DNDarray
            The flipped array.

        Examples
        --------
        >>> a = ht.array([[0,1],[2,3]])
        >>> ht.flipud(a)
        tensor([[2, 3],
            [0, 1]])

        >>> b = ht.array([[0,1,2],[3,4,5]], split=0)
        >>> ht.flipud(b)
        (1/2) tensor([3,4,5])
        (2/2) tensor([0,1,2])
    """
    return flip(a, 0)


def hstack(tup):
    """
    Stack arrays in sequence horizontally (column wise).
    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    Parameters
    ----------
    tup : sequence of DNDarrays
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.
    Returns
    -------
    stacked : DNDarray
        The array formed by stacking the given arrays.

    Examples
    --------
    >>> a = ht.array((1,2,3))
    >>> b = ht.array((2,3,4))
    >>> ht.hstack((a,b))._DNDarray__array
    [0/1] tensor([1, 2, 3, 2, 3, 4])
    [1/1] tensor([1, 2, 3, 2, 3, 4])
    >>> a = ht.array((1,2,3), split=0)
    >>> b = ht.array((2,3,4), split=0)
    >>> ht.hstack((a,b))._DNDarray__array
    [0/1] tensor([1, 2, 3])
    [1/1] tensor([2, 3, 4])
    >>> a = ht.array([[1],[2],[3]], split=0)
    >>> b = ht.array([[2],[3],[4]], split=0)
    >>> ht.hstack((a,b))._DNDarray__array
    [0/1] tensor([[1, 2],
    [0/1]         [2, 3]])
    [1/1] tensor([[3, 4]])
    """
    tup = list(tup)
    axis = 1
    all_vec = False
    if len(tup) == 2 and all(len(x.gshape) == 1 for x in tup):
        axis = 0
        all_vec = True
    if not all_vec:
        for cn, arr in enumerate(tup):
            if len(arr.gshape) == 1:
                tup[cn] = arr.expand_dims(1)

    return concatenate(tup, axis=axis)


def reshape(a, shape, axis=None):
    """
    Returns a tensor with the same data and number of elements as a, but with the specified shape.

    Parameters
    ----------
    a : ht.DNDarray
        The input tensor
    shape : tuple, list
        Shape of the new tensor
    axis : int, optional
        The new split axis. None denotes same axis
        Default : None

    Returns
    -------
    reshaped : ht.DNDarray
        The DNDarray with the specified shape

    Raises
    ------
    ValueError
        If the number of elements changes in the new shape.

    Examples
    --------
    >>> a = ht.zeros((3,4))
    >>> ht.reshape(a, (4,3))
    tensor([[0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]])

    >>> a = ht.linspace(0, 14, 8, split=0)
    >>> ht.reshape(a, (2,4))
    (1/2) tensor([[0., 2., 4., 6.]])
    (2/2) tensor([[ 8., 10., 12., 14.]])
    """
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("'a' must be a DNDarray, currently {}".format(type(a)))
    if not isinstance(shape, (list, tuple)):
        raise TypeError("shape must be list, tuple, currently {}".format(type(shape)))
        # check axis parameter
    if axis is None:
        axis = a.split
    stride_tricks.sanitize_axis(shape, axis)
    tdtype, tdevice = a.dtype.torch_type(), a.device.torch_device
    # Check the type of shape and number elements
    shape = stride_tricks.sanitize_shape(shape)
    if torch.prod(torch.tensor(shape, device=tdevice)) != a.size:
        raise ValueError("cannot reshape array of size {} into shape {}".format(a.size, shape))

    def reshape_argsort_counts_displs(
        shape1, lshape1, displs1, axis1, shape2, displs2, axis2, comm
    ):
        """
        Compute the send order, counts, and displacements.
        """
        shape1 = torch.tensor(shape1, dtype=tdtype, device=tdevice)
        lshape1 = torch.tensor(lshape1, dtype=tdtype, device=tdevice)
        shape2 = torch.tensor(shape2, dtype=tdtype, device=tdevice)
        # constants
        width = torch.prod(lshape1[axis1:], dtype=torch.int)
        height = torch.prod(lshape1[:axis1], dtype=torch.int)
        global_len = torch.prod(shape1[axis1:])
        ulen = torch.prod(shape2[axis2 + 1 :])
        gindex = displs1[comm.rank] * torch.prod(shape1[axis1 + 1 :])

        # Get axis position on new split axis
        mask = torch.arange(width, device=tdevice) + gindex
        mask = mask + torch.arange(height, device=tdevice).reshape([height, 1]) * global_len
        mask = (torch.floor_divide(mask, ulen)) % shape2[axis2]
        mask = mask.flatten()

        # Compute return values
        counts = torch.zeros(comm.size, dtype=torch.int, device=tdevice)
        displs = torch.zeros_like(counts)
        argsort = torch.empty_like(mask, dtype=torch.long)
        plz = 0
        for i in range(len(displs2) - 1):
            mat = torch.where((mask >= displs2[i]) & (mask < displs2[i + 1]))[0]
            counts[i] = mat.numel()
            argsort[plz : counts[i] + plz] = mat
            plz += counts[i]
        displs[1:] = torch.cumsum(counts[:-1], dim=0)
        return argsort, counts, displs

    # Forward to Pytorch directly
    if a.split is None:
        return factories.array(
            torch.reshape(a._DNDarray__array, shape), dtype=a.dtype, device=a.device, comm=a.comm
        )

    # Create new flat result tensor
    _, local_shape, _ = a.comm.chunk(shape, axis)
    data = torch.empty(local_shape, dtype=tdtype, device=tdevice).flatten()

    # Calculate the counts and displacements
    _, old_displs, _ = a.comm.counts_displs_shape(a.shape, a.split)
    _, new_displs, _ = a.comm.counts_displs_shape(shape, axis)

    old_displs += (a.shape[a.split],)
    new_displs += (shape[axis],)

    sendsort, sendcounts, senddispls = reshape_argsort_counts_displs(
        a.shape, a.lshape, old_displs, a.split, shape, new_displs, axis, a.comm
    )
    recvsort, recvcounts, recvdispls = reshape_argsort_counts_displs(
        shape, local_shape, new_displs, axis, a.shape, old_displs, a.split, a.comm
    )

    # rearange order
    send = a._DNDarray__array.flatten()[sendsort]
    a.comm.Alltoallv((send, sendcounts, senddispls), (data, recvcounts, recvdispls))

    # original order
    backsort = torch.argsort(recvsort)
    data = data[backsort]

    # Reshape local tensor
    data = data.reshape(local_shape)

    return factories.array(data, dtype=a.dtype, is_split=axis, device=a.device, comm=a.comm)


def rot90(m, k=1, axes=(0, 1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.
    Rotation direction is from the first towards the second axis.

    Parameters
    ----------
    m : DNDarray
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes: (2,) int list or tuple
        The array is rotated in the plane defined by the axes.
        Axes must be different.

    Returns
    -------
    DNDarray

    Notes
    -----
    rot90(m, k=1, axes=(1,0)) is the reverse of rot90(m, k=1, axes=(0,1))
    rot90(m, k=1, axes=(1,0)) is equivalent to rot90(m, k=-1, axes=(0,1))

    May change the split axis on distributed tensors

    Raises
    ------
    TypeError
        If first parameter is not a :class:DNDarray.
    TypeError
        If parameter ``k`` is not castable to integer.
    ValueError
        If ``len(axis)!=2``.
    ValueError
        If the axes are the same.
    ValueError
        If axes are out of range.

    Examples
    --------
    >>> m = ht.array([[1,2],[3,4]], dtype=ht.int)
    >>> m
    tensor([[1, 2],
            [3, 4]], dtype=torch.int32)
    >>> ht.rot90(m)
    tensor([[2, 4],
            [1, 3]], dtype=torch.int32)
    >>> ht.rot90(m, 2)
    tensor([[4, 3],
            [2, 1]], dtype=torch.int32)
    >>> m = ht.arange(8).reshape((2,2,2))
    >>> ht.rot90(m, 1, (1,2))
    tensor([[[1, 3],
             [0, 2]],
            [[5, 7],
             [4, 6]]], dtype=torch.int32)
    """
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    if not isinstance(m, dndarray.DNDarray):
        raise TypeError("expected m to be a ht.DNDarray, but was {}".format(type(m)))

    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    if axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or axes[1] < -m.ndim:
        raise ValueError("Axes={} out of range for array of ndim={}.".format(axes, m.ndim))

    if m.split is None:
        return factories.array(
            torch.rot90(m._DNDarray__array, k, axes), dtype=m.dtype, device=m.device, comm=m.comm
        )

    try:
        k = int(k)
    except (TypeError, ValueError):
        raise TypeError("Unknown type, must be castable to integer")

    k %= 4

    if k == 0:
        return m.copy()
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])

    axes_list = np.arange(0, m.ndim).tolist()
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]], axes_list[axes[0]])

    if k == 1:
        return linalg.transpose(flip(m, axes[1]), axes_list)
    else:
        # k == 3
        return flip(linalg.transpose(m, axes_list), axes[1])


def shape(a):
    """
    Returns the shape of a DNDarray `a`.

    Parameters
    ----------
    a : DNDarray

    Returns
    -------
    tuple of ints
    """
    # sanitize input
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("Expected a to be a DNDarray but was {}".format(type(a)))

    return a.gshape


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
    descending : bool, optional
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
        final_result, final_indices = torch.sort(
            a._DNDarray__array, dim=axis, descending=descending
        )

    else:
        # sorting is affected by split, processes need to communicate results
        # transpose so we can work along the 0 axis
        transposed = a._DNDarray__array.transpose(axis, 0)
        local_sorted, local_indices = torch.sort(transposed, dim=0, descending=descending)

        size = a.comm.Get_size()
        rank = a.comm.Get_rank()
        counts, disp, _ = a.comm.counts_displs_shape(a.gshape, axis=axis)

        actual_indices = local_indices.to(dtype=local_sorted.dtype) + disp[rank]

        length = local_sorted.size()[0]

        # Separate the sorted tensor into size + 1 equal length partitions
        partitions = [x * length // (size + 1) for x in range(1, size + 1)]
        local_pivots = (
            local_sorted[partitions]
            if counts[rank]
            else torch.empty((0,) + local_sorted.size()[1:], dtype=local_sorted.dtype)
        )

        # Only processes with elements should share their pivots
        gather_counts = [int(x > 0) * size for x in counts]
        gather_displs = (0,) + tuple(np.cumsum(gather_counts[:-1]))

        pivot_dim = list(transposed.size())
        pivot_dim[0] = size * sum([1 for x in counts if x > 0])

        # share the local pivots with root process
        pivot_buffer = torch.empty(
            pivot_dim, dtype=a.dtype.torch_type(), device=a.device.torch_device
        )
        a.comm.Gatherv(local_pivots, (pivot_buffer, gather_counts, gather_displs), root=0)

        pivot_dim[0] = size - 1
        global_pivots = torch.empty(
            pivot_dim, dtype=a.dtype.torch_type(), device=a.device.torch_device
        )

        # root process creates new pivots and shares them with other processes
        if rank == 0:
            sorted_pivots, _ = torch.sort(pivot_buffer, descending=descending, dim=0)
            length = sorted_pivots.size()[0]
            global_partitions = [x * length // size for x in range(1, size)]
            global_pivots = sorted_pivots[global_partitions]

        a.comm.Bcast(global_pivots, root=0)

        lt_partitions = torch.empty((size,) + local_sorted.shape, dtype=torch.int64)
        last = torch.zeros_like(local_sorted, dtype=torch.int64)
        comp_op = torch.gt if descending else torch.lt
        # Iterate over all pivots and store which pivot is the first greater than the elements value
        for idx, p in enumerate(global_pivots):
            lt = comp_op(local_sorted, p).int()
            if idx > 0:
                lt_partitions[idx] = lt - last
            else:
                lt_partitions[idx] = lt
            last = lt
        lt_partitions[size - 1] = torch.ones_like(local_sorted, dtype=last.dtype) - last

        # Matrix holding information how many values will be sent where
        local_partitions = torch.sum(lt_partitions, dim=1)

        partition_matrix = torch.empty_like(local_partitions)
        a.comm.Allreduce(local_partitions, partition_matrix, op=MPI.SUM)

        # Matrix that holds information which value will be shipped where
        index_matrix = torch.empty_like(local_sorted, dtype=torch.int64)

        # Matrix holding information which process get how many values from where
        shape = (size,) + transposed.size()[1:]
        send_matrix = torch.zeros(shape, dtype=partition_matrix.dtype)
        recv_matrix = torch.zeros(shape, dtype=partition_matrix.dtype)

        for i, x in enumerate(lt_partitions):
            index_matrix[x > 0] = i
            send_matrix[i] += torch.sum(x, dim=0)

        a.comm.Alltoall(send_matrix, recv_matrix)

        scounts = local_partitions
        rcounts = recv_matrix

        shape = (partition_matrix[rank].max(),) + transposed.size()[1:]
        first_result = torch.empty(shape, dtype=local_sorted.dtype)
        first_indices = torch.empty_like(first_result)

        # Iterate through one layer and send values with alltoallv
        for idx in np.ndindex(local_sorted.shape[1:]):
            idx_slice = [slice(None)] + [slice(ind, ind + 1) for ind in idx]

            send_count = scounts[idx_slice].reshape(-1).tolist()
            send_disp = [0] + list(np.cumsum(send_count[:-1]))
            s_val = local_sorted[idx_slice].clone()
            s_ind = actual_indices[idx_slice].clone().to(dtype=local_sorted.dtype)

            recv_count = rcounts[idx_slice].reshape(-1).tolist()
            recv_disp = [0] + list(np.cumsum(recv_count[:-1]))
            rcv_length = rcounts[idx_slice].sum().item()
            r_val = torch.empty((rcv_length,) + s_val.shape[1:], dtype=local_sorted.dtype)
            r_ind = torch.empty_like(r_val)

            a.comm.Alltoallv((s_val, send_count, send_disp), (r_val, recv_count, recv_disp))
            a.comm.Alltoallv((s_ind, send_count, send_disp), (r_ind, recv_count, recv_disp))
            first_result[idx_slice][:rcv_length] = r_val
            first_indices[idx_slice][:rcv_length] = r_ind

        # The process might not have the correct number of values therefore the tensors need to be rebalanced
        send_vec = torch.zeros(local_sorted.shape[1:] + (size, size), dtype=torch.int64)
        target_cumsum = np.cumsum(counts)
        for idx in np.ndindex(local_sorted.shape[1:]):
            idx_slice = [slice(None)] + [slice(ind, ind + 1) for ind in idx]
            current_counts = partition_matrix[idx_slice].reshape(-1).tolist()
            current_cumsum = list(np.cumsum(current_counts))
            for proc in range(size):
                if current_cumsum[proc] > target_cumsum[proc]:
                    # process has to many values which will be sent to higher ranks
                    first = next(i for i in range(size) if send_vec[idx][:, i].sum() < counts[i])
                    last = next(
                        i
                        for i in range(size + 1)
                        if i == size or current_cumsum[proc] < target_cumsum[i]
                    )
                    sent = 0
                    for i, x in enumerate(counts[first:last]):
                        # Each following process gets as many elements as it needs
                        amount = int(x - send_vec[idx][:, first + i].sum())
                        send_vec[idx][proc][first + i] = amount
                        current_counts[first + i] += amount
                        sent += amount
                    if last < size:
                        # Send all left over values to the highest last process
                        amount = partition_matrix[proc][idx]
                        send_vec[idx][proc][last] = int(amount - sent)
                        current_counts[last] += int(amount - sent)
                elif current_cumsum[proc] < target_cumsum[proc]:
                    # process needs values from higher rank
                    first = (
                        0
                        if proc == 0
                        else next(
                            i for i, x in enumerate(current_cumsum) if target_cumsum[proc - 1] < x
                        )
                    )
                    last = next(i for i, x in enumerate(current_cumsum) if target_cumsum[proc] <= x)
                    for i, x in enumerate(partition_matrix[idx_slice][first:last]):
                        # Taking as many elements as possible from each following process
                        send_vec[idx][first + i][proc] = int(x - send_vec[idx][first + i].sum())
                        current_counts[first + i] = 0
                    # Taking just enough elements from the last element to fill the current processes tensor
                    send_vec[idx][last][proc] = int(target_cumsum[proc] - current_cumsum[last - 1])
                    current_counts[last] -= int(target_cumsum[proc] - current_cumsum[last - 1])
                else:
                    # process doesn't need more values
                    send_vec[idx][proc][proc] = (
                        partition_matrix[proc][idx] - send_vec[idx][proc].sum()
                    )
                current_counts[proc] = counts[proc]
                current_cumsum = list(np.cumsum(current_counts))

        # Iterate through one layer again to create the final balanced local tensors
        second_result = torch.empty_like(local_sorted)
        second_indices = torch.empty_like(second_result)
        for idx in np.ndindex(local_sorted.shape[1:]):
            idx_slice = [slice(None)] + [slice(ind, ind + 1) for ind in idx]

            send_count = send_vec[idx][rank]
            send_disp = [0] + list(np.cumsum(send_count[:-1]))

            recv_count = send_vec[idx][:, rank]
            recv_disp = [0] + list(np.cumsum(recv_count[:-1]))

            end = partition_matrix[rank][idx]
            s_val, indices = first_result[0:end][idx_slice].sort(descending=descending, dim=0)
            s_ind = first_indices[0:end][idx_slice][indices].reshape_as(s_val)

            r_val = torch.empty((counts[rank],) + s_val.shape[1:], dtype=local_sorted.dtype)
            r_ind = torch.empty_like(r_val)

            a.comm.Alltoallv((s_val, send_count, send_disp), (r_val, recv_count, recv_disp))
            a.comm.Alltoallv((s_ind, send_count, send_disp), (r_ind, recv_count, recv_disp))

            second_result[idx_slice] = r_val
            second_indices[idx_slice] = r_ind

        second_result, tmp_indices = second_result.sort(dim=0, descending=descending)
        final_result = second_result.transpose(0, axis)
        final_indices = torch.empty_like(second_indices)
        # Update the indices in case the ordering changed during the last sort
        for idx in np.ndindex(tmp_indices.shape):
            val = tmp_indices[idx]
            final_indices[idx] = second_indices[val.item()][idx[1:]]
        final_indices = final_indices.transpose(0, axis)
    return_indices = factories.array(
        final_indices, dtype=dndarray.types.int32, is_split=a.split, device=a.device, comm=a.comm
    )
    if out is not None:
        out._DNDarray__array = final_result
        return return_indices
    else:
        tensor = factories.array(
            final_result, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm
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
               Split semantics: see note below.

    Examples:
    ---------
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

    Note:
    -----
    Split semantics: a distributed tensor will keep its original split dimension after "squeezing",
    which, depending on the squeeze axis, may result in a lower numerical 'split' value, as in:
    >>> x.shape
    (10, 1, 12, 13)
    >>> x.split
    2
    >>> x.squeeze().shape
    (10, 12, 13)
    >>> x.squeeze().split
    1
    """

    # Sanitize input
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("expected x to be a ht.DNDarray, but was {}".format(type(x)))
    # Sanitize axis
    axis = stride_tricks.sanitize_axis(x.shape, axis)
    if axis is not None:
        if isinstance(axis, int):
            dim_is_one = x.shape[axis] == 1
            axis = (axis,)
        elif isinstance(axis, tuple):
            dim_is_one = bool(torch.tensor(list(x.shape[dim] == 1 for dim in axis)).all())
        if not dim_is_one:
            raise ValueError("Dimension along axis {} is not 1 for shape {}".format(axis, x.shape))

    if axis is None:
        axis = tuple(i for i, dim in enumerate(x.shape) if dim == 1)

    if x.split is not None and x.split in axis:
        # split dimension is about to disappear, set split to None
        x.resplit_(axis=None)

    out_lshape = tuple(x.lshape[dim] for dim in range(x.ndim) if dim not in axis)
    out_gshape = tuple(x.gshape[dim] for dim in range(x.ndim) if dim not in axis)
    x_lsqueezed = x._DNDarray__array.reshape(out_lshape)

    # Calculate new split axis according to squeezed shape
    if x.split is not None:
        split = x.split - len(list(dim for dim in axis if dim < x.split))
    else:
        split = None

    return dndarray.DNDarray(
        x_lsqueezed, out_gshape, x.dtype, split=split, device=x.device, comm=x.comm
    )


def stack(arrays, axis=0, out=None):
    """
    Join a sequence of ``DNDarray``s along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the dimensions of the result.
    For example, if ``axis=0``, the arrays will be stacked along the first dimension; if ``axis=-1``,
    they will be stacked along the last dimension. See Notes below for split semantics.

    Parameters
    ----------
    arrays : Sequence[DNDarrays,...]
        Each DNDarray must have the same shape, must be split along the same axis, and must be balanced.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : DNDarray, optional
        If provided, the destination to place the result. The shape and split axis must be correct, matching
        that of what stack would have returned if no out argument were specified (see Notes below).

    Raises
    ------
    TypeError
        If arrays in sequence are not ``DNDarray``s, or if their ``dtype`` attribute does not match.
    ValueError
        If ``arrays`` contains less than 2 ``DNDarray``s.
    ValueError
        If the ``DNDarray``s are of different shapes, or if they are split along different axes (``split`` attribute).
    RuntimeError
        If the ``DNDarrays`` reside of different devices, or if they are unevenly distributed across ranks (method ``is_balanced()`` returns ``False``)

    Returns
    -------
    DNDarray

    Notes
    -----
    Split semantics: :func:`stack` requires that all arrays in the sequence be split along the same dimension.
    After stacking, the data are still distributed along the original dimension, however a new dimension has been added at `axis`,
    therefore:

    - if :math:`axis <= split`, output will be distributed along :math:`split+1`

    - if :math:`axis > split`, output will be distributed along `split`

    Examples
    --------
    >>> a = ht.arange(20).reshape(4, 5)
    >>> b = ht.arange(20, 40).reshape(4, 5)
    >>> ht.stack((a,b), axis=0)._DNDarray__array
    tensor([[[ 0,  1,  2,  3,  4],
             [ 5,  6,  7,  8,  9],
             [10, 11, 12, 13, 14],
             [15, 16, 17, 18, 19]],

            [[20, 21, 22, 23, 24],
             [25, 26, 27, 28, 29],
             [30, 31, 32, 33, 34],
             [35, 36, 37, 38, 39]]])
    >>> # distributed DNDarrays, 3 processes, stack along last dimension
    >>> a = ht.arange(20, split=0).reshape(4, 5)
    >>> b = ht.arange(20, 40, split=0).reshape(4, 5)
    >>> ht.stack((a,b), axis=-1)._DNDarray__array
    [0/2] tensor([[[ 0, 20],
    [0/2]          [ 1, 21],
    [0/2]          [ 2, 22],
    [0/2]          [ 3, 23],
    [0/2]          [ 4, 24]],
    [0/2]
    [0/2]         [[ 5, 25],
    [0/2]          [ 6, 26],
    [0/2]          [ 7, 27],
    [0/2]          [ 8, 28],
    [0/2]          [ 9, 29]]])
    [1/2] tensor([[[10, 30],
    [1/2]          [11, 31],
    [1/2]          [12, 32],
    [1/2]          [13, 33],
    [1/2]          [14, 34]]])
    [2/2] tensor([[[15, 35],
    [2/2]          [16, 36],
    [2/2]          [17, 37],
    [2/2]          [18, 38],
    [2/2]          [19, 39]]])
    """

    # sanitation
    if not isinstance(arrays, (tuple, list)):
        raise TypeError("arrays must be a list or a tuple")

    if len(arrays) < 2:
        raise ValueError("stack expects a sequence of at least 2 DNDarrays")

    for i, array in enumerate(arrays):
        if not isinstance(array, dndarray.DNDarray):
            raise TypeError(
                "all arrays in sequence must be DNDarrays, array {} was {}".format(i, type(array))
            )

    arrays_metadata = list(
        [array.gshape, array.split, array.device, array.is_balanced()] for array in arrays
    )
    num_arrays = len(arrays)
    # metadata must be identical for all arrays
    if arrays_metadata.count(arrays_metadata[0]) != num_arrays:
        shapes = list(array.gshape for array in arrays)
        if shapes.count(shapes[0]) != num_arrays:
            raise ValueError(
                "All DNDarrays in sequence must have the same shape, got shapes {}".format(shapes)
            )
        splits = list(array.split for array in arrays)
        if splits.count(splits[0]) != num_arrays:
            raise ValueError(
                "All DNDarrays in sequence must have the same split axis, got splits {}"
                "Check out the heat.resplit() documentation.".format(splits)
            )
        devices = list(array.device for array in arrays)
        if devices.count(devices[0]) != num_arrays:
            raise RuntimeError(
                "DNDarrays in sequence must reside on the same device, got devices {}".format(
                    devices
                )
            )
        balance = list(array.is_balanced() for array in arrays)
        if balance.count(balance[0]) != num_arrays:
            raise RuntimeError(
                "DNDarrays distribution must be balanced across ranks, is_balanced() returns {}"
                "You can balance a DNDarray with the balance_() method.".format(balance)
            )
    else:
        array_shape, array_split, array_device = arrays_metadata[0][:3]
        # extract torch tensors
        t_arrays = list(array._DNDarray__array for array in arrays)
        # output dtype
        t_dtypes = list(t_array.dtype for t_array in t_arrays)
        t_array_dtype = t_dtypes[0]
        if t_dtypes.count(t_dtypes[0]) != num_arrays:
            for d in range(1, len(t_dtypes)):
                t_array_dtype = (
                    t_array_dtype
                    if t_array_dtype is t_dtypes[d]
                    else torch.promote_types(t_array_dtype, t_dtypes[d])
                )
            t_arrays = list(t_array.type(t_array_dtype) for t_array in t_arrays)
        array_dtype = types.canonical_heat_type(t_array_dtype)

    # sanitate axis
    axis = stride_tricks.sanitize_axis(array_shape + (num_arrays,), axis)

    # output shape and split
    stacked_shape = array_shape[:axis] + (num_arrays,) + array_shape[axis:]
    if array_split is not None:
        stacked_split = array_split + 1 if axis <= array_split else array_split
    else:
        stacked_split = None

    # sanitate output
    if out is not None:
        if not isinstance(out, dndarray.DNDarray):
            raise TypeError("expected out to be None or ht.DNDarray, but was {}".format(type(out)))
        if out.dtype is not array_dtype:
            raise TypeError("expected out to be {}, but was {}".format(array_dtype, out.dtype))
        if out.gshape != stacked_shape:
            raise ValueError(
                "expected out.shape to be {}, got {}".format(out.gshape, stacked_shape)
            )
        if out.split is not stacked_split:
            raise ValueError("expected out.split to be {}, got {}".format(out.split, stacked_split))
    # end of sanitation

    # stack locally
    t_stacked = torch.stack(t_arrays, dim=axis)

    # return stacked DNDarrays
    if out is not None:
        out._DNDarray__array = t_stacked
        return out

    stacked = dndarray.DNDarray(
        t_stacked,
        gshape=stacked_shape,
        dtype=array_dtype,
        split=stacked_split,
        device=array_device,
        comm=arrays[0].comm,
    )
    return stacked


def unique(a, sorted=False, return_inverse=False, axis=None):
    """
    Finds and returns the unique elements of an array.

    Works most effective if axis != a.split.

    Parameters
    ----------
    a : ht.DNDarray
        Input array where unique elements should be found.
    sorted : bool, optional
        Whether the found elements should be sorted before returning as output.
        Warning: sorted is not working if 'axis != None and axis != a.split'
        Default: False
    return_inverse : bool, optional
        Whether to also return the indices for where elements in the original input ended up in the returned
        unique list.
        Default: False
    axis : int, optional
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
        torch_output = torch.unique(
            a._DNDarray__array, sorted=sorted, return_inverse=return_inverse, dim=axis
        )
        if isinstance(torch_output, tuple):
            heat_output = tuple(
                factories.array(i, dtype=a.dtype, split=None, device=a.device) for i in torch_output
            )
        else:
            heat_output = factories.array(torch_output, dtype=a.dtype, split=None, device=a.device)
        return heat_output

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
        lres, inverse_pos = torch.unique(
            local_data, sorted=sorted, return_inverse=True, dim=unique_axis
        )

    # Share and gather the results with the other processes
    uniques = torch.tensor([lres.shape[0]]).to(torch.int32)
    uniques_buf = torch.empty((a.comm.Get_size(),), dtype=torch.int32)
    a.comm.Allgather(uniques, uniques_buf)

    if axis is None or axis == a.split:
        is_split = None
        split = a.split

        output_dim = list(lres.shape)
        output_dim[0] = uniques_buf.sum().item()

        # Gather all unique vectors
        counts = list(uniques_buf.tolist())
        displs = list([0] + uniques_buf.cumsum(0).tolist()[:-1])
        gres_buf = torch.empty(output_dim, dtype=a.dtype.torch_type())
        a.comm.Allgatherv(lres, (gres_buf, counts, displs), recv_axis=0)

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
            a.comm.Allgatherv(
                inverse_pos, (inverse_buf, inverse_counts, inverse_displs), recv_axis=0
            )
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
                local_elements = [displ * elements_per_layer for displ in inverse_displs][1:] + [
                    float("inf")
                ]

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
                    for num, x in enumerate(inverse_buf[begin:end]):
                        inverse_indices[begin + num] = g_inverse[begin + x]

    else:
        # Tensor is already split and does not need to be redistributed afterward
        split = None
        is_split = a.split
        max_uniques, max_pos = uniques_buf.max(0)
        # find indices of vectors
        if a.comm.Get_rank() == max_pos.item():
            # Get indices of the unique vectors to share with all over processes
            indices = inverse_pos.reshape(-1).unique()
        else:
            indices = torch.empty((max_uniques.item(),), dtype=inverse_pos.dtype)

        a.comm.Bcast(indices, root=max_pos)

        gres = local_data[indices.tolist()]

        inverse_indices = indices
        if sorted:
            raise ValueError(
                "Sorting with axis != split is not supported yet. "
                "See https://github.com/helmholtz-analytics/heat/issues/363"
            )

    if axis is not None:
        # transpose matrix back
        gres = gres.transpose(0, axis)

    split = split if a.split < len(gres.shape) else None
    result = factories.array(
        gres, dtype=a.dtype, device=a.device, comm=a.comm, split=split, is_split=is_split
    )
    if split is not None:
        result.resplit_(a.split)

    return_value = result
    if return_inverse:
        return_value = [return_value, inverse_indices.to(a.device.torch_device)]

    return return_value


def resplit(arr, axis=None):
    """
    Out-of-place redistribution of the content of the tensor. Allows to "unsplit" (i.e. gather) all values from all
    nodes as well as the definition of new axis along which the tensor is split without changes to the values.
    WARNING: this operation might involve a significant communication overhead. Use it sparingly and preferably for
    small tensors.

    Parameters
    ----------
    arr : ht.DNDarray
        The tensor from which to resplit
    axis : int, None
        The new split axis, None denotes gathering, an int will set the new split axis

    Returns
    -------
    resplit: ht.DNDarray
        A new tensor that is a copy of 'arr', but split along 'axis'

    Examples
    --------
    >>> a = ht.zeros((4, 5,), split=0)
    >>> a.lshape
    (0/2) (2, 5)
    (1/2) (2, 5)
    >>> b = resplit(a, None)
    >>> b.split
    None
    >>> b.lshape
    (0/2) (4, 5)
    (1/2) (4, 5)
    >>> a = ht.zeros((4, 5,), split=0)
    >>> a.lshape
    (0/2) (2, 5)
    (1/2) (2, 5)
    >>> b = resplit(a, 1)
    >>> b.split
    1
    >>> b.lshape
    (0/2) (4, 3)
    (1/2) (4, 2)
    """
    # sanitize the axis to check whether it is in range
    axis = stride_tricks.sanitize_axis(arr.shape, axis)

    # early out for unchanged content
    if axis == arr.split:
        return arr.copy()
    if axis is None:
        # new_arr = arr.copy()
        gathered = torch.empty(
            arr.shape, dtype=arr.dtype.torch_type(), device=arr.device.torch_device
        )
        counts, displs, _ = arr.comm.counts_displs_shape(arr.shape, arr.split)
        arr.comm.Allgatherv(arr._DNDarray__array, (gathered, counts, displs), recv_axis=arr.split)
        new_arr = factories.array(gathered, is_split=axis, device=arr.device, dtype=arr.dtype)
        return new_arr
    # tensor needs be split/sliced locally
    if arr.split is None:
        temp = arr._DNDarray__array[arr.comm.chunk(arr.shape, axis)[2]]
        new_arr = factories.array(temp, is_split=axis, device=arr.device, dtype=arr.dtype)
        return new_arr

    arr_tiles = tiling.SplitTiles(arr)
    new_arr = factories.empty(arr.gshape, split=axis, dtype=arr.dtype, device=arr.device)
    new_tiles = tiling.SplitTiles(new_arr)
    rank = arr.comm.rank
    waits = []
    rcv_waits = {}
    for rpr in range(arr.comm.size):
        # need to get where the tiles are on the new one first
        # rpr is the destination
        new_locs = torch.where(new_tiles.tile_locations == rpr)
        new_locs = torch.stack([new_locs[i] for i in range(arr.ndim)], dim=1)

        for i in range(new_locs.shape[0]):
            key = tuple(new_locs[i].tolist())
            spr = arr_tiles.tile_locations[key].item()
            to_send = arr_tiles[key]
            if spr == rank and spr != rpr:
                waits.append(arr.comm.Isend(to_send.clone(), dest=rpr, tag=rank))
            elif spr == rpr and rpr == rank:
                new_tiles[key] = to_send.clone()
            elif rank == rpr:
                buf = torch.zeros_like(new_tiles[key])
                rcv_waits[key] = [arr.comm.Irecv(buf=buf, source=spr, tag=spr), buf]
    for w in waits:
        w.wait()
    for k in rcv_waits.keys():
        rcv_waits[k][0].wait()
        new_tiles[k] = rcv_waits[k][1]

    return new_arr


def row_stack(arrays):
    """
    Stack 1-D or 2-D ``DNDarray``s as rows into a 2-D ``DNDarray``.
    If the input arrays are 1-D, they will be stacked as rows. If they are 2-D,
    they will be concatenated along the first axis.

    Parameters
    ----------
    arrays : Sequence[DNDarrays,...]

    Raises
    ------
    ValueError
        If arrays have more than 2 dimensions

    Returns
    -------
    DNDarray

    Note
    ----
    All ``DNDarray``s in the sequence must have the same number of columns.
    All ``DNDarray``s must be split along the same axis!

    Examples
    --------
    >>> # 1-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.row_stack((a, b))._DNDarray__array
    tensor([[1, 2, 3],
            [2, 3, 4]])
    >>> # 1-D and 2-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([[2, 3, 4], [5, 6, 7]])
    >>> c = ht.array([[7, 8, 9], [10, 11, 12]])
    >>> ht.row_stack((a, b, c))._DNDarray__array
    tensor([[ 1,  2,  3],
            [ 2,  3,  4],
            [ 5,  6,  7],
            [ 7,  8,  9],
            [10, 11, 12]])
    >>> # distributed DNDarrays, 3 processes
    >>> a = ht.arange(10, split=0).reshape((2, 5))
    >>> b = ht.arange(5, 20, split=0).reshape((3, 5))
    >>> c = ht.arange(20, 40, split=0).reshape((4, 5))
    >>> ht.row_stack((a, b, c))._DNDarray__array
    [0/2] tensor([[0, 1, 2, 3, 4],
    [0/2]         [5, 6, 7, 8, 9],
    [0/2]         [5, 6, 7, 8, 9]], dtype=torch.int32)
    [1/2] tensor([[10, 11, 12, 13, 14],
    [1/2]         [15, 16, 17, 18, 19],
    [1/2]         [20, 21, 22, 23, 24]], dtype=torch.int32)
    [2/2] tensor([[25, 26, 27, 28, 29],
    [2/2]         [30, 31, 32, 33, 34],
    [2/2]         [35, 36, 37, 38, 39]], dtype=torch.int32)
    >>> # distributed 1-D and 2-D DNDarrays, 3 processes
    >>> a = ht.arange(5, split=0)
    >>> b = ht.arange(5, 20, split=0).reshape((3, 5))
    >>> ht.row_stack((a, b))._DNDarray__array
    [0/2] tensor([[0, 1, 2, 3, 4],
    [0/2]         [5, 6, 7, 8, 9]])
    [1/2] tensor([[10, 11, 12, 13, 14]])
    [2/2] tensor([[15, 16, 17, 18, 19]])
    """
    arr_dims = list(array.ndim for array in arrays)
    # sanitation, arrays can be 1-d or 2-d, see sanitation module #468
    over_dims = [i for i, j in enumerate(arr_dims) if j > 2]
    if len(over_dims) > 0:
        raise ValueError("Arrays must be 1-D or 2-D")
    if arr_dims.count(1) == len(arr_dims):
        # all arrays are 1-D, stack
        return stack(arrays, axis=0)
    else:
        if arr_dims.count(1) > 0:
            arr_1d = [i for i, j in enumerate(arr_dims) if j == 1]
            # 1-D arrays must be row arrays
            arrays = list(arrays)
            for ind in arr_1d:
                arrays[ind] = arrays[ind].reshape((1, arrays[ind].size))
        return concatenate(arrays, axis=0)


def vstack(tup):
    """
    Stack arrays in sequence vertically (row wise).
    This is equivalent to concatenation along the first axis.
    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    NOTE: the split axis will be switched to 1 in the case that both elements are 1D and split=0
    Parameters
    ----------
    tup : sequence of DNDarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.
    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    Examples
    --------
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.vstack((a,b))._DNDarray__array
    [0/1] tensor([[1, 2, 3],
    [0/1]         [2, 3, 4]])
    [1/1] tensor([[1, 2, 3],
    [1/1]         [2, 3, 4]])
    >>> a = ht.array([1, 2, 3], split=0)
    >>> b = ht.array([2, 3, 4], split=0)
    >>> ht.vstack((a,b))._DNDarray__array
    [0/1] tensor([[1, 2],
    [0/1]         [2, 3]])
    [1/1] tensor([[3],
    [1/1]         [4]])
    >>> a = ht.array([[1], [2], [3]], split=0)
    >>> b = ht.array([[2], [3], [4]], split=0)
    >>> ht.vstack((a,b))._DNDarray__array
    [0/1] tensor([[1],
    [0/1]         [2],
    [0/1]         [3]])
    [1/1] tensor([[2],
    [1/1]         [3],
    [1/1]         [4]])
    """
    tup = list(tup)
    for cn, arr in enumerate(tup):
        if len(arr.gshape) == 1:
            tup[cn] = arr.expand_dims(0).resplit_(arr.split)

    return concatenate(tup, axis=0)


def topk(a, k, dim=None, largest=True, sorted=True, out=None):
    """
    Returns the k highest entries in the array.
    (Not Stable for split arrays)

    Parameters:
    -------
    a: DNDarray
        Array to take items from
    k: int
        Number of items to take
    dim: int
        Dimension along which to take, per default the last dimension
    largest: bool
        Return either the k largest or smallest items
    sorted: bool
        Whether to sort the output (descending if largest=True, else ascending)
    out: tuple of ht.DNDarrays
        (items, indices) to put the result in

    Returns
    -------
    items: ht.DNDarray of shape (k,)
        The selected items
    indices: ht.DNDarray of shape (k,)
        The respective indices

    Examples
    --------
    >>> a = ht.array([1, 2, 3])
    >>> ht.topk(a,2)
    (tensor([3, 2]), tensor([2, 1]))
    >>> a = ht.array([[1,2,3],[1,2,3]])
    >>> ht.topk(a,2,dim=1)
   (tensor([[3, 2],
        [3, 2]]),
    tensor([[2, 1],
        [2, 1]]))
    >>> a = ht.array([[1,2,3],[1,2,3]], split=1)
    >>> ht.topk(a,2,dim=1)
   (tensor([[3],
        [3]]), tensor([[1],
        [1]]))
    (tensor([[2],
        [2]]), tensor([[1],
        [1]]))
    """

    if dim is None:
        dim = len(a.shape) - 1

    if largest:
        neutral_value = -constants.sanitize_infinity(a._DNDarray__array.dtype)
    else:
        neutral_value = constants.sanitize_infinity(a._DNDarray__array.dtype)

    def local_topk(*args, **kwargs):
        shape = a.lshape

        if shape[dim] < k:
            result, indices = torch.topk(args[0], shape[dim], largest=largest, sorted=sorted)
            if dim == a.split:
                # Pad the result with neutral values to fill the buffer
                size = list(result.shape)
                padding_sizes = [
                    k - size[dim] if index == dim else 0
                    for index, item in enumerate(list(result.shape))
                ]
                padding = torch.nn.ConstantPad1d(padding_sizes, neutral_value)
                result = padding(result)
                # Different value for indices padding to prevent type casting issues
                padding = torch.nn.ConstantPad1d(padding_sizes, 0)
                indices = padding(indices)
        else:
            result, indices = torch.topk(args[0], k=k, dim=dim, largest=largest, sorted=sorted)

        # add offset of data chunks if reduction is computed across split axis
        if dim == a.split:
            offset, _, _ = a.comm.chunk(shape, a.split)
            indices = indices.clone()
            indices += torch.tensor(offset * a.comm.rank, dtype=indices.dtype)

        local_shape = list(result.shape)
        local_shape_len = len(shape)

        metadata = torch.tensor([k, dim, largest, sorted, local_shape_len, *local_shape])
        send_buffer = torch.cat(
            (metadata.double(), result.double().flatten(), indices.flatten().double())
        )

        return send_buffer

    gres = operations.__reduce_op(
        a,
        local_topk,
        MPI_TOPK,
        axis=dim,
        neutral=neutral_value,
        dim=dim,
        sorted=sorted,
        largest=largest,
    )

    # Split data again to return a tuple
    local_result = gres._DNDarray__array
    shape_len = int(local_result[4])

    gres, gindices = local_result[5 + shape_len :].chunk(2)
    gres = gres.reshape(*local_result[5 : 5 + shape_len].int())
    gindices = gindices.reshape(*local_result[5 : 5 + shape_len].int())

    # Create output with correct split
    if dim == a.split:
        is_split = None
        split = a.split
    else:
        is_split = a.split
        split = None

    final_array = factories.array(
        gres, dtype=a.dtype, device=a.device, split=split, is_split=is_split
    )
    final_indices = factories.array(
        gindices, dtype=torch.int64, device=a.device, split=split, is_split=is_split
    )

    if out is not None:
        if out[0].shape != final_array.shape or out[1].shape != final_indices.shape:
            raise ValueError(
                "Expecting output buffer tuple of shape ({}, {}), got ({}, {})".format(
                    gres.shape, gindices.shape, out[0].shape, out[1].shape
                )
            )
        out[0]._DNDarray__array.storage().copy_(final_array._DNDarray__array.storage())
        out[1]._DNDarray__array.storage().copy_(final_indices._DNDarray__array.storage())

        out[0]._DNDarray__dtype = a.dtype
        out[1]._DNDarray__dtype = types.int64

    return final_array, final_indices


def mpi_topk(a, b, mpi_type):
    # Parse Buffer
    a_parsed = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    b_parsed = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

    # Collect metadata from Buffer
    k = int(a_parsed[0].item())
    dim = int(a_parsed[1].item())
    largest = bool(a_parsed[2].item())
    sorted = bool(a_parsed[3].item())

    # Offset is the length of the shape on the buffer
    len_shape_a = int(a_parsed[4])
    shape_a = a_parsed[5 : 5 + len_shape_a].int().tolist()
    len_shape_b = int(b_parsed[4])
    shape_b = b_parsed[5 : 5 + len_shape_b].int().tolist()

    # separate the data into values, indices
    a_values, a_indices = a_parsed[len_shape_a + 5 :].chunk(2)
    b_values, b_indices = b_parsed[len_shape_b + 5 :].chunk(2)

    # reconstruct the flatened data by shape
    a_values = a_values.reshape(shape_a)
    a_indices = a_indices.reshape(shape_a)
    b_values = b_values.reshape(shape_b)
    b_indices = b_indices.reshape(shape_b)

    # stack the data to actually run topk on
    values = torch.cat((a_values, b_values), dim=dim)
    indices = torch.cat((a_indices, b_indices), dim=dim)

    result, k_indices = torch.topk(values, k, dim=dim, largest=largest, sorted=sorted)
    indices = torch.gather(indices, dim, k_indices)

    metadata = a_parsed[0 : len_shape_a + 5]
    final_result = torch.cat((metadata, result.double().flatten(), indices.double().flatten()))

    b_parsed.copy_(final_result)


MPI_TOPK = MPI.Op.Create(mpi_topk, commute=True)
