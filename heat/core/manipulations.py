"""
Manipulation operations for (potentially distributed) `DNDarray`s.
"""
from __future__ import annotations

import numpy as np
import torch
import warnings

from typing import Iterable, Type, List, Callable, Union, Tuple, Sequence, Optional

from .communication import MPI
from .dndarray import DNDarray

from . import arithmetics
from . import factories
from . import indexing
from . import linalg
from . import sanitation
from . import stride_tricks
from . import tiling
from . import types
from . import _operations

__all__ = [
    "balance",
    "broadcast_arrays",
    "broadcast_to",
    "column_stack",
    "concatenate",
    "diag",
    "diagonal",
    "dsplit",
    "expand_dims",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "moveaxis",
    "pad",
    "ravel",
    "redistribute",
    "repeat",
    "reshape",
    "resplit",
    "roll",
    "rot90",
    "row_stack",
    "shape",
    "sort",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "tile",
    "topk",
    "unique",
    "vsplit",
    "vstack",
]


def balance(array: DNDarray, copy=False) -> DNDarray:
    """
    Out of place balance function. More information on the meaning of balance can be found in
    :func:`DNDarray.balance_() <heat.core.dndarray.DNDarray.balance_()>`.

    Parameters
    ----------
    array : DNDarray
        the DNDarray to be balanced
    copy : bool, optional
        if the DNDarray should be copied before being balanced. If false (default) this will balance
        the original array and return that array. Otherwise (true), a balanced copy of the array
        will be returned.
        Default: False
    """
    cpy = array.copy() if copy else array
    cpy.balance_()
    return cpy


DNDarray.balance = lambda self, copy=False: balance(self, copy)
DNDarray.balance.__doc__ = balance.__doc__


def broadcast_arrays(*arrays: DNDarray) -> List[DNDarray]:
    """
    Broadcasts one or more arrays against one another. Returns the broadcasted arrays, distributed along the split dimension of the first array in the list. If the first array is not distributed, the output will not be distributed.

    Parameters
    ----------
    arrays : DNDarray
        An arbitrary number of to-be broadcasted ``DNDarray``s.

    Notes
    -----
    Broadcasted arrays are a view of the original arrays if possible, otherwise a copy is made.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.ones((100, 10), split=0)
    >>> b = ht.ones((10,), split=None)
    >>> c = ht.ones((1, 10), split=1)
    >>> d, e, f = ht.broadcast_arrays(a, b, c)
    >>> d.shape
    (100, 10)
    >>> e.shape
    (100, 10)
    >>> f.shape
    (100, 10)
    >>> d.split
    0
    >>> e.split
    0
    >>> f.split
    0
    """
    if len(arrays) <= 1:
        return arrays

    try:
        arrays = sanitation.sanitize_distribution(*arrays, target=arrays[0])
        output_split, output_comm, output_balanced = (
            arrays[0].split,
            arrays[0].comm,
            arrays[0].balanced,
        )
    except NotImplementedError as e:
        raise ValueError(e)

    gshapes = []
    t_arrays = []
    for array in arrays:
        # extract global shapes
        gshapes.append(array.gshape)
        # extract local torch tensors
        t_arrays.append(array.larray)
    t_arrays = tuple(t_arrays)

    # broadcast the global shapes
    try:
        output_shape = tuple(torch.broadcast_shapes(*gshapes))
    except RuntimeError:
        raise ValueError(
            f"Shape mismatch: objects cannot be broadcast to a single shape. Original shapes: {gshapes}"
        )

    # broadcast the local torch tensors: this is a view of the original data
    broadcasted = torch.broadcast_tensors(*t_arrays)

    out = []
    for i in range(len(broadcasted)):
        out.append(
            DNDarray(
                broadcasted[i],
                gshape=output_shape,
                dtype=arrays[i].dtype,
                split=output_split,
                device=arrays[i].device,
                comm=output_comm,
                balanced=output_balanced,
            )
        )

    return out


def broadcast_to(x: DNDarray, shape: Tuple[int, ...]) -> DNDarray:
    """
    Broadcasts an array to a specified shape. Returns a view of ``x`` if ``x`` is not distributed, otherwise it returns a broadcasted, distributed, load-balanced copy of ``x``.

    Parameters
    ----------
    x : DNDarray
        `DNDarray` to broadcast.
    shape : Tuple[int, ...]
        Array shape. Must be compatible with ``x``.

    Raises
    ------
    ValueError
        If the array is not compatible with the new shape according to PyTorch's broadcasting rules.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.arange(100, split=0)
    >>> b = ht.broadcast_to(a, (10,100))
    >>> b.shape
    (10, 100)
    >>> b.split
    1
    >>> c = ht.broadcast_to(a, (100, 10))
    ValueError: Shape mismatch: object cannot be broadcast to the given shape. Original shape: (100,), target shape: (100, 10)
    """
    sanitation.sanitize_in(x)

    # figure out the output split axis via dndarray.__torch_proxy__ and named tensors functionality
    torch_proxy = x.__torch_proxy__()
    split_tags = [None] * x.ndim
    if x.split is not None:
        split_tags[x.split] = "split"
        torch_proxy = torch.tensor(torch_proxy, names=split_tags)
        try:
            torch_proxy = torch_proxy.broadcast_to(shape)
        except RuntimeError:
            raise ValueError(
                f"Shape mismatch: object cannot be broadcast to the given shape. Original shape: {x.shape}, target shape: {shape}"
            )
        output_split = torch_proxy.names.index("split")
    else:
        try:
            torch_proxy = torch_proxy.broadcast_to(shape)
        except RuntimeError:
            raise ValueError(
                f"Shape mismatch: object cannot be broadcast to the given shape. Original shape: {x.shape}, target shape: {shape}"
            )
        output_split = None

    if not x.is_distributed():
        # return a view of the input data
        broadcasted = DNDarray(
            x.larray.broadcast_to(shape),
            gshape=shape,
            dtype=x.dtype,
            split=output_split,
            device=x.device,
            comm=x.comm,
            balanced=True,
        )
    else:
        # input is distributed, return a broadcasted copy of input
        # exploit binary operations broadcasting
        broadcasted = factories.zeros(
            shape, dtype=x.dtype, split=output_split, device=x.device, comm=x.comm
        )
        broadcasted += x
        del x

    return broadcasted


def column_stack(arrays: Sequence[DNDarray, ...]) -> DNDarray:
    """
    Stack 1-D or 2-D `DNDarray`s as columns into a 2-D `DNDarray`.
    If the input arrays are 1-D, they will be stacked as columns. If they are 2-D,
    they will be concatenated along the second axis.

    Parameters
    ----------
    arrays : Sequence[DNDarray, ...]
        Sequence of `DNDarray`s.

    Raises
    ------
    ValueError
        If arrays have more than 2 dimensions

    Notes
    -----
    All `DNDarray`s in the sequence must have the same number of rows.
    All `DNDarray`s must be split along the same axis! Note that distributed
    1-D arrays (`split = 0`) by default will be transposed into distributed
    column arrays with `split == 1`.

    See Also
    --------
    :func:`concatenate`
    :func:`hstack`
    :func:`row_stack`
    :func:`stack`
    :func:`vstack`

    Examples
    --------
    >>> # 1-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.column_stack((a, b)).larray
    tensor([[1, 2],
            [2, 3],
            [3, 4]])
    >>> # 1-D and 2-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([[2, 5], [3, 6], [4, 7]])
    >>> c = ht.array([[7, 10], [8, 11], [9, 12]])
    >>> ht.column_stack((a, b, c)).larray
    tensor([[ 1,  2,  5,  7, 10],
            [ 2,  3,  6,  8, 11],
            [ 3,  4,  7,  9, 12]])
    >>> # distributed DNDarrays, 3 processes
    >>> a = ht.arange(10, split=0).reshape((5, 2))
    >>> b = ht.arange(5, 20, split=0).reshape((5, 3))
    >>> c = ht.arange(20, 40, split=0).reshape((5, 4))
    >>> ht_column_stack((a, b, c)).larray
    [0/2] tensor([[ 0,  1,  5,  6,  7, 20, 21, 22, 23],
    [0/2]         [ 2,  3,  8,  9, 10, 24, 25, 26, 27]], dtype=torch.int32)
    [1/2] tensor([[ 4,  5, 11, 12, 13, 28, 29, 30, 31],
    [1/2]         [ 6,  7, 14, 15, 16, 32, 33, 34, 35]], dtype=torch.int32)
    [2/2] tensor([[ 8,  9, 17, 18, 19, 36, 37, 38, 39]], dtype=torch.int32)
    >>> # distributed 1-D and 2-D DNDarrays, 3 processes
    >>> a = ht.arange(5, split=0)
    >>> b = ht.arange(5, 20, split=1).reshape((5, 3))
    >>> ht_column_stack((a, b)).larray
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


def concatenate(arrays: Sequence[DNDarray, ...], axis: int = 0) -> DNDarray:
    """
    Join 2 or more `DNDarrays` along an existing axis.

    Parameters
    ----------
    arrays: Sequence[DNDarray, ...]
        The arrays must have the same shape, except in the dimension corresponding to axis.
    axis: int, optional
        The axis along which the arrays will be joined (default is 0).

    Raises
    ------
    RuntimeError
        If the concatenated :class:`~heat.core.dndarray.DNDarray` meta information, e.g. `split` or `comm`, does not match.
    TypeError
        If the passed parameters are not of correct type.
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
    # input sanitation
    arrays = sanitation.sanitize_sequence(arrays)
    for arr in arrays:
        sanitation.sanitize_in(arr)

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
    arr0, arr1 = arrays

    if not isinstance(axis, int):
        raise TypeError(f"axis must be an integer, currently: {type(axis)}")
    axis = stride_tricks.sanitize_axis(arr0.gshape, axis)

    if arr0.ndim != arr1.ndim:
        raise ValueError("DNDarrays must have the same number of dimensions")

    if any(arr0.gshape[i] != arr1.gshape[i] for i in range(len(arr0.gshape)) if i != axis):
        raise ValueError(
            f"Arrays cannot be concatenated, shapes must be the same in every axis except the selected axis: {arr0.gshape}, {arr1.gshape}"
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
            torch.cat((arr0.larray, arr1.larray), dim=axis), device=arr0.device, comm=arr0.comm
        )

    # non-matching splits when both arrays are split
    elif s0 != s1 and all([s is not None for s in [s0, s1]]):
        raise RuntimeError(f"DNDarrays given have differing split axes, arr0 {s0} arr1 {s1}")

    elif (s0 is None and s1 != axis) or (s1 is None and s0 != axis):
        _, _, arr0_slice = arr1.comm.chunk(arr0.shape, arr1.split)
        _, _, arr1_slice = arr0.comm.chunk(arr1.shape, arr0.split)
        out = factories.array(
            torch.cat((arr0.larray[arr0_slice], arr1.larray[arr1_slice]), dim=axis),
            dtype=out_dtype,
            is_split=s1 if s1 is not None else s0,
            device=arr1.device,
            comm=arr0.comm,
        )

        return out

    elif s0 == s1 or any(s is None for s in [s0, s1]):
        if s0 != axis and all(s is not None for s in [s0, s1]):
            # the axis is different than the split axis, this case can be easily implemented
            # torch cat arrays together and return a new array that is_split

            out = factories.array(
                torch.cat((arr0.larray, arr1.larray), dim=axis),
                dtype=out_dtype,
                is_split=s0,
                device=arr0.device,
                comm=arr0.comm,
            )
            return out

        else:
            t_arr0 = arr0.larray
            t_arr1 = arr1.larray
            # maps are created for where the data is and the output shape is calculated
            lshape_map = torch.zeros((2, arr0.comm.size, len(arr0.gshape)), dtype=torch.int)
            lshape_map[0, arr0.comm.rank, :] = torch.Tensor(arr0.lshape)
            lshape_map[1, arr0.comm.rank, :] = torch.Tensor(arr1.lshape)
            lshape_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

            arr0_shape, arr1_shape = list(arr0.shape), list(arr1.shape)
            arr0_shape[axis] += arr1_shape[axis]
            out_shape = tuple(arr0_shape)

            # the chunk map is used to determine how much data should be on each process
            chunk_map = torch.zeros((arr0.comm.size, len(arr0.gshape)), dtype=torch.int)
            _, _, chk = arr0.comm.chunk(out_shape, s0 if s0 is not None else s1)
            for i in range(len(out_shape)):
                chunk_map[arr0.comm.rank, i] = chk[i].stop - chk[i].start
            chunk_map_comm = arr0.comm.Iallreduce(MPI.IN_PLACE, chunk_map, MPI.SUM)

            lshape_map_comm.Wait()
            chunk_map_comm.Wait()

            if s0 is not None:
                send_slice = [slice(None)] * arr0.ndim
                keep_slice = [slice(None)] * arr0.ndim
                # data is first front-loaded onto the first size/2 processes
                for spr in range(1, arr0.comm.size):
                    if arr0.comm.rank == spr:
                        for pr in range(spr):
                            send_amt = abs((chunk_map[pr, axis] - lshape_map[0, pr, axis]).item())
                            send_amt = (
                                send_amt if send_amt < t_arr0.shape[axis] else t_arr0.shape[axis]
                            )
                            if send_amt:
                                send_slice[arr0.split] = slice(0, send_amt)
                                keep_slice[arr0.split] = slice(send_amt, t_arr0.shape[axis])
                                send = arr0.comm.Isend(
                                    t_arr0[send_slice].clone(),
                                    dest=pr,
                                    tag=pr + arr0.comm.size + spr,
                                )
                                t_arr0 = t_arr0[keep_slice].clone()
                                send.Wait()
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
                            t_arr0 = torch.cat((t_arr0, data), dim=arr0.split)
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
                                send_amt if send_amt < t_arr1.shape[axis] else t_arr1.shape[axis]
                            )
                            if send_amt:
                                send_slice[axis] = slice(
                                    t_arr1.shape[axis] - send_amt, t_arr1.shape[axis]
                                )
                                keep_slice[axis] = slice(0, t_arr1.shape[axis] - send_amt)
                                send = arr1.comm.Isend(
                                    t_arr1[send_slice].clone(),
                                    dest=pr,
                                    tag=pr + arr1.comm.size + spr,
                                )
                                t_arr1 = t_arr1[keep_slice].clone()
                                send.Wait()
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
                            t_arr1 = torch.cat((data, t_arr1), dim=axis)
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
                    t_arr0 = t_arr0[lcl_slice].clone().squeeze()
                ttl = chunk_map[0, axis].item()
                for en in range(1, arr0.comm.size):
                    sz = chunk_map[en, axis]
                    if arr0.comm.rank == en:
                        lcl_slice = [slice(None)] * arr0.ndim
                        lcl_slice[axis] = slice(ttl, sz.item() + ttl, 1)
                        t_arr0 = t_arr0[lcl_slice].clone().squeeze()
                    ttl += sz.item()

                if len(t_arr0.shape) < len(t_arr1.shape):
                    t_arr0.unsqueeze_(axis)

            if s1 is None:
                arb_slice = [None] * len(arr0.shape)
                for c in range(len(chunk_map)):
                    arb_slice[axis] = c
                    chunk_map[arb_slice] -= lshape_map[tuple([0] + arb_slice)]

                # get the desired data in arr1 on each node with a local slice
                if arr1.comm.rank == arr1.comm.size - 1:
                    lcl_slice = [slice(None)] * arr1.ndim
                    lcl_slice[axis] = slice(
                        t_arr1.shape[axis] - chunk_map[-1, axis].item(), t_arr1.shape[axis], 1
                    )
                    t_arr1 = t_arr1[lcl_slice].clone().squeeze()
                ttl = chunk_map[-1, axis].item()
                for en in range(arr1.comm.size - 2, -1, -1):
                    sz = chunk_map[en, axis]
                    if arr1.comm.rank == en:
                        lcl_slice = [slice(None)] * arr1.ndim
                        lcl_slice[axis] = slice(
                            t_arr1.shape[axis] - (sz.item() + ttl), t_arr1.shape[axis] - ttl, 1
                        )
                        t_arr1 = t_arr1[lcl_slice].clone().squeeze()
                    ttl += sz.item()
                if len(t_arr1.shape) < len(t_arr0.shape):
                    t_arr1.unsqueeze_(axis)

            res = torch.cat((t_arr0, t_arr1), dim=axis)
            out = factories.array(
                res,
                is_split=s0 if s0 is not None else s1,
                dtype=out_dtype,
                device=arr0.device,
                comm=arr0.comm,
            )

            return out


def diag(a: DNDarray, offset: int = 0) -> DNDarray:
    """
    Extract a diagonal or construct a diagonal array.
    See the documentation for :func:`diagonal` for more information about extracting the diagonal.

    Parameters
    ----------
    a: DNDarray
        The array holding data for creating a diagonal array or extracting a diagonal.
        If `a` is a 1-dimensional array, a diagonal 2d-array will be returned.
        If `a` is a n-dimensional array with n > 1 the diagonal entries will be returned in an n-1 dimensional array.
    offset: int, optional
        The offset from the main diagonal.
        Offset greater than zero means above the main diagonal, smaller than zero is below the main diagonal.

    See Also
    --------
    :func:`diagonal`

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.array([1, 2])
    >>> ht.diag(a)
    DNDarray([[1, 0],
              [0, 2]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diag(a, offset=1)
    DNDarray([[0, 1, 0],
              [0, 0, 2],
              [0, 0, 0]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.equal(ht.diag(ht.diag(a)), a)
    True
    >>> a = ht.array([[1, 2], [3, 4]])
    >>> ht.diag(a)
    DNDarray([1, 4], dtype=ht.int64, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(a)

    if len(a.shape) > 1:
        return diagonal(a, offset=offset)
    elif len(a.shape) < 1:
        raise ValueError("input array must be of dimension 1 or greater")
    if not isinstance(offset, int):
        raise ValueError("offset must be an integer, got", type(offset))

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
    local[indices_x, indices_y] = a.larray[indices_x]

    return factories.array(local, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm)


def diagonal(a: DNDarray, offset: int = 0, dim1: int = 0, dim2: int = 1) -> DNDarray:
    """
    Extract a diagonal of an n-dimensional array with n > 1.
    The returned array will be of dimension n-1.

    Parameters
    ----------
    a: DNDarray
        The array of which the diagonal should be extracted.
    offset: int, optional
        The offset from the main diagonal.
        Offset greater than zero means above the main diagonal, smaller than zero is below the main diagonal.
        Default is 0 which means the main diagonal will be selected.
    dim1: int, optional
        First dimension with respect to which to take the diagonal.
    dim2: int, optional
        Second dimension with respect to which to take the diagonal.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.array([[1, 2], [3, 4]])
    >>> ht.diagonal(a)
    DNDarray([1, 4], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diagonal(a, offset=1)
    DNDarray([2], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diagonal(a, offset=-1)
    DNDarray([3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = ht.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    >>> ht.diagonal(a)
    DNDarray([[0, 6],
              [1, 7]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.diagonal(a, dim2=2)
    DNDarray([[0, 5],
              [2, 7]], dtype=ht.int64, device=cpu:0, split=None)
    """
    dim1, dim2 = stride_tricks.sanitize_axis(a.shape, (dim1, dim2))

    if dim1 == dim2:
        raise ValueError("Dim1 and dim2 need to be different")
    if not isinstance(a, DNDarray):
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
        result = torch.diagonal(a.larray, offset=offset, dim1=dim1, dim2=dim2).contiguous()
    else:
        vz = 1 if a.split == dim1 else -1
        off, _, _ = a.comm.chunk(a.shape, a.split)
        result = torch.diagonal(
            a.larray, offset=offset + vz * off, dim1=dim1, dim2=dim2
        ).contiguous()
    return factories.array(result, dtype=a.dtype, is_split=split, device=a.device, comm=a.comm)


def dsplit(x: Sequence[DNDarray, ...], indices_or_sections: Iterable) -> List[DNDarray, ...]:
    """
    Split array into multiple sub-DNDarrays along the 3rd axis (depth).
    Returns a list of sub-DNDarrays as copies of parts of `x`.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : int or 1-dimensional array_like (i.e. undistributed DNDarray, list or tuple)
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along the 3rd axis.
        If such a split is not possible, an error is raised.
        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along the 3rd axis
        the array is split.
        If an index exceeds the dimension of the array along the 3rd axis, an empty sub-DNDarray is returned correspondingly.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Notes
    -----
    Please refer to the split documentation. dsplit is equivalent to split with axis=2,
    the array is always split along the third axis provided the array dimension is greater than or equal to 3.

    See Also
    ------
    :func:`split`
    :func:`hsplit`
    :func:`vsplit`

    Examples
    --------
    >>> x = ht.array(24).reshape((2, 3, 4))
    >>> ht.dsplit(x, 2)
        [DNDarray([[[ 0,  1],
                   [ 4,  5],
                   [ 8,  9]],
                   [[12, 13],
                   [16, 17],
                   [20, 21]]]),
        DNDarray([[[ 2,  3],
                   [ 6,  7],
                   [10, 11]],
                   [[14, 15],
                   [18, 19],
                   [22, 23]]])]
    >>> ht.dsplit(x, [1, 4])
        [DNDarray([[[ 0],
                    [ 4],
                    [ 8]],
                   [[12],
                    [16],
                    [20]]]),
        DNDarray([[[ 1,  2,  3],
                    [ 5,  6,  7],
                    [ 9, 10, 11]],
                    [[13, 14, 15],
                     [17, 18, 19],
                     [21, 22, 23]]]),
        DNDarray([])]
    """
    return split(x, indices_or_sections, 2)


def expand_dims(a: DNDarray, axis: int) -> DNDarray:
    """
    Expand the shape of an array.
    Insert a new axis that will appear at the axis position in the expanded array shape.

    Parameters
    ----------
    a : DNDarray
        Input array to be expanded.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Raises
    ------
    ValueError
        If `axis` is not consistent with the available dimensions.

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
    # sanitize input
    sanitation.sanitize_in(a)

    # sanitize axis, introduce arbitrary dummy dimension to model expansion
    axis = stride_tricks.sanitize_axis(a.shape + (1,), axis)

    return DNDarray(
        a.larray.unsqueeze(dim=axis),
        a.shape[:axis] + (1,) + a.shape[axis:],
        a.dtype,
        a.split if a.split is None or a.split < axis else a.split + 1,
        a.device,
        a.comm,
        a.balanced,
    )


DNDarray.expand_dims = lambda self, axis: expand_dims(self, axis)
DNDarray.expand_dims.__doc__ = expand_dims.__doc__


def flatten(a: DNDarray) -> DNDarray:
    """
    Flattens an array into one dimension.

    Parameters
    ----------
    a : DNDarray
        Array to collapse

    Warning
    ----------
    If `a.split>0`, the array must be redistributed along the first axis (see :func:`resplit`).


    See Also
    --------
    :func:`ravel`

    Examples
    --------
    >>> a = ht.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    >>> ht.flatten(a)
    DNDarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=ht.int64, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(a)

    if a.split is None:
        return factories.array(
            torch.flatten(a.larray), dtype=a.dtype, is_split=None, device=a.device, comm=a.comm
        )

    if a.split > 0:
        a = resplit(a, 0)

    a = factories.array(
        torch.flatten(a.larray), dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm
    )
    a.balance_()

    return a


DNDarray.flatten = lambda self: flatten(self)
DNDarray.flatten.__doc__ = flatten.__doc__


def flip(a: DNDarray, axis: Union[int, Tuple[int, ...]] = None) -> DNDarray:
    """
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    a: DNDarray
        Input array to be flipped
    axis: int or Tuple[int,...]
        A list of axes to be flipped

    See Also
    --------
    :func:`fliplr`
    :func:`flipud`

    Examples
    --------
    >>> a = ht.array([[0,1],[2,3]])
    >>> ht.flip(a, [0])
    DNDarray([[2, 3],
              [0, 1]], dtype=ht.int64, device=cpu:0, split=None)
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

    flipped = torch.flip(a.larray, axis)

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
    received = torch.empty(new_lshape, dtype=a.larray.dtype, device=a.device.torch_device)
    a.comm.Recv(received, source=dest_proc)

    res = factories.array(received, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm)
    res.balance_()  # after swapping, first processes may be empty
    req.Wait()
    return res


def fliplr(a: DNDarray) -> DNDarray:
    """
    Flip array in the left/right direction. If `a.ndim>2`, flip along dimension 1.

    Parameters
    ----------
    a: DNDarray
        Input array to be flipped, must be at least 2-D

    See Also
    --------
    :func:`flip`
    :func:`flipud`

    Examples
    --------
    >>> a = ht.array([[0,1],[2,3]])
    >>> ht.fliplr(a)
    DNDarray([[1, 0],
              [3, 2]], dtype=ht.int64, device=cpu:0, split=None)
    >>> b = ht.array([[0,1,2],[3,4,5]], split=0)
    >>> ht.fliplr(b)
    (1/2) tensor([[2, 1, 0]])
    (2/2) tensor([[5, 4, 3]])
    """
    return flip(a, 1)


def flipud(a: DNDarray) -> DNDarray:
    """
    Flip array in the up/down direction.

    Parameters
    ----------
    a: DNDarray
        Input array to be flipped

    See Also
    --------
    :func:`flip`
    :func:`fliplr`

    Examples
    --------
    >>> a = ht.array([[0,1],[2,3]])
    >>> ht.flipud(a)
    DNDarray([[2, 3],
              [0, 1]], dtype=ht.int64, device=cpu:0, split=None))
    >>> b = ht.array([[0,1,2],[3,4,5]], split=0)
    >>> ht.flipud(b)
    (1/2) tensor([3,4,5])
    (2/2) tensor([0,1,2])
    """
    return flip(a, 0)


def hsplit(x: DNDarray, indices_or_sections: Iterable) -> List[DNDarray, ...]:
    """
    Split array into multiple sub-DNDarrays along the 2nd axis (horizontally/column-wise).
    Returns a list of sub-DNDarrays as copies of parts of `x`.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : int or 1-dimensional array_like (i.e. undistributed DNDarray, list or tuple)
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along the 2nd axis.
        If such a split is not possible, an error is raised.
        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along the 2nd axis
        the array is split.
        If an index exceeds the dimension of the array along the 2nd axis, an empty sub-DNDarray is returned correspondingly.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Notes
    -----
    Please refer to the split documentation. hsplit is nearly equivalent to split with axis=1,
    the array is always split along the second axis though, in contrary to split, regardless of the array dimension.

    See Also
    --------
    :func:`split`
    :func:`dsplit`
    :func:`vsplit`

    Examples
    --------
    >>> x = ht.arange(24).reshape((2, 4, 3))
    >>> ht.hsplit(x, 2)
        [DNDarray([[[ 0,  1,  2],
                   [ 3,  4,  5]],
                  [[12, 13, 14],
                   [15, 16, 17]]]),
        DNDarray([[[ 6,  7,  8],
                   [ 9, 10, 11]],
                  [[18, 19, 20],
                   [21, 22, 23]]])]
    >>> ht.hsplit(x, [1, 3])
        [DNDarray([[[ 0,  1,  2]],
                  [[12, 13, 14]]]),
        DNDarray([[[ 3,  4,  5],
                   [ 6,  7,  8]],
                  [[15, 16, 17],
                   [18, 19, 20]]]),
        DNDarray([[[ 9, 10, 11]],
                  [[21, 22, 23]]])]
    """
    sanitation.sanitize_in(x)

    if len(x.lshape) < 2:
        x = reshape(x, (1, x.lshape[0]))
        result = split(x, indices_or_sections, 1)
        result = [flatten(sub_array) for sub_array in result]
    else:
        result = split(x, indices_or_sections, 1)

    return result


def hstack(arrays: Sequence[DNDarray, ...]) -> DNDarray:
    """
    Stack arrays in sequence horizontally (column-wise).
    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis.

    Parameters
    ----------
    arrays : Sequence[DNDarray, ...]
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    See Also
    --------
    :func:`concatenate`
    :func:`stack`
    :func:`vstack`
    :func:`column_stack`
    :func:`row_stack`

    Examples
    --------
    >>> a = ht.array((1,2,3))
    >>> b = ht.array((2,3,4))
    >>> ht.hstack((a,b)).larray
    [0/1] tensor([1, 2, 3, 2, 3, 4])
    [1/1] tensor([1, 2, 3, 2, 3, 4])
    >>> a = ht.array((1,2,3), split=0)
    >>> b = ht.array((2,3,4), split=0)
    >>> ht.hstack((a,b)).larray
    [0/1] tensor([1, 2, 3])
    [1/1] tensor([2, 3, 4])
    >>> a = ht.array([[1],[2],[3]], split=0)
    >>> b = ht.array([[2],[3],[4]], split=0)
    >>> ht.hstack((a,b)).larray
    [0/1] tensor([[1, 2],
    [0/1]         [2, 3]])
    [1/1] tensor([[3, 4]])
    """
    arrays = list(arrays)
    axis = 1
    all_vec = False
    if len(arrays) == 2 and all(len(x.gshape) == 1 for x in arrays):
        axis = 0
        all_vec = True
    if not all_vec:
        for cn, arr in enumerate(arrays):
            if len(arr.gshape) == 1:
                arrays[cn] = arr.expand_dims(1)

    return concatenate(arrays, axis=axis)


def moveaxis(
    x: DNDarray, source: Union[int, Sequence[int]], destination: Union[int, Sequence[int]]
) -> DNDarray:
    """
    Moves axes at the positions in `source` to new positions.

    Parameters
    ----------
    x : DNDarray
        The input array.
    source : int or Sequence[int, ...]
        Original positions of the axes to move. These must be unique.
    destination : int or Sequence[int, ...]
        Destination positions for each of the original axes. These must also be unique.

    See Also
    --------
    ~heat.core.linalg.basics.transpose
        Permute the dimensions of an array.

    Raises
    ------
    TypeError
        If `source` or `destination` are not ints, lists or tuples.
    ValueError
        If `source` and `destination` do not have the same number of elements.


    Examples
    --------
    >>> x = ht.zeros((3, 4, 5))
    >>> ht.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> ht.moveaxis(x, -1, 0).shape
    (5, 3, 4)
    """
    if isinstance(source, int):
        source = (source,)
    if isinstance(source, list):
        source = tuple(source)
    try:
        source = stride_tricks.sanitize_axis(x.shape, source)
    except TypeError:
        raise TypeError("'source' must be ints, lists or tuples.")

    if isinstance(destination, int):
        destination = (destination,)
    if isinstance(destination, list):
        destination = tuple(destination)
    try:
        destination = stride_tricks.sanitize_axis(x.shape, destination)
    except TypeError:
        raise TypeError("'destination' must be ints, lists or tuples.")

    if len(source) != len(destination):
        raise ValueError("'source' and 'destination' must have the same number of elements.")

    order = [n for n in range(x.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return linalg.transpose(x, order)


def pad(
    array: DNDarray,
    pad_width: Union[int, Sequence[Sequence[int, int], ...]],
    mode: str = "constant",
    constant_values: int = 0,
) -> DNDarray:
    """
    Pads tensor with a specific value (default=0).
    (Not all dimensions supported)

    Parameters
    ----------
    array : DNDarray
        Array to be padded
    pad_width: Union[int, Sequence[Sequence[int, int], ...]]
        Number of values padded to the edges of each axis. ((before_1, after_1),...(before_N, after_N)) unique pad widths for each axis.
        Determines how many elements are padded along which dimension.\n
        Shortcuts:

            - ((before, after),)  or (before, after): before and after pad width for each axis.
            - (pad_width,) or int: before = after = pad width for all axes.

        Therefore:

        - pad last dimension: (padding_left, padding_right)
        - pad last 2 dimensions: ((padding_top, padding_bottom),(padding_left, padding_right))
        - pad last 3 dimensions: ((padding_front, padding_back),(padding_top, padding_bottom),(paddling_left, padding_right) )
        - ... (same pattern)
    mode : str, optional
        - 'constant' (default): Pads the input tensor boundaries with a constant value. This is available for arbitrary dimensions
    constant_values: Union[int, float, Sequence[Sequence[int,int], ...], Sequence[Sequence[float,float], ...]]
        Number or tuple of 2-element-sequences (containing numbers), optional (default=0)
        The fill values for each axis (1 tuple per axis).
        ((before_1, after_1), ... (before_N, after_N)) unique pad values for each axis.

        Shortcuts:

            - ((before, after),) or (before, after): before and after padding values for each axis.
            - (value,) or int: before = after = padding value for all axes.


    Notes
    -----------
    This function follows the principle of datatype integrity.
    Therefore, an array can only be padded with values of the same datatype.
    All values that violate this rule are implicitly cast to the datatype of the `DNDarray`.

    Examples
    --------
    >>> a = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    >>> b = ht.array(a, split = 0)
    Pad last dimension
    >>> c = ht.pad(b, (2,1), constant_values=1)
    tensor([[[ 1,  1,  0,  1,  2,  3,  1],
         [ 1,  1,  4,  5,  6,  7,  1],
         [ 1,  1,  8,  9, 10, 11,  1]],
        [[ 1,  1, 12, 13, 14, 15,  1],
         [ 1,  1, 16, 17, 18, 19,  1],
         [ 1,  1, 20, 21, 22, 23,  1]]])
    Pad last 2 dimensions
    >>> d = ht.pad(b, [(1,0), (2,1)])
    DNDarray([[[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  1,  2,  3,  0],
               [ 0,  0,  4,  5,  6,  7,  0],
               [ 0,  0,  8,  9, 10, 11,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0, 12, 13, 14, 15,  0],
               [ 0,  0, 16, 17, 18, 19,  0],
               [ 0,  0, 20, 21, 22, 23,  0]]], dtype=ht.int64, device=cpu:0, split=0)
    Pad last 3 dimensions
    >>> e = ht.pad(b, ((2,1), [1,0], (2,1)))
    DNDarray([[[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  1,  2,  3,  0],
               [ 0,  0,  4,  5,  6,  7,  0],
               [ 0,  0,  8,  9, 10, 11,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0, 12, 13, 14, 15,  0],
               [ 0,  0, 16, 17, 18, 19,  0],
               [ 0,  0, 20, 21, 22, 23,  0]],

              [[ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0]]], dtype=ht.int64, device=cpu:0, split=0)
    """
    # early out if pad width is 0
    if pad_width == 0:
        return array

    if not isinstance(array, DNDarray):
        raise TypeError(f"expected array to be a ht.DNDarray, but was {type(array)}")

    if not isinstance(mode, str):
        raise TypeError(f"expected mode to be a string, but was {type(mode)}")

    # shortcut int for all dimensions
    if isinstance(pad_width, int):
        pad = (pad_width,) * 2 * len(array.shape)

    elif not isinstance(pad_width, (tuple, list)):
        raise TypeError(
            f"expected pad_width to be an integer or a sequence (tuple or list), but was {type(pad_width)}"
        )

    # shortcut one sequence within a sequence for all dimensions - ((before,after), ) = pad_width
    elif len(pad_width) == 1:
        if isinstance(pad_width[0], int):
            pad = (pad_width[0],) * 2 * len(array.shape)
        elif not (isinstance(pad_width[0], (tuple, list))):
            raise TypeError(
                f"For shortcut option '1 sequence for all dimensions', expected element within pad_width to be a tuple or list, but was {type(pad_width[0])}"
            )
        elif len(pad_width[0]) == 2:
            pad = pad_width[0] * len(array.shape)
        else:
            raise ValueError(
                f"Pad_width {pad_width} invalid.\n Apart from shortcut options (--> documentation), "
                "each sequence within pad_width must contain 2 elements."
            )
    # shortcut - one sequence for all dimensions - (before,after) = pad_width
    elif len(pad_width) == 2 and isinstance(pad_width[0], int) and isinstance(pad_width[1], int):
        pad_width = tuple(pad_width)
        pad = pad_width * len(array.shape)

    # no shortcut - padding of various dimensions
    else:
        if any(not (isinstance(pad_tuple, (tuple, list))) for pad_tuple in pad_width):
            raise TypeError(
                f"Invalid type for pad_width {pad_width}.\nApart from shortcut options (--> documentation),"
                "pad_width has to be a sequence of (2 elements) sequences (sequence=tuple or list)."
            )
        pad = ()
        # Transform numpy pad_width to torch pad (--> one tuple containing all padding spans)
        for pad_tuple in pad_width:
            if isinstance(pad_tuple, list):
                pad_tuple = tuple(pad_tuple)
            pad = pad_tuple + pad

        if len(pad) % 2 != 0:
            raise ValueError(
                f"Pad_width {pad_width} invalid.\n Apart from shortcut options (--> documentation), "
                "each sequence within pad_width must contain 2 elements."
            )

        if len(pad) // 2 > len(array.shape):
            raise ValueError(
                f"Not enough dimensions to pad.\n"
                f"Padding a {len(array.shape)}-dimensional tensor for {len(pad) // 2}"
                f" dimensions is not possible."
            )

    # value_tuple = all padding values stored in 1 tuple
    if isinstance(constant_values, (tuple, list)):
        value_tuple = ()
        # sequences for each dimension defined within one sequence
        if isinstance(constant_values[0], (tuple, list)):
            # one sequence for all dimensions - values = ((before, after),)
            if len(constant_values) == 1:
                value_tuple = constant_values[0] * (len(pad) // 2)
            else:
                for value_pair in constant_values:
                    if isinstance(value_pair, tuple):
                        pass
                    elif isinstance(value_pair, list):
                        value_pair = tuple(value_pair)
                    else:
                        raise TypeError(
                            f"Value pair {value_pair} within values invalid. Expected all elements within values to be sequences(list/tuple),"
                            f"but one was: {type(value_pair)}"
                        )
                    value_tuple = value_pair + value_tuple

            if len(value_tuple) % 2 != 0:
                raise ValueError(
                    f"Expected values to contain an even amount of elements, but got {len(value_tuple)}"
                )

        # One sequence for all dimensions - values = (before, after)
        elif len(constant_values) == 2:
            value_tuple = constant_values * (len(pad) // 2)

    rank_array = len(array.shape)
    amount_pad_dim = len(pad) // 2
    pad_dim = [rank_array - i for i in range(1, amount_pad_dim + 1)]

    array_torch = array.larray

    if array.split is not None:
        counts = array.comm.counts_displs_shape(array.gshape, array.split)[0]
        amount_of_processes = len(counts)

    # calculate gshape for output tensor
    output_shape_list = list(array.gshape)

    for i in range(0, len(pad), 2):
        output_shape_list[-((i // 2) + 1)] += sum(pad[i : i + 2])

    output_shape = tuple(output_shape_list)

    # -------------------------------------------------------------------------------------------------------------------
    # CASE 1: Padding in non split dimension or no distribution at all
    # ------------------------------------------------------------------------------------------------------------------
    # no data
    if 0 in list(array.lshape):
        adapted_lshape_list = [
            0 if i == array.split else output_shape[i] for i in range(len(output_shape))
        ]
        adapted_lshape = tuple(adapted_lshape_list)
        padded_torch_tensor = torch.empty(
            adapted_lshape, dtype=array._DNDarray__array.dtype, device=array.device.torch_device
        )
    else:
        if array.split is None or array.split not in pad_dim or amount_of_processes == 1:
            # values = scalar
            if isinstance(constant_values, int) or isinstance(constant_values, float):
                padded_torch_tensor = torch.nn.functional.pad(
                    array_torch, pad, mode, constant_values
                )
            # values = sequence with one value for all dimensions
            elif len(constant_values) == 1 and (
                isinstance(constant_values[0], int) or isinstance(constant_values[0], float)
            ):
                padded_torch_tensor = torch.nn.functional.pad(
                    array_torch, pad, mode, constant_values[0]
                )
            else:
                padded_torch_tensor = array_torch
                for i in range(len(value_tuple) - 1, -1, -1):
                    pad_list = [0] * 2 * rank_array
                    pad_list[i] = pad[i]
                    pad_tuple = tuple(pad_list)
                    padded_torch_tensor = torch.nn.functional.pad(
                        padded_torch_tensor, pad_tuple, mode, value_tuple[i]
                    )
        else:
            # ------------------------------------------------------------------------------------------------------------------
            # CASE 2: padding in split dimension and function runs on more than 1 process
            #
            # Pad only first/last tensor portion on node (i.e. only beginning/end in split dimension)
            # --> "Calculate" pad tuple for the corresponding tensor portion/ the two indices which have to be set to zero
            #      in different paddings depending on the dimension
            #       Calculate the index of the first element in tuple that has to change/set to zero in
            #       some dimensions (the following is the second)
            # ------------------------------------------------------------------------------------------------------------------

            pad_beginning_list = list(pad)
            pad_end_list = list(pad)
            pad_middle_list = list(pad)

            # calculate the corresponding pad tuples
            first_idx_set_zero = 2 * (rank_array - array.split - 1)

            pad_end_list[first_idx_set_zero] = 0
            pad_beginning_list[first_idx_set_zero + 1] = 0
            pad_middle_list[first_idx_set_zero : first_idx_set_zero + 2] = [0, 0]

            pad_beginning = tuple(pad_beginning_list)
            pad_end = tuple(pad_end_list)
            pad_middle = tuple(pad_middle_list)

            if amount_of_processes >= array.shape[array.split]:
                last_ps_with_data = array.shape[array.split] - 1
            else:
                last_ps_with_data = amount_of_processes - 1

            rank = array.comm.rank

            # first process - pad beginning
            if rank == 0:
                pad_tuple_curr_rank = pad_beginning

            # last process - pad end
            elif rank == last_ps_with_data:
                pad_tuple_curr_rank = pad_end

            # pad middle
            else:
                pad_tuple_curr_rank = pad_middle

            if isinstance(constant_values, (int, float)):
                padded_torch_tensor = torch.nn.functional.pad(
                    array_torch, pad_tuple_curr_rank, mode, constant_values
                )

            elif len(constant_values) == 1 and isinstance(constant_values[0], (int, float)):
                padded_torch_tensor = torch.nn.functional.pad(
                    array_torch, pad_tuple_curr_rank, mode, constant_values[0]
                )

            else:
                padded_torch_tensor = array_torch
                for i in range(len(value_tuple) - 1, -1, -1):
                    pad_list = [0] * 2 * rank_array
                    pad_list[i] = pad_tuple_curr_rank[i]
                    pad_tuple = tuple(pad_list)
                    padded_torch_tensor = torch.nn.functional.pad(
                        padded_torch_tensor, pad_tuple, mode, value_tuple[i]
                    )

    padded_tensor = factories.array(
        padded_torch_tensor,
        dtype=array.dtype,
        is_split=array.split,
        device=array.device,
        comm=array.comm,
    )

    padded_tensor.balance_()

    return padded_tensor


def ravel(a: DNDarray) -> DNDarray:
    """
    Return a flattened view of `a` if possible. A copy is returned otherwise.

    Parameters
    ----------
    a : DNDarray
        array to collapse

    Notes
    ------
    Returning a view of distributed data is only possible when `split != 0`. The returned DNDarray may be unbalanced.
    Otherwise, data must be communicated among processes, and `ravel` falls back to `flatten`.

    See Also
    --------
    :func:`flatten`

    Examples
    --------
    >>> a = ht.ones((2,3), split=0)
    >>> b = ht.ravel(a)
    >>> a[0,0] = 4
    >>> b
    DNDarray([4., 1., 1., 1., 1., 1.], dtype=ht.float32, device=cpu:0, split=0)
    """
    sanitation.sanitize_in(a)

    if a.split is None:
        return factories.array(
            torch.flatten(a._DNDarray__array),
            dtype=a.dtype,
            copy=False,
            is_split=None,
            device=a.device,
            comm=a.comm,
        )

    # Redistribution necessary
    if a.split != 0:
        return flatten(a)

    result = factories.array(
        torch.flatten(a._DNDarray__array),
        dtype=a.dtype,
        copy=False,
        is_split=a.split,
        device=a.device,
        comm=a.comm,
    )

    return result


def redistribute(
    arr: DNDarray, lshape_map: torch.Tensor = None, target_map: torch.Tensor = None
) -> DNDarray:
    """
    Redistributes the data of the :class:`DNDarray` *along the split axis* to match the given target map.
    This function does not modify the non-split dimensions of the ``DNDarray``.
    This is an abstraction and extension of the balance function.

    Parameters
    ----------
    arr: DNDarray
        DNDarray to redistribute
    lshape_map : torch.Tensor, optional
        The current lshape of processes.
        Units are ``[rank, lshape]``.
    target_map : torch.Tensor, optional
        The desired distribution across the processes.
        Units are ``[rank, target lshape]``.
        Note: the only important parts of the target map are the values along the split axis,
        values which are not along this axis are there to mimic the shape of the ``lshape_map``.

    Examples
    --------
    >>> st = ht.ones((50, 81, 67), split=2)
    >>> target_map = torch.zeros((st.comm.size, 3), dtype=torch.int64)
    >>> target_map[0, 2] = 67
    >>> print(target_map)
    [0/2] tensor([[ 0,  0, 67],
    [0/2]         [ 0,  0,  0],
    [0/2]         [ 0,  0,  0]], dtype=torch.int32)
    [1/2] tensor([[ 0,  0, 67],
    [1/2]         [ 0,  0,  0],
    [1/2]         [ 0,  0,  0]], dtype=torch.int32)
    [2/2] tensor([[ 0,  0, 67],
    [2/2]         [ 0,  0,  0],
    [2/2]         [ 0,  0,  0]], dtype=torch.int32)
    >>> print(st.lshape)
    [0/2] (50, 81, 23)
    [1/2] (50, 81, 22)
    [2/2] (50, 81, 22)
    >>> ht.redistribute_(st, target_map=target_map)
    >>> print(st.lshape)
    [0/2] (50, 81, 67)
    [1/2] (50, 81, 0)
    [2/2] (50, 81, 0)
    """
    arr2 = arr.copy()
    arr2.redistribute_(lshape_map=lshape_map, target_map=target_map)
    return arr2


DNDarray.redistribute = lambda arr, lshape_map=None, target_map=None: redistribute(
    arr, lshape_map, target_map
)
DNDarray.redistribute.__doc__ = redistribute.__doc__


def repeat(a: Iterable, repeats: Iterable, axis: Optional[int] = None) -> DNDarray:
    """
    Creates a new `DNDarray` by repeating elements of array `a`. The output has
    the same shape as `a`, except along the given axis. If axis is None, this
    function returns a flattened `DNDarray`.

    Parameters
    ----------
    a : array_like (i.e. int, float, or tuple/ list/ np.ndarray/ ht.DNDarray of ints/floats)
        Array containing the elements to be repeated.
    repeats : int, or 1-dimensional/ DNDarray/ np.ndarray/ list/ tuple of ints
        The number of repetitions for each element, indicates broadcast if int or array_like of 1 element.
        In this case, the given value is broadcasted to fit the shape of the given axis.
        Otherwise, its length must be the same as a in the specified axis. To put it differently, the
        amount of repetitions has to be determined for each element in the corresponding dimension
        (or in all dimensions if axis is None).
    axis: int, optional
        The axis along which to repeat values. By default, use the flattened input array and return a flat output
        array.

    Examples
    --------
    >>> ht.repeat(3, 4)
    DNDarray([3, 3, 3, 3])

    >>> x = ht.array([[1,2],[3,4]])
    >>> ht.repeat(x, 2)
    DNDarray([1, 1, 2, 2, 3, 3, 4, 4])

    >>> x = ht.array([[1,2],[3,4]])
    >>> ht.repeat(x, [0, 1, 2, 0])
    DNDarray([2, 3, 3])

    >>> ht.repeat(x, [1,2], axis=0)
    DNDarray([[1, 2],
            [3, 4],
            [3, 4]])
    """
    # sanitation `a`
    if not isinstance(a, DNDarray):
        if isinstance(a, (int, float)):
            a = factories.array([a])
        elif isinstance(a, (tuple, list, np.ndarray)):
            a = factories.array(a)
        else:
            raise TypeError(
                f"`a` must be a ht.DNDarray, np.ndarray, list, tuple, integer, or float, currently: {type(a)}"
            )

    # sanitation `axis`
    if axis is not None and not isinstance(axis, int):
        raise TypeError(f"`axis` must be an integer or None, currently: {type(axis)}")

    if axis is not None and (axis >= len(a.shape) or axis < 0):
        raise ValueError(
            f"Invalid input for `axis`. Value has to be either None or between 0 and {len(a.shape) - 1}, not {axis}."
        )

    # sanitation `repeats`
    if not isinstance(repeats, (int, list, tuple, np.ndarray, DNDarray)):
        raise TypeError(
            f"`repeats` must be an integer, list, tuple, np.ndarray or ht.DNDarray of integers, currently: {type(repeats)}"
        )

    # no broadcast implied
    if not isinstance(repeats, int):
        # make sure everything inside `repeats` is int
        if isinstance(repeats, DNDarray):
            if repeats.dtype == types.int64:
                pass
            elif types.can_cast(repeats.dtype, types.int64):
                repeats = factories.array(
                    repeats,
                    dtype=types.int64,
                    is_split=repeats.split,
                    device=repeats.device,
                    comm=repeats.comm,
                )
            else:
                raise TypeError(
                    f"Invalid dtype for ht.DNDarray `repeats`. Has to be integer, but was {repeats.dtype}"
                )
        elif isinstance(repeats, np.ndarray):
            if not types.can_cast(repeats.dtype.type, types.int64):
                raise TypeError(
                    f"Invalid dtype for np.ndarray `repeats`. Has to be integer, but was {repeats.dtype.type}"
                )
            repeats = factories.array(
                repeats, dtype=types.int64, is_split=None, device=a.device, comm=a.comm
            )
        elif not all(isinstance(r, int) for r in repeats):
            raise TypeError(
                "Invalid type within `repeats`. All components of `repeats` must be integers."
            )
        else:
            repeats = factories.array(
                repeats, dtype=types.int64, is_split=None, device=a.device, comm=a.comm
            )

        # check `repeats` is not empty
        if repeats.gnumel == 0:
            raise ValueError("Invalid input for `repeats`. `repeats` must contain data.")

        # check `repeats` is 1-dimensional
        if len(repeats.shape) != 1:
            raise ValueError(
                f"Invalid input for `repeats`. `repeats` must be a 1d-object or integer, but was {len(repeats.shape)}-dimensional."
            )

    # start of algorithm

    if 0 in a.gshape:
        return a

    # Broadcast (via int or 1-element DNDarray)
    if isinstance(repeats, int) or repeats.gnumel == 1:
        if axis is None and a.split is not None and a.split != 0:
            warnings.warn(
                f"If axis is None, `a` has to be split along axis 0 (not {a.split}) if distributed.\n`a` will be copied with new split axis 0."
            )
            a = resplit(a, 0)
        if isinstance(repeats, int):
            repeated_array_torch = torch.repeat_interleave(a._DNDarray__array, repeats, axis)
        else:
            if repeats.split is not None:
                warnings.warn(
                    "For broadcast via array_like repeats, `repeats` must not be "
                    "distributed (along axis {}).\n`repeats` will be "
                    "copied with new split axis None.".format(repeats.split)
                )
                repeats = resplit(repeats, None)
            repeated_array_torch = torch.repeat_interleave(
                a._DNDarray__array, repeats._DNDarray__array, axis
            )
    # No broadcast
    else:
        # check if the data chunks of `repeats` and/or `a` have to be (re)distributed before call of torch function.

        # UNDISTRIBUTED CASE (a not distributed)
        if a.split is None:
            if repeats.split is not None:
                warnings.warn(
                    f"If `a` is undistributed, `repeats` also has to be undistributed (not split along axis {repeats.split}).\n`repeats` will be copied "
                    "with new split axis None."
                )
                repeats = resplit(repeats, None)

            # Check correct input
            if axis is None:
                # check matching shapes (repetition defined for every element)
                if a.gnumel != repeats.gnumel:
                    raise ValueError(
                        f"Invalid input. Sizes of flattened `a` ({a.gnumel}) and `repeats` ({repeats.gnumel}) are not same. "
                        "Please revise your definition specifying repetitions for all elements of the DNDarray `a` "
                        "or replace repeats with a single scalar."
                    )
            # axis is not None
            elif a.lshape[axis] != repeats.lnumel:
                raise ValueError(
                    f"Invalid input. Amount of elements of `repeats` ({repeats.lnumel}) and of `a` in the specified axis ({a.lshape[axis]}) "
                    "are not the same. Please revise your definition specifying repetitions for all elements "
                    "of the DNDarray `a` or replace `repeats` with a single scalar"
                )
        # DISTRIBUTED CASE (a distributed)
        elif axis is None:
            if a.gnumel != repeats.gnumel:
                raise ValueError(
                    f"Invalid input. Sizes of flattened `a` ({a.gnumel}) and `repeats` ({repeats.gnumel}) are not same. "
                    "Please revise your definition specifying repetitions for all elements of the DNDarray `a` "
                    "or replace `repeats` with a single scalar."
                )

            if a.split != 0:
                warnings.warn(
                    f"If `axis` is None, `a` has to be split along axis 0 (not {a.split}) if distributed.\n`a` will be copied "
                    "with new split axis 0."
                )
                a = resplit(a, 0)

            repeats = repeats.reshape(a.gshape)
            if repeats.split != 0:
                warnings.warn(
                    f"If `axis` is None, `repeats` has to be split along axis 0 (not {repeats.split}) if distributed.\n`repeats` will be copied "
                    "with new split axis 0."
                )
                repeats = resplit(repeats, 0)
            flatten_repeats_t = torch.flatten(repeats._DNDarray__array)
            repeats = factories.array(
                flatten_repeats_t,
                is_split=repeats.split,
                device=repeats.device,
                comm=repeats.comm,
            )

        # axis is not None
        elif a.split == axis:
            if repeats.split != 0:
                warnings.warn(
                    f"If `axis` equals `a.split`, `repeats` has to be split along axis 0 (not {repeats.split}) if distributed.\n`repeats` will be copied "
                    "with new split axis 0"
                )
                repeats = resplit(repeats, 0)

        # a.split != axis
        else:
            if repeats.split is not None:
                warnings.warn(
                    f"If `axis` != `a.split`, `repeast` must not be distributed (along axis {repeats.split}).\n`repeats` will be copied with new "
                    "split axis None."
                )
                repeats = resplit(repeats, None)

            if a.lshape[axis] != repeats.lnumel:
                raise ValueError(
                    f"Invalid input. Amount of elements of `repeats` ({repeats.lnumel}) and of `a` in the specified axis ({a.lshape[axis]}) "
                    "are not the same. Please revise your definition specifying repetitions for all elements "
                    "of the DNDarray `a` or replace `repeats` with a single scalar"
                )

        repeated_array_torch = torch.repeat_interleave(
            a._DNDarray__array, repeats._DNDarray__array, axis
        )

    repeated_array = factories.array(
        repeated_array_torch, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm
    )
    repeated_array.balance_()

    return repeated_array


def reshape(a: DNDarray, *shape: Union[int, Tuple[int, ...]], **kwargs) -> DNDarray:
    """
    Returns an array with the same data and number of elements as `a`, but with the specified shape.

    Parameters
    ----------
    a : DNDarray
        The input array
    shape : Union[int, Tuple[int,...]]
        Shape of the new array. Must be compatible with the original shape. If an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    new_split : int, optional
        The distribution axis of the reshaped array. Default: None (same distribution axis as `a`).

    Raises
    ------
    ValueError
        If the number of elements in the new shape is inconsistent with the input data.

    Notes
    -----
    `reshape()` might require significant communication among processes. Operating along split axis 0 is recommended.

    See Also
    --------
    :func:`ravel`

    Examples
    --------
    >>> a = ht.zeros((3,4))
    >>> ht.reshape(a, (4,3))
    DNDarray([[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> a = ht.linspace(0, 14, 8, split=0)
    >>> ht.reshape(a, (2,4))
    (1/2) tensor([[0., 2., 4., 6.]])
    (2/2) tensor([[ 8., 10., 12., 14.]])
    """
    if not isinstance(a, DNDarray):
        raise TypeError("'a' must be a DNDarray, currently {}".format(type(a)))

    # use numpys _ShapeLike but expand to handle torch and heat Tensors
    np_proxy = np.lib.stride_tricks.as_strided(np.ones(1), a.gshape, [0] * a.ndim, writeable=False)
    try:
        np_proxy.reshape(shape)  # numpy defines their own _ShapeLike
    except TypeError as e:  # handle Tensors and DNDarrays
        try:  # make shape a np.ndarray
            if len(shape) == 1:
                shape = shape[0]
            if hasattr(shape, "cpu"):  # move to cpu
                shape = shape.cpu()
            if hasattr(shape, "detach"):  # torch.Tensors have to detach before numpy call
                shape = shape.detach()
            if hasattr(shape, "numpy"):  # for DNDarrays
                shape = shape.numpy()
            else:  # Try to coerce everything else.
                shape = np.asarray(shape)
        except Exception:
            raise TypeError(e)
    shape = np_proxy.reshape(shape).shape  # sanitized shape according to numpy

    # tdtype, tdevice = a.dtype.torch_type(), a.device.torch_device
    tdevice = a.device.torch_device

    # always operate along split axis 0
    orig_split = a.split
    if a.split is not None:
        a.resplit_(axis=0)

    local_a = a.larray

    # check new_split parameter
    new_split = kwargs.get("new_split")
    if new_split is None:
        new_split = orig_split
    new_split = stride_tricks.sanitize_axis(shape, new_split)

    if not a.is_distributed():
        if a.comm.size > 1 and new_split is not None:
            # keep local slice only
            _, _, local_slice = a.comm.chunk(shape, new_split)
            _ = local_a.reshape(shape)
            local_a = _[local_slice].contiguous()
            del _
        else:
            local_a = local_a.reshape(shape)
        return DNDarray(
            local_a,
            gshape=shape,
            dtype=a.dtype,
            split=new_split,
            device=a.device,
            comm=a.comm,
            balanced=True,
        )

    lshape_map = a.lshape_map
    rank = a.comm.rank
    size = a.comm.size

    # flatten dimensions from split axis on, e.g. if split = 1, (3,4,5) -> (3,20)
    local_elements_off_split = torch.prod(lshape_map[:, a.split :], dim=1)
    first_local_shape = tuple(lshape_map[rank, : a.split].tolist()) + (
        local_elements_off_split[rank].item(),
    )
    local_a = local_a.reshape(first_local_shape)
    first_global_shape = tuple(lshape_map[rank, : a.split].tolist()) + (
        local_elements_off_split.sum().item(),
    )
    reshape_first_pass = DNDarray(
        local_a,
        gshape=first_global_shape,
        dtype=a.dtype,
        split=a.split,
        device=a.device,
        comm=a.comm,
        balanced=a.balanced,
    )
    new_lshape_map = lshape_map[:, : a.split + 1]
    new_lshape_map[:, a.split] = local_elements_off_split
    reshape_first_pass.__lshape_map = new_lshape_map

    # redistribute if necessary. All splits but a.split = 0 have been ruled out

    lshape_map = reshape_first_pass.lshape_map
    current_numel = torch.prod(lshape_map, dim=1)
    # calculate target number of elements on each rank
    target_numel = torch.zeros((size, len(shape)), dtype=torch.int64, device=tdevice)
    for i in range(size):
        _, local_shape, _ = a.comm.chunk(shape, a.split, rank=i)
        target_numel[i] = torch.tensor(local_shape)
        if i == rank:
            second_local_shape = local_shape
    target_numel = torch.prod(target_numel, dim=1)
    if (target_numel == current_numel).all():
        local_a = local_a.reshape(second_local_shape)
    else:
        # redistribution is necessary before reshaping
        target_map = lshape_map.clone()
        target_map[:, a.split] = target_numel
        reshape_first_pass.redistribute_(target_map=target_map)
        local_a = reshape_first_pass.larray.reshape(second_local_shape)
    reshape_final = DNDarray(
        local_a,
        gshape=shape,
        dtype=a.dtype,
        split=0,
        device=a.device,
        comm=a.comm,
        balanced=None,
    )
    reshape_final.resplit_(axis=new_split)

    return reshape_final


DNDarray.reshape = lambda self, *shape, **kwargs: reshape(self, *shape, **kwargs)
DNDarray.reshape.__doc__ = reshape.__doc__


def roll(
    x: DNDarray, shift: Union[int, Tuple[int]], axis: Optional[Union[int, Tuple[int]]] = None
) -> DNDarray:
    """
    Rolls array elements along a specified axis. Array elements that roll beyond the last position are re-introduced at the first position.
    Array elements that roll beyond the first position are re-introduced at the last position.

    Parameters
    ----------
    x : DNDarray
        input array
    shift : Union[int, Tuple[int, ...]]
        number of places by which the elements are shifted. If 'shift' is a tuple, then 'axis' must be a tuple of the same size, and each of
        the given axes is shifted by the corrresponding element in 'shift'. If 'shift' is an `int` and 'axis' a `tuple`, then the same shift
        is used for all specified axes.
    axis : Optional[Union[int, Tuple[int, ...]]]
        axis (or axes) along which elements to shift. If 'axis' is `None`, the array is flattened, shifted, and then restored to its original shape.
        Default: `None`.

    Raises
    ------
    TypeError
        If 'shift' or 'axis' is not of type `int`, `list` or `tuple`.
    ValueError
        If 'shift' and 'axis' are tuples with different sizes.

    Examples
    --------
    >>> a = ht.arange(20).reshape((4,5))
    >>> a
    DNDarray([[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.roll(a, 1)
    DNDarray([[19,  0,  1,  2,  3],
          [ 4,  5,  6,  7,  8],
          [ 9, 10, 11, 12, 13],
          [14, 15, 16, 17, 18]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.roll(a, -1, 0)
    DNDarray([[ 5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [ 0,  1,  2,  3,  4]], dtype=ht.int32, device=cpu:0, split=None)
    """
    sanitation.sanitize_in(x)

    if axis is None:
        return roll(x.flatten(), shift, 0).reshape(x.shape, new_split=x.split)

    # inputs are ints
    if isinstance(shift, int):
        if isinstance(axis, int):
            if x.split is not None and (axis == x.split or (axis + x.ndim) == x.split):
                # roll along split axis
                size = x.comm.Get_size()
                rank = x.comm.Get_rank()

                # local elements along axis:
                lshape_map = x.create_lshape_map(force_check=False)[:, x.split]
                cumsum_map = torch.cumsum(lshape_map, dim=0)  # cumulate along axis
                indices = torch.arange(size, device=x.device.torch_device)
                # NOTE Can be removed when min version>=1.9
                if "1.8." in torch.__version__:  # pragma: no cover
                    lshape_map = lshape_map.to(torch.int64)
                index_map = torch.repeat_interleave(indices, lshape_map)  # index -> process

                # compute index positions
                index_old = torch.arange(lshape_map[rank], device=x.device.torch_device)
                if rank > 0:
                    index_old += cumsum_map[rank - 1]

                send_index = (index_old + shift) % x.gshape[x.split]
                recv_index = (index_old - shift) % x.gshape[x.split]

                # exchange arrays
                recv = torch.empty_like(x.larray)
                recv_splits = torch.split(recv, 1, dim=x.split)
                recv_requests = [None for i in range(x.lshape[x.split])]

                for i in range(x.lshape[x.split]):
                    recv_requests[i] = x.comm.Irecv(
                        recv_splits[i], index_map[recv_index[i]], index_old[i]
                    )

                send_splits = torch.split(x.larray, 1, dim=x.split)
                send_requests = [None for i in range(x.lshape[x.split])]

                for i in range(x.lshape[x.split]):
                    send_requests[i] = x.comm.Isend(
                        send_splits[i], index_map[send_index[i]], send_index[i]
                    )

                for i in range(x.lshape[x.split]):
                    recv_requests[i].Wait()
                for i in range(x.lshape[x.split]):
                    send_requests[i].Wait()

                return DNDarray(recv, x.gshape, x.dtype, x.split, x.device, x.comm, x.balanced)

        else:  # pytorch does not support int / sequence combo at the time, make shift a list instead
            try:
                axis = sanitation.sanitize_sequence(axis)
            except TypeError:
                raise TypeError(f"axis must be a int, list or a tuple, got {type(axis)}")

            shift = [shift] * len(axis)

            return roll(x, shift, axis)

    else:  # input must be tuples now
        try:
            shift = sanitation.sanitize_sequence(shift)
        except TypeError:
            raise TypeError(f"shift must be an integer, list or a tuple, got {type(shift)}")

        try:
            axis = sanitation.sanitize_sequence(axis)
        except TypeError:
            raise TypeError(f"axis must be an integer, list or a tuple, got {type(axis)}")

        if len(shift) != len(axis):
            raise ValueError(
                f"shift and axis length must be the same, got {len(shift)} and {len(axis)}"
            )

        for i in range(len(shift)):
            if not isinstance(shift[i], int):
                raise TypeError(f"Element {i} in shift is not an integer, got {type(shift[i])}")
            if not isinstance(axis[i], int):
                raise TypeError(f"Element {i} in axis is not an integer, got {type(axis[i])}")

        if x.split is not None and (x.split in axis or (x.split - x.ndim) in axis):
            # remove split axis elements
            shift_split = 0
            for y in (x.split, x.split - x.ndim):
                idx = [i for i in range(len(axis)) if axis[i] == y]
                for i in idx:
                    shift_split += shift[i]
                for i in reversed(idx):
                    axis.remove(y)
                    del shift[i]

            # compute new array along split axis
            x = roll(x, shift_split, x.split)
            if len(axis) == 0:
                return x

    # use PyTorch for all other axes
    rolled = torch.roll(x.larray, shift, axis)
    return DNDarray(
        rolled,
        gshape=x.shape,
        dtype=x.dtype,
        split=x.split,
        device=x.device,
        comm=x.comm,
        balanced=x.balanced,
    )


def rot90(m: DNDarray, k: int = 1, axes: Sequence[int, int] = (0, 1)) -> DNDarray:
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    Rotation direction is from the first towards the second axis.

    Parameters
    ----------
    m : DNDarray
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes: (2,) Sequence[int, int]
        The array is rotated in the plane defined by the axes.
        Axes must be different.

    Raises
    ------
    ValueError
        If `len(axis)!=2`.
    ValueError
        If the axes are the same.
    ValueError
        If axes are out of range.

    Notes
    -----
    - ``rot90(m, k=1, axes=(1,0))`` is the reverse of ``rot90(m, k=1, axes=(0,1))``.\n
    - ``rot90(m, k=1, axes=(1,0))`` is equivalent to ``rot90(m, k=-1, axes=(0,1))``.

    May change the split axis on distributed tensors.

    Examples
    --------
    >>> m = ht.array([[1,2],[3,4]], dtype=ht.int)
    >>> m
    DNDarray([[1, 2],
              [3, 4]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.rot90(m)
    DNDarray([[2, 4],
              [1, 3]], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.rot90(m, 2)
    DNDarray([[4, 3],
              [2, 1]], dtype=ht.int32, device=cpu:0, split=None)
    >>> m = ht.arange(8).reshape((2,2,2))
    >>> ht.rot90(m, 1, (1,2))
    DNDarray([[[1, 3],
               [0, 2]],

              [[5, 7],
               [4, 6]]], dtype=ht.int32, device=cpu:0, split=None)
    """
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    if not isinstance(m, DNDarray):
        raise TypeError(f"expected m to be a ht.DNDarray, but was {type(m)}")

    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    if axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or axes[1] < -m.ndim:
        raise ValueError(f"Axes={axes} out of range for array of ndim={m.ndim}.")

    if m.split is None:
        return factories.array(
            torch.rot90(m.larray, k, axes), dtype=m.dtype, device=m.device, comm=m.comm
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


DNDarray.rot90 = lambda self, k=1, axis=(0, 1): rot90(self, k, axis)
DNDarray.rot90.__doc__ = rot90.__doc__


def shape(a: DNDarray) -> Tuple[int, ...]:
    """
    Returns the global shape of a (potentially distributed) `DNDarray` as a tuple.

    Parameters
    ----------
    a : DNDarray
        The input `DNDarray`.
    """
    # sanitize input
    if not isinstance(a, DNDarray):
        raise TypeError(f"Expected a to be a DNDarray but was {type(a)}")

    return a.gshape


def sort(a: DNDarray, axis: int = -1, descending: bool = False, out: Optional[DNDarray] = None):
    """
    Sorts the elements of `a` along the given dimension (by default in ascending order) by their value.
    The sorting is not stable which means that equal elements in the result may have a different ordering than in the
    original array.
    Sorting where `axis==a.split` needs a lot of communication between the processes of MPI.
    Returns a tuple `(values, indices)` with the sorted local results and the indices of the elements in the original data

    Parameters
    ----------
    a : DNDarray
        Input array to be sorted.
    axis : int, optional
        The dimension to sort along.
        Default is the last axis.
    descending : bool, optional
        If set to `True`, values are sorted in descending order.
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to `None`, a fresh array is allocated.

    Raises
    ------
    ValueError
        If `axis` is not consistent with the available dimensions.

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
    stride_tricks.sanitize_axis(a.shape, axis)

    if a.split is None or axis != a.split:
        # sorting is not affected by split -> we can just sort along the axis
        final_result, final_indices = torch.sort(a.larray, dim=axis, descending=descending)

    else:
        # sorting is affected by split, processes need to communicate results
        # transpose so we can work along the 0 axis
        transposed = a.larray.transpose(axis, 0)
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
        pivot_dim[0] = size * sum(x > 0 for x in counts)

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
                        sent = send_vec[idx][proc][: first + i + 1].sum().item()
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
        final_indices, dtype=types.int32, is_split=a.split, device=a.device, comm=a.comm
    )
    if out is not None:
        out.larray = final_result
        return return_indices
    else:
        tensor = factories.array(
            final_result, dtype=a.dtype, is_split=a.split, device=a.device, comm=a.comm
        )
        return tensor, return_indices


def split(x: DNDarray, indices_or_sections: Iterable, axis: int = 0) -> List[DNDarray, ...]:
    """
    Split a DNDarray into multiple sub-DNDarrays.
    Returns a list of sub-DNDarrays as copies of parts of `x`.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : int or 1-dimensional array_like (i.e. undistributed DNDarray, list or tuple)
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along axis.
        If such a split is not possible, an error is raised.
        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along axis
        the array is split.
        For example, `indices_or_sections = [2, 3]` would, for `axis = 0`, result in

        - `x[:2]`
        - `x[2:3]`
        - `x[3:]`

        If an index exceeds the dimension of the array along axis, an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
        `axis` is not allowed to equal `x.split` if `x` is distributed.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Warnings
    --------
    Though it is possible to distribute `x`, this function has nothing to do with the split
    parameter of a DNDarray.

    See Also
    --------
    :func:`dsplit`
    :func:`hsplit`
    :func:`vsplit`

    Examples
    --------
    >>> x = ht.arange(12).reshape((4,3))
    >>> ht.split(x, 2)
        [ DNDarray([[0, 1, 2],
                    [3, 4, 5]]),
          DNDarray([[ 6,  7,  8],
                    [ 9, 10, 11]])]
    >>> ht.split(x, [2, 3, 5])
        [ DNDarray([[0, 1, 2],
                    [3, 4, 5]]),
          DNDarray([[6, 7, 8]]
          DNDarray([[ 9, 10, 11]]),
          DNDarray([])]
    >>> ht.split(x, [1, 2], 1)
        [DNDarray([[0],
                [3],
                [6],
                [9]]),
        DNDarray([[ 1],
                [ 4],
                [ 7],
                [10]],
        DNDarray([[ 2],
                [ 5],
                [ 8],
                [11]])]
    """
    # sanitize x
    sanitation.sanitize_in(x)

    # sanitize axis
    if not isinstance(axis, int):
        raise TypeError(f"Expected `axis` to be an integer, but was {type(axis)}")
    if axis < 0 or axis > len(x.gshape) - 1:
        raise ValueError(
            f"Invalid input for `axis`. Valid range is between 0 and {len(x.gshape) - 1}, but was {axis}"
        )

    # sanitize indices_or_sections
    if isinstance(indices_or_sections, int):
        if x.gshape[axis] % indices_or_sections != 0:
            raise ValueError(
                f"DNDarray with shape {x.gshape} can't be divided equally into {indices_or_sections} chunks along axis {axis}"
            )
        # np to torch mapping - calculate size of resulting data chunks
        indices_or_sections_t = x.gshape[axis] // indices_or_sections

    elif isinstance(indices_or_sections, (list, tuple, DNDarray)):
        if isinstance(indices_or_sections, (list, tuple)):
            indices_or_sections = factories.array(indices_or_sections)
        if len(indices_or_sections.gshape) != 1:
            raise ValueError(
                f"Expected indices_or_sections to be 1-dimensional, but was {len(indices_or_sections.gshape) - 1}-dimensional instead."
            )
    else:
        raise TypeError(
            f"Expected `indices_or_sections` to be array_like (DNDarray, list or tuple), but was {type(indices_or_sections)}"
        )

    # start of actual algorithm

    if x.split == axis and x.is_distributed():
        if isinstance(indices_or_sections, int):
            # CASE 1 number of processes == indices_or_selections -> split already done due to distribution
            if x.comm.size == indices_or_sections:
                new_lshape = list(x.lshape)
                new_lshape[axis] = 0
                sub_arrays_t = [
                    torch.empty(new_lshape) if i != x.comm.rank else x._DNDarray__array
                    for i in range(indices_or_sections)
                ]

            # CASE 2 number of processes != indices_or_selections -> reorder (and split) chunks correctly
            else:
                # no data
                if x.lshape[axis] == 0:
                    sub_arrays_t = [torch.empty(x.lshape) for i in range(indices_or_sections)]
                else:
                    offset, local_shape, slices = x.comm.chunk(x.gshape, axis)
                    idx_frst_chunk_affctd = offset // indices_or_sections_t
                    left_data_chunk = indices_or_sections_t - (offset % indices_or_sections_t)
                    left_data_process = x.lshape[axis]

                    new_indices = torch.zeros(indices_or_sections, dtype=int)

                    if left_data_chunk >= left_data_process:
                        new_indices[idx_frst_chunk_affctd] = left_data_process
                    else:
                        new_indices[idx_frst_chunk_affctd] = left_data_chunk
                        left_data_process -= left_data_chunk
                        idx_frst_chunk_affctd += 1

                        # calculate chunks which can be filled completely
                        left_chunks_to_fill = left_data_process // indices_or_sections_t
                        new_indices[
                            idx_frst_chunk_affctd : (left_chunks_to_fill + idx_frst_chunk_affctd)
                        ] = indices_or_sections_t

                        # assign residual to following process
                        new_indices[left_chunks_to_fill + idx_frst_chunk_affctd] = (
                            left_data_process % indices_or_sections_t
                        )

                    sub_arrays_t = torch.split(x._DNDarray__array, new_indices.tolist(), axis)
        # indices or sections == DNDarray
        else:
            if indices_or_sections.split is not None:
                warnings.warn(
                    f"`indices_or_sections` might not be distributed (along axis {indices_or_sections.split}) "
                    "if `x` is not distributed.\n`indices_or_sections` will be copied with new split axis None."
                )
                indices_or_sections = resplit(indices_or_sections, None)

            offset, local_shape, slices = x.comm.chunk(x.gshape, axis)
            slice_axis = slices[axis]

            # reduce information to the (chunk) relevant
            indices_or_sections_t = indexing.where(
                indices_or_sections <= slice_axis.start, slice_axis.start, indices_or_sections
            )

            indices_or_sections_t = indexing.where(
                indices_or_sections_t >= slice_axis.stop, slice_axis.stop, indices_or_sections_t
            )

            # np to torch mapping

            # 2. add first and last value to DNDarray
            # 3. calculate the 1-st discrete difference therefore corresponding chunk sizes
            indices_or_sections_t = arithmetics.diff(
                indices_or_sections_t, prepend=slice_axis.start, append=slice_axis.stop
            )
            indices_or_sections_t = factories.array(
                indices_or_sections_t,
                dtype=types.int64,
                is_split=indices_or_sections_t.split,
                comm=indices_or_sections_t.comm,
                device=indices_or_sections_t.device,
            )

            # 4. transform the result into a list (torch requirement)
            indices_or_sections_t = indices_or_sections_t.tolist()

            sub_arrays_t = torch.split(x._DNDarray__array, indices_or_sections_t, axis)
    else:
        if isinstance(indices_or_sections, int):
            sub_arrays_t = torch.split(x._DNDarray__array, indices_or_sections_t, axis)
        else:
            if indices_or_sections.split is not None:
                warnings.warn(
                    "`indices_or_sections` might not be distributed (along axis {}) if `x` is not distributed.\n"
                    "`indices_or_sections` will be copied with new split axis None.".format(
                        indices_or_sections.split
                    )
                )
                indices_or_sections = resplit(indices_or_sections, None)

            # np to torch mapping

            # 1. replace all values out of range with gshape[axis] to generate size 0
            indices_or_sections_t = indexing.where(
                indices_or_sections <= x.gshape[axis], indices_or_sections, x.gshape[axis]
            )

            # 2. add first and last value to DNDarray
            # 3. calculate the 1-st discrete difference therefore corresponding chunk sizes
            indices_or_sections_t = arithmetics.diff(
                indices_or_sections_t, prepend=0, append=x.gshape[axis]
            )
            indices_or_sections_t = factories.array(
                indices_or_sections_t,
                dtype=types.int64,
                is_split=indices_or_sections_t.split,
                comm=indices_or_sections_t.comm,
                device=indices_or_sections_t.device,
            )

            # 4. transform the result into a list (torch requirement)
            indices_or_sections_t = indices_or_sections_t.tolist()

            sub_arrays_t = torch.split(x._DNDarray__array, indices_or_sections_t, axis)
    sub_arrays_ht = [
        factories.array(sub_DNDarray, dtype=x.dtype, is_split=x.split, device=x.device, comm=x.comm)
        for sub_DNDarray in sub_arrays_t
    ]

    for sub_DNDarray in sub_arrays_ht:
        sub_DNDarray.balance_()

    return sub_arrays_ht


def squeeze(x: DNDarray, axis: Union[int, Tuple[int, ...]] = None) -> DNDarray:
    """
    Remove single-element entries from the shape of a `DNDarray`.
    Returns the input array, but with all or a subset (indicated by `axis`) of the dimensions of length 1 removed.
    Split semantics: see Notes below.

    Parameters
    -----------
    x : DNDarray
        Input data.
    axis : None or int or Tuple[int,...], optional
           Selects a subset of the single-element entries in the shape.
           If axis is `None`, all single-element entries will be removed from the shape.

    Raises
    ------
    `ValueError`, if an axis is selected with shape entry greater than one.

    Notes
    -----
    Split semantics: a distributed DNDarray will keep its original split dimension after "squeezing",
    which, depending on the squeeze axis, may result in a lower numerical `split` value (see Examples).

    Examples
    ---------
    >>> import heat as ht
    >>> a = ht.random.randn(1,3,1,5)
    >>> a
    DNDarray([[[[-0.2604,  1.3512,  0.1175,  0.4197,  1.3590]],
               [[-0.2777, -1.1029,  0.0697, -1.3074, -1.1931]],
               [[-0.4512, -1.2348, -1.1479, -0.0242,  0.4050]]]], dtype=ht.float32, device=cpu:0, split=None)
    >>> a.shape
    (1, 3, 1, 5)
    >>> ht.squeeze(a).shape
    (3, 5)
    >>> ht.squeeze(a)
    DNDarray([[-0.2604,  1.3512,  0.1175,  0.4197,  1.3590],
              [-0.2777, -1.1029,  0.0697, -1.3074, -1.1931],
              [-0.4512, -1.2348, -1.1479, -0.0242,  0.4050]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.squeeze(a,axis=0).shape
    (3, 1, 5)
    >>> ht.squeeze(a,axis=-2).shape
    (1, 3, 5)
    >>> ht.squeeze(a,axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: Dimension along axis 1 is not 1 for shape (1, 3, 1, 5)
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
    sanitation.sanitize_in(x)
    # Sanitize axis
    axis = stride_tricks.sanitize_axis(x.shape, axis)
    if axis is not None:
        if isinstance(axis, int):
            dim_is_one = x.shape[axis] == 1
            axis = (axis,)
        elif isinstance(axis, tuple):
            dim_is_one = bool(torch.tensor(list(x.shape[dim] == 1 for dim in axis)).all())
        if not dim_is_one:
            raise ValueError(f"Dimension along axis {axis} is not 1 for shape {x.shape}")

    if axis is None:
        axis = tuple(i for i, dim in enumerate(x.shape) if dim == 1)

    if x.split is not None and x.split in axis:
        # split dimension is about to disappear, set split to None
        x.resplit_(axis=None)

    out_lshape = tuple(x.lshape[dim] for dim in range(x.ndim) if dim not in axis)
    out_gshape = tuple(x.gshape[dim] for dim in range(x.ndim) if dim not in axis)
    x_lsqueezed = x.larray.reshape(out_lshape)

    # Calculate new split axis according to squeezed shape
    if x.split is not None:
        split = x.split - len([dim for dim in axis if dim < x.split])
    else:
        split = None

    return DNDarray(
        x_lsqueezed,
        out_gshape,
        x.dtype,
        split=split,
        device=x.device,
        comm=x.comm,
        balanced=x.balanced,
    )


DNDarray.squeeze: Callable[
    [DNDarray, Union[int, Tuple[int, ...]]], DNDarray
] = lambda self, axis=None: squeeze(self, axis)
DNDarray.squeeze.__doc__ = squeeze.__doc__


def stack(
    arrays: Sequence[DNDarray, ...], axis: int = 0, out: Optional[DNDarray] = None
) -> DNDarray:
    """
    Join a sequence of `DNDarray`s along a new axis.
    The `axis` parameter specifies the index of the new axis in the dimensions of the result.
    For example, if `axis=0`, the arrays will be stacked along the first dimension; if `axis=-1`,
    they will be stacked along the last dimension. See Notes below for split semantics.

    Parameters
    ----------
    arrays : Sequence[DNDarrays, ...]
        Each DNDarray must have the same shape, must be split along the same axis, and must be balanced.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : DNDarray, optional
        If provided, the destination to place the result. The shape and split axis must be correct, matching
        that of what stack would have returned if no out argument were specified (see Notes below).

    Raises
    ------
    TypeError
        If arrays in sequence are not `DNDarray`s, or if their `dtype` attribute does not match.
    ValueError
        If `arrays` contains less than 2 `DNDarray`s.
    ValueError
        If the `DNDarray`s are of different shapes, or if they are split along different axes (`split` attribute).
    RuntimeError
        If the `DNDarrays` reside on different devices.

    Notes
    -----
    Split semantics: :func:`stack` requires that all arrays in the sequence be split along the same dimension.
    After stacking, the data are still distributed along the original dimension, however a new dimension has been added at `axis`,
    therefore:

    - if :math:`axis <= split`, output will be distributed along :math:`split+1`

    - if :math:`axis > split`, output will be distributed along `split`

    See Also
    --------
    :func:`column_stack`
    :func:`concatenate`
    :func:`hstack`
    :func:`row_stack`
    :func:`vstack`

    Examples
    --------
    >>> a = ht.arange(20).reshape((4, 5))
    >>> b = ht.arange(20, 40).reshape((4, 5))
    >>> ht.stack((a,b), axis=0).larray
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
    >>> ht.stack((a,b), axis=-1).larray
    [0/2] tensor([[[ 0, 20],
    [0/2]          [ 1, 21],
    [0/2]          [ 2, 22],
    [0/2]          [ 3, 23],
    [0/2]          [ 4, 24]],
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
    arrays = sanitation.sanitize_sequence(arrays)
    if len(arrays) < 2:
        raise ValueError("stack expects a sequence of at least 2 DNDarrays")

    target = arrays[0]
    try:
        arrays = sanitation.sanitize_distribution(
            *arrays, target=target
        )  # also checks target again
    except NotImplementedError as e:  # transform split axis error to ValueError
        raise ValueError(e) from e

    # extract torch tensors
    t_arrays = [array.larray for array in arrays]

    # output shape and split
    axis = stride_tricks.sanitize_axis(target.gshape + (len(arrays),), axis)
    stacked_shape = target.gshape[:axis] + (len(arrays),) + target.gshape[axis:]
    if target.split is not None:
        stacked_split = target.split + 1 if axis <= target.split else target.split
    else:
        stacked_split = None

    # stack locally
    try:
        t_stacked = torch.stack(t_arrays, dim=axis)
        result_dtype = types.canonical_heat_type(t_stacked.dtype)
    except Exception as e:
        if "size" in e.args[0] or "shape" in e.args[0]:
            raise ValueError(e) from e
        raise e

    # return stacked DNDarrays
    if out is not None:
        sanitation.sanitize_out(out, stacked_shape, stacked_split, target.device)
        out.larray = t_stacked.type(out.larray.dtype)
        return out

    stacked = DNDarray(
        t_stacked,
        gshape=stacked_shape,
        dtype=result_dtype,
        split=stacked_split,
        device=target.device,
        comm=target.comm,
        balanced=target.balanced,
    )
    return stacked


def swapaxes(x: DNDarray, axis1: int, axis2: int) -> DNDarray:
    """
    Interchanges two axes of an array.

    Parameters
    ----------
    x : DNDarray
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    See Also
    --------
    :func:`~heat.core.linalg.basics.transpose`
        Permute the dimensions of an array.

    Examples
    --------
    >>> x = ht.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> ht.swapaxes(x, 0, 1)
    DNDarray([[[0, 1],
               [4, 5]],
              [[2, 3],
               [6, 7]]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.swapaxes(x, 0, 2)
    DNDarray([[[0, 4],
               [2, 6]],
              [[1, 5],
               [3, 7]]], dtype=ht.int64, device=cpu:0, split=None)
    """
    axes = list(range(x.ndim))
    try:
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    except TypeError:
        raise TypeError(
            f"'axis1' and 'axis2' must be of type int, found {type(axis1)} and {type(axis2)}"
        )

    return linalg.transpose(x, axes)


DNDarray.swapaxes = lambda self, axis1, axis2: swapaxes(self, axis1, axis2)
DNDarray.swapaxes.__doc__ = swapaxes.__doc__


def unique(
    a: DNDarray, sorted: bool = False, return_inverse: bool = False, axis: int = None
) -> Tuple[DNDarray, torch.tensor]:
    """
    Finds and returns the unique elements of a `DNDarray`.
    If return_inverse is `True`, the second tensor will hold the list of inverse indices
    If distributed, it is most efficient if `axis!=a.split`.

    Parameters
    ----------
    a : DNDarray
        Input array.
    sorted : bool, optional
        Whether the found elements should be sorted before returning as output.
        Warning: sorted is not working if `axis!=None and axis!=a.split`
    return_inverse : bool, optional
        Whether to also return the indices for where elements in the original input ended up in the returned
        unique list.
    axis : int, optional
        Axis along which unique elements should be found. Default to `None`, which will return a one dimensional list of
        unique values.

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
            a.larray, sorted=sorted, return_inverse=return_inverse, dim=axis
        )
        if isinstance(torch_output, tuple):
            heat_output = tuple(
                factories.array(i, dtype=a.dtype, split=None, device=a.device, comm=a.comm)
                for i in torch_output
            )
        else:
            heat_output = factories.array(
                torch_output, dtype=a.dtype, split=None, device=a.device, comm=a.comm
            )
        return heat_output

    local_data = a.larray
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
        gres_buf = torch.empty(output_dim, dtype=a.dtype.torch_type(), device=a.device.torch_device)
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


DNDarray.unique: Callable[
    [DNDarray, bool, bool, int], Tuple[DNDarray, torch.tensor]
] = lambda self, sorted=False, return_inverse=False, axis=None: unique(
    self, sorted, return_inverse, axis
)
DNDarray.unique.__doc__ = unique.__doc__


def vsplit(x: DNDarray, indices_or_sections: Iterable) -> List[DNDarray, ...]:
    """
    Split array into multiple sub-DNDNarrays along the 1st axis (vertically/row-wise).
    Returns a list of sub-DNDarrays as copies of parts of ``x``.

    Parameters
    ----------
    x : DNDarray
        DNDArray to be divided into sub-DNDarrays.
    indices_or_sections : Iterable
        If `indices_or_sections` is an integer, N, the DNDarray will be divided into N equal DNDarrays along the 1st axis.\n
        If such a split is not possible, an error is raised.\n
        If `indices_or_sections` is a 1-D DNDarray of sorted integers, the entries indicate where along the 1st axis the array is split.\n
        If an index exceeds the dimension of the array along the 1st axis, an empty sub-DNDarray is returned correspondingly.\n

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as integer, but a split does not result in equal division.

    Notes
    -----
    Please refer to the split documentation. :func:`hsplit` is equivalent to split with `axis=0`,
    the array is always split along the first axis regardless of the array dimension.

    See Also
    --------
    :func:`split`
    :func:`dsplit`
    :func:`hsplit`

    Examples
    --------
    >>> x = ht.arange(24).reshape((4, 3, 2))
    >>> ht.vsplit(x, 2)
        [DNDarray([[[ 0,  1],
                   [ 2,  3],
                   [ 4,  5]],
                  [[ 6,  7],
                   [ 8,  9],
                   [10, 11]]]),
        DNDarray([[[12, 13],
                   [14, 15],
                   [16, 17]],
                  [[18, 19],
                   [20, 21],
                   [22, 23]]])]
        >>> ht.vsplit(x, [1, 3])
        [DNDarray([[[0, 1],
                   [2, 3],
                   [4, 5]]]),
        DNDarray([[[ 6,  7],
                   [ 8,  9],
                   [10, 11]],
                  [[12, 13],
                   [14, 15],
                   [16, 17]]]),
        DNDarray([[[18, 19],
                   [20, 21],
                   [22, 23]]])]
    """
    return split(x, indices_or_sections, 0)


def resplit(arr: DNDarray, axis: int = None) -> DNDarray:
    """
    Out-of-place redistribution of the content of the `DNDarray`. Allows to "unsplit" (i.e. gather) all values from all
    nodes,  as well as to define a new axis along which the array is split without changes to the values.

    Parameters
    ----------
    arr : DNDarray
        The array from which to resplit
    axis : int or None
        The new split axis, `None` denotes gathering, an int will set the new split axis

    Warning
    ----------
    This operation might involve a significant communication overhead. Use it sparingly and preferably for
    small arrays.

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
    if not arr.is_distributed():
        return factories.array(arr.larray, split=axis, device=arr.device, comm=arr.comm, copy=True)

    if axis is None:
        # new_arr = arr.copy()
        gathered = torch.empty(
            arr.shape, dtype=arr.dtype.torch_type(), device=arr.device.torch_device
        )
        counts, displs = arr.counts_displs()
        arr.comm.Allgatherv(arr.larray, (gathered, counts, displs), recv_axis=arr.split)
        new_arr = factories.array(
            gathered, is_split=axis, device=arr.device, comm=arr.comm, dtype=arr.dtype
        )
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
            elif spr == rpr == rank:
                new_tiles[key] = to_send.clone()
            elif rank == rpr:
                buf = torch.zeros_like(new_tiles[key])
                rcv_waits[key] = [arr.comm.Irecv(buf=buf, source=spr, tag=spr), buf]
    for w in waits:
        w.Wait()
    for k in rcv_waits.keys():
        rcv_waits[k][0].Wait()
        new_tiles[k] = rcv_waits[k][1]

    return new_arr


DNDarray.resplit: Callable[[DNDarray, Optional[int]], DNDarray] = lambda self, axis=None: resplit(
    self, axis
)
DNDarray.resplit.__doc__ = resplit.__doc__


def row_stack(arrays: Sequence[DNDarray, ...]) -> DNDarray:
    """
    Stack 1-D or 2-D `DNDarray`s as rows into a 2-D `DNDarray`.
    If the input arrays are 1-D, they will be stacked as rows. If they are 2-D,
    they will be concatenated along the first axis.

    Parameters
    ----------
    arrays : Sequence[DNDarrays, ...]
        Sequence of `DNDarray`s.

    Raises
    ------
    ValueError
        If arrays have more than 2 dimensions

    Notes
    -----
    All ``DNDarray``s in the sequence must have the same number of columns.
    All ``DNDarray``s must be split along the same axis!

    See Also
    --------
    :func:`column_stack`
    :func:`concatenate`
    :func:`hstack`
    :func:`stack`
    :func:`vstack`

    Examples
    --------
    >>> # 1-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.row_stack((a, b)).larray
    tensor([[1, 2, 3],
            [2, 3, 4]])
    >>> # 1-D and 2-D tensors
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([[2, 3, 4], [5, 6, 7]])
    >>> c = ht.array([[7, 8, 9], [10, 11, 12]])
    >>> ht.row_stack((a, b, c)).larray
    tensor([[ 1,  2,  3],
            [ 2,  3,  4],
            [ 5,  6,  7],
            [ 7,  8,  9],
            [10, 11, 12]])
    >>> # distributed DNDarrays, 3 processes
    >>> a = ht.arange(10, split=0).reshape((2, 5))
    >>> b = ht.arange(5, 20, split=0).reshape((3, 5))
    >>> c = ht.arange(20, 40, split=0).reshape((4, 5))
    >>> ht.row_stack((a, b, c)).larray
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
    >>> ht.row_stack((a, b)).larray
    [0/2] tensor([[0, 1, 2, 3, 4],
    [0/2]         [5, 6, 7, 8, 9]])
    [1/2] tensor([[10, 11, 12, 13, 14]])
    [2/2] tensor([[15, 16, 17, 18, 19]])
    """
    arr_dims = [array.ndim for array in arrays]
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


def vstack(arrays: Sequence[DNDarray, ...]) -> DNDarray:
    """
    Stack arrays in sequence vertically (row wise).
    This is equivalent to concatenation along the first axis.
    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The :func:`concatenate` function provides more general
    stacking operations.

    Parameters
    ----------
    arrays : Sequence[DNDarray,...]
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Notes
    -------
    The split axis will be switched to 1 in the case that both elements are 1D and split=0

    See Also
    --------
    :func:`concatenate`
    :func:`stack`
    :func:`hstack`
    :func:`column_stack`
    :func:`row_stack`


    Examples
    --------
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.array([2, 3, 4])
    >>> ht.vstack((a,b)).larray
    [0/1] tensor([[1, 2, 3],
    [0/1]         [2, 3, 4]])
    [1/1] tensor([[1, 2, 3],
    [1/1]         [2, 3, 4]])
    >>> a = ht.array([1, 2, 3], split=0)
    >>> b = ht.array([2, 3, 4], split=0)
    >>> ht.vstack((a,b)).larray
    [0/1] tensor([[1, 2],
    [0/1]         [2, 3]])
    [1/1] tensor([[3],
    [1/1]         [4]])
    >>> a = ht.array([[1], [2], [3]], split=0)
    >>> b = ht.array([[2], [3], [4]], split=0)
    >>> ht.vstack((a,b)).larray
    [0] tensor([[1],
    [0]         [2],
    [0]         [3]])
    [1] tensor([[2],
    [1]         [3],
    [1]         [4]])
    """
    arrays = list(arrays)
    for cn, arr in enumerate(arrays):
        if len(arr.gshape) == 1:
            arrays[cn] = arr.expand_dims(0).resplit_(arr.split)

    return concatenate(arrays, axis=0)


def tile(x: DNDarray, reps: Sequence[int, ...]) -> DNDarray:
    """
    Construct a new DNDarray by repeating 'x' the number of times given by 'reps'.

    If 'reps' has length 'd', the result will have 'max(d, x.ndim)' dimensions:

    - if 'x.ndim < d', 'x' is promoted to be d-dimensional by prepending new axes.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3)
    for 3-D replication (if this is not the desired behavior, promote 'x' to d-dimensions
    manually before calling this function);

    - if 'x.ndim > d', 'reps' will replicate the last 'd' dimensions of 'x', i.e., if
    'x.shape' is (2, 3, 4, 5), a 'reps' of (2, 2) will be expanded to (1, 1, 2, 2).

    Parameters
    ----------
    x : DNDarray
        Input

    reps : Sequence[ints,...]
        Repetitions

    Returns
    -------
    tiled : DNDarray
            Split semantics: if `x` is distributed, the tiled data will be distributed along the
            same dimension. Note that nominally `tiled.split != x.split` in the case where
            `len(reps) > x.ndim`.  See example below.

    Examples
    --------
    >>> x = ht.arange(12).reshape((4,3)).resplit_(0)
    >>> x
    DNDarray([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]], dtype=ht.int32, device=cpu:0, split=0)
    >>> reps = (1, 2, 2)
    >>> tiled = ht.tile(x, reps)
    >>> tiled
    DNDarray([[[ 0,  1,  2,  0,  1,  2],
               [ 3,  4,  5,  3,  4,  5],
               [ 6,  7,  8,  6,  7,  8],
               [ 9, 10, 11,  9, 10, 11],
               [ 0,  1,  2,  0,  1,  2],
               [ 3,  4,  5,  3,  4,  5],
               [ 6,  7,  8,  6,  7,  8],
               [ 9, 10, 11,  9, 10, 11]]], dtype=ht.int32, device=cpu:0, split=1)
    """
    # x can be DNDarray or scalar
    try:
        _ = x.larray
    except AttributeError:
        try:
            _ = x.shape
            raise TypeError(f"Input can be a DNDarray or a scalar, is {type(x)}")
        except AttributeError:
            x = factories.array(x).reshape(1)

    x_proxy = x.__torch_proxy__()

    # torch-proof args/kwargs:
    # torch `reps`: int or sequence of ints; numpy `reps`: can be array-like
    try:
        _ = x_proxy.repeat(reps)
    except TypeError:
        # `reps` is array-like or contains non-int elements
        try:
            reps = resplit(reps, None).tolist()
        except AttributeError:
            try:
                reps = reps.tolist()
            except AttributeError:
                try:
                    _ = x_proxy.repeat(reps)
                except TypeError:
                    raise TypeError(
                        f"reps must be a sequence of ints, got {[type(i) for i in reps]}"
                    )
                except RuntimeError:
                    pass
    except RuntimeError:
        pass

    try:
        reps = list(reps)
    except TypeError:
        # scalar to list
        reps = [reps]

    # torch reps vs. numpy reps: dimensions
    if len(reps) != x.ndim:
        added_dims = abs(len(reps) - x.ndim)
        if len(reps) > x.ndim:
            new_shape = added_dims * (1,) + x.gshape
            new_split = None if x.split is None else x.split + added_dims
            x = x.reshape(new_shape, new_split=new_split)
        else:
            reps = added_dims * [1] + reps

    out_gshape = tuple(x_proxy.repeat(reps).shape)

    if not x.is_distributed() or reps[x.split] == 1:
        # no repeats along the split axis: local operation
        t_tiled = x.larray.repeat(reps)
        out_gshape = tuple(x_proxy.repeat(reps).shape)
        return DNDarray(
            t_tiled,
            out_gshape,
            dtype=x.dtype,
            split=x.split,
            device=x.device,
            comm=x.comm,
            balanced=x.balanced,
        )
    # repeats along the split axis, work along dim 0
    size = x.comm.Get_size()
    rank = x.comm.Get_rank()
    trans_axes = list(range(x.ndim))
    if x.split != 0:
        trans_axes[0], trans_axes[x.split] = x.split, 0
        reps[0], reps[x.split] = reps[x.split], reps[0]
        x = linalg.transpose(x, trans_axes)
        x_proxy = x.__torch_proxy__()
        out_gshape = tuple(x_proxy.repeat(reps).shape)

    local_x = x.larray

    # allocate tiled DNDarray, at first tiled along split axis only
    split_reps = [rep if i == x.split else 1 for i, rep in enumerate(reps)]
    split_tiled_shape = tuple(x_proxy.repeat(split_reps).shape)
    tiled = factories.empty(split_tiled_shape, dtype=x.dtype, split=x.split, comm=x.comm)
    # collect slicing information from all processes.
    slices_map = []
    for array in [x, tiled]:
        counts, displs = array.counts_displs()
        t_slices_starts = torch.tensor(displs, device=local_x.device)
        t_slices_ends = t_slices_starts + torch.tensor(counts, device=local_x.device)
        slices_map.append([t_slices_starts, t_slices_ends])

    t_slices_x, t_slices_tiled = slices_map

    # keep track of repetitions:
    # local_x_starts.shape, local_x_ends.shape changing from (size,) to (reps[split], size)
    reps_indices = [x.gshape[x.split] * rep for rep in (range(reps[x.split]))]
    t_reps_indices = torch.tensor(reps_indices, dtype=torch.int32, device=local_x.device).reshape(
        len(reps_indices), 1
    )
    for i, t in enumerate(t_slices_x):
        t = t.repeat((reps[x.split], 1))
        t += t_reps_indices
        t_slices_x[i] = t

    # distribution logic on current rank:
    distr_map = []
    slices_map = []
    for i in range(2):
        if i == 0:
            # send logic for x slices on rank
            local_x_starts = t_slices_x[0][:, rank].reshape(reps[x.split], 1)
            local_x_ends = t_slices_x[1][:, rank].reshape(reps[x.split], 1)
            t_tiled_starts, t_tiled_ends = t_slices_tiled
        else:
            # recv logic for tiled slices on rank
            local_x_starts, local_x_ends = t_slices_x
            t_tiled_starts = t_slices_tiled[0][rank]
            t_tiled_ends = t_slices_tiled[1][rank]
        t_max_starts = torch.max(local_x_starts, t_tiled_starts)
        t_min_ends = torch.min(local_x_ends, t_tiled_ends)
        coords = torch.where(t_min_ends - t_max_starts > 0)
        # remove repeat offset from slices if sending
        if i == 0:
            t_max_starts -= t_reps_indices
            t_min_ends -= t_reps_indices
        starts = t_max_starts[coords].unsqueeze_(0)
        ends = t_min_ends[coords].unsqueeze_(0)
        slices_map.append(torch.cat((starts, ends), dim=0))
        distr_map.append(coords)

    # bookkeeping in preparation for Alltoallv
    send_map, recv_map = distr_map
    send_rep, send_to_ranks = send_map
    recv_rep, recv_from_ranks = recv_map
    send_slices, recv_slices = slices_map

    # do not assume that `x` is balanced
    _, displs = x.counts_displs()
    offset_x = displs[rank]
    # impose load-balance on output
    offset_tiled, _, _ = tiled.comm.chunk(tiled.gshape, tiled.split)
    t_tiled = tiled.larray

    active_send_counts = send_slices.clone()
    active_send_counts[0] *= -1
    active_send_counts = active_send_counts.sum(0)
    active_recv_counts = recv_slices.clone()
    active_recv_counts[0] *= -1
    active_recv_counts = active_recv_counts.sum(0)
    send_slices -= offset_x
    recv_slices -= offset_tiled
    recv_buf = t_tiled.clone()
    # we need as many Alltoallv calls as repeats along the split axis
    for rep in range(reps[x.split]):
        # send_data, send_counts, send_displs on rank
        all_send_counts = [0] * size
        all_send_displs = [0] * size
        send_this_rep = torch.where(send_rep == rep)[0].tolist()
        dest_this_rep = send_to_ranks[send_this_rep].tolist()
        for i, j in zip(send_this_rep, dest_this_rep):
            all_send_counts[j] = active_send_counts[i].item()
            all_send_displs[j] = send_slices[0][i].item()
        local_send_slice = [slice(None)] * x.ndim
        local_send_slice[x.split] = slice(
            all_send_displs[0], all_send_displs[0] + sum(all_send_counts)
        )
        send_buf = local_x[local_send_slice].clone()

        # recv_data, recv_counts, recv_displs on rank
        all_recv_counts = [0] * size
        all_recv_displs = [0] * size
        recv_this_rep = torch.where(recv_rep == rep)[0].tolist()
        orig_this_rep = recv_from_ranks[recv_this_rep].tolist()
        for i, j in zip(recv_this_rep, orig_this_rep):
            all_recv_counts[j] = active_recv_counts[i].item()
            all_recv_displs[j] = recv_slices[0][i].item()
        local_recv_slice = [slice(None)] * x.ndim
        local_recv_slice[x.split] = slice(
            all_recv_displs[0], all_recv_displs[0] + sum(all_recv_counts)
        )
        x.comm.Alltoallv(
            (send_buf, all_send_counts, all_send_displs),
            (recv_buf, all_recv_counts, all_recv_displs),
        )
        t_tiled[local_recv_slice] = recv_buf[local_recv_slice]

    # finally tile along non-split axes if needed
    reps[x.split] = 1
    tiled = DNDarray(
        t_tiled.repeat(reps),
        out_gshape,
        dtype=x.dtype,
        split=x.split,
        device=x.device,
        comm=x.comm,
        balanced=True,
    )
    if trans_axes != list(range(x.ndim)):
        # transpose back to original shape
        x = linalg.transpose(x, trans_axes)
        tiled = linalg.transpose(tiled, trans_axes)

    return tiled


def topk(
    a: DNDarray,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[Tuple[DNDarray, DNDarray]] = None,
) -> Tuple[DNDarray, DNDarray]:
    """
    Returns the :math:`k` highest entries in the array.
    (Not Stable for split arrays)

    Parameters
    -----------
    a: DNDarray
        Input data
    k: int
        Desired number of output items
    dim: int, optional
        Dimension along which to sort, per default the last dimension
    largest: bool, optional
        If `True`, return the :math:`k` largest items, otherwise return the :math:`k` smallest items
    sorted: bool, optional
        Whether to sort the output (descending if `largest` is `True`, else ascending)
    out: Tuple[DNDarray, ...], optional
        output buffer

    Examples
    --------
    >>> a = ht.array([1, 2, 3])
    >>> ht.topk(a,2)
    (DNDarray([3, 2], dtype=ht.int64, device=cpu:0, split=None), DNDarray([2, 1], dtype=ht.int64, device=cpu:0, split=None))
    >>> a = ht.array([[1,2,3],[1,2,3]])
    >>> ht.topk(a,2,dim=1)
    (DNDarray([[3, 2],
               [3, 2]], dtype=ht.int64, device=cpu:0, split=None),
     DNDarray([[2, 1],
               [2, 1]], dtype=ht.int64, device=cpu:0, split=None))
    >>> a = ht.array([[1,2,3],[1,2,3]], split=1)
    >>> ht.topk(a,2,dim=1)
    (DNDarray([[3, 2],
               [3, 2]], dtype=ht.int64, device=cpu:0, split=1),
     DNDarray([[2, 1],
               [2, 1]], dtype=ht.int64, device=cpu:0, split=1))
    """
    if out is not None:
        if out[0].dtype != a.dtype:
            raise RuntimeError(
                "dtypes of 'out[0]' and 'a' do not match, found {} != {}".format(
                    out[0].dtype, a.dtype
                )
            )
        if out[1].dtype != types.int64:
            raise RuntimeError("dtype of 'out[1]' is not ht.int64, found {}".format(out[1].dtype))

    dim = stride_tricks.sanitize_axis(a.gshape, dim)

    neutral_value = sanitation.sanitize_infinity(a)
    if largest:
        neutral_value = -neutral_value

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
            indices += torch.tensor(
                offset * a.comm.rank, dtype=indices.dtype, device=indices.device
            )

        local_shape = list(result.shape)
        local_shape_len = len(shape)

        metadata = torch.tensor(
            [k, dim, largest, sorted, local_shape_len, *local_shape], device=indices.device
        )
        send_buffer = torch.cat(
            (metadata.double(), result.double().flatten(), indices.flatten().double())
        )

        return send_buffer

    gres = _operations.__reduce_op(
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
    local_result = gres.larray
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
        gindices, dtype=types.int64, device=a.device, comm=a.comm, split=split, is_split=is_split
    )

    if out is not None:
        if out[0].shape != final_array.shape or out[1].shape != final_indices.shape:
            raise ValueError(
                "Expecting output buffer tuple of shape ({}, {}), got ({}, {})".format(
                    gres.shape, gindices.shape, out[0].shape, out[1].shape
                )
            )
        out[0].larray.storage().copy_(final_array.larray.storage())
        out[1].larray.storage().copy_(final_indices.larray.storage())

        out[0]._DNDarray__dtype = a.dtype
        out[1]._DNDarray__dtype = types.int64

    return final_array, final_indices


def mpi_topk(a, b, mpi_type):
    """
    MPI function for distributed :func:`topk`
    """
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

    # reconstruct the flattened data by shape
    a_values = a_values.reshape(shape_a)
    a_indices = a_indices.reshape(shape_a)
    b_values = b_values.reshape(shape_b)
    b_indices = b_indices.reshape(shape_b)

    # concatenate the data to actually run topk on
    values = torch.cat((a_values, b_values), dim=dim)
    indices = torch.cat((a_indices, b_indices), dim=dim)

    result, k_indices = torch.topk(values, k, dim=dim, largest=largest, sorted=sorted)
    indices = torch.gather(indices, dim, k_indices)

    metadata = a_parsed[0 : len_shape_a + 5]
    final_result = torch.cat((metadata, result.double().flatten(), indices.double().flatten()))

    b_parsed.copy_(final_result)


MPI_TOPK = MPI.Op.Create(mpi_topk, commute=True)
