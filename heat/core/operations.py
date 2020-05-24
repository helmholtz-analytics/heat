import builtins
import numpy as np
import torch
import warnings

from .communication import MPI, MPI_WORLD
from . import factories
from . import stride_tricks
from . import dndarray
from . import types

__all__ = []
__BOOLEAN_OPS = [MPI.LAND, MPI.LOR, MPI.BAND, MPI.BOR]


def __binary_op(operation, t1, t2):
    """
    Generic wrapper for element-wise binary operations of two operands (either can be tensor or scalar).
    Takes the operation function and the two operands involved in the operation as arguments.

    Parameters
    ----------
    operation : function
        The operation to be performed. Function that performs operation elements-wise on the involved tensors,
        e.g. add values from other to self

    t1: dndarray or scalar
        The first operand involved in the operation,

    t2: dndarray or scalar
        The second operand involved in the operation,

    Returns
    -------
    result: ht.DNDarray
        A DNDarray containing the results of element-wise operation.
    """
    if np.isscalar(t1):
        try:
            t1 = factories.array([t1])
        except (ValueError, TypeError):
            raise TypeError("Data type not supported, input was {}".format(type(t1)))

        if np.isscalar(t2):
            try:
                t2 = factories.array([t2])
            except (ValueError, TypeError):
                raise TypeError(
                    "Only numeric scalars are supported, but input was {}".format(type(t2))
                )
            output_shape = (1,)
            output_split = None
            output_device = None
            output_comm = MPI_WORLD
        elif isinstance(t2, dndarray.DNDarray):
            t1.gpu() if t2.device.device_type == "gpu" else t1.cpu()

            output_shape = t2.shape
            output_split = t2.split
            output_device = t2.device
            output_comm = t2.comm
        else:
            raise TypeError(
                "Only tensors and numeric scalars are supported, but input was {}".format(type(t2))
            )

        if t1.dtype != t2.dtype:
            t1 = t1.astype(t2.dtype)

    elif isinstance(t1, dndarray.DNDarray):
        if np.isscalar(t2):
            try:
                t2 = factories.array([t2], device=t1.device)
                output_shape = t1.shape
                output_split = t1.split
                output_device = t1.device
                output_comm = t1.comm
            except (ValueError, TypeError):
                raise TypeError("Data type not supported, input was {}".format(type(t2)))

        elif isinstance(t2, dndarray.DNDarray):
            if t1.split is None:
                t1 = factories.array(
                    t1,
                    split=t2.split,
                    copy=False,
                    comm=t1.comm,
                    device=t1.device,
                    ndmin=-t2.numdims,
                )
            elif t2.split is None:
                t2 = factories.array(
                    t2,
                    split=t1.split,
                    copy=False,
                    comm=t2.comm,
                    device=t2.device,
                    ndmin=-t1.numdims,
                )
            elif t1.split != t2.split:
                # It is NOT possible to perform binary operations on tensors with different splits, e.g. split=0
                # and split=1
                raise NotImplementedError("Not implemented for other splittings")

            output_shape = stride_tricks.broadcast_shape(t1.shape, t2.shape)
            output_split = t1.split
            output_device = t1.device
            output_comm = t1.comm

            # ToDo: Fine tuning in case of comm.size>t1.shape[t1.split]. Send torch tensors only to ranks, that will hold data.
            if t1.split is not None:
                if t1.shape[t1.split] == 1 and t1.comm.is_distributed():
                    warnings.warn(
                        "Broadcasting requires transferring data of first operator between MPI ranks!"
                    )
                    if t1.comm.rank > 0:
                        t1._DNDarray__array = torch.zeros(
                            t1.shape, dtype=t1.dtype.torch_type(), device=t1.device.torch_device
                        )
                    t1.comm.Bcast(t1)

            if t2.split is not None:
                if t2.shape[t2.split] == 1 and t2.comm.is_distributed():
                    warnings.warn(
                        "Broadcasting requires transferring data of second operator between MPI ranks!"
                    )
                    if t2.comm.rank > 0:
                        t2._DNDarray__array = torch.zeros(
                            t2.shape, dtype=t2.dtype.torch_type(), device=t2.device.torch_device
                        )
                    t2.comm.Bcast(t2)

        else:
            raise TypeError(
                "Only tensors and numeric scalars are supported, but input was {}".format(type(t2))
            )

        if t2.dtype != t1.dtype:
            t2 = t2.astype(t1.dtype)

    else:
        raise NotImplementedError("Not implemented for non scalar")

    promoted_type = types.promote_types(t1.dtype, t2.dtype).torch_type()
    if t1.split is not None:
        if len(t1.lshape) > t1.split and t1.lshape[t1.split] == 0:
            result = t1._DNDarray__array.type(promoted_type)
        else:
            result = operation(
                t1._DNDarray__array.type(promoted_type), t2._DNDarray__array.type(promoted_type)
            )
    elif t2.split is not None:

        if len(t2.lshape) > t2.split and t2.lshape[t2.split] == 0:
            result = t2._DNDarray__array.type(promoted_type)
        else:
            result = operation(
                t1._DNDarray__array.type(promoted_type), t2._DNDarray__array.type(promoted_type)
            )
    else:
        result = operation(
            t1._DNDarray__array.type(promoted_type), t2._DNDarray__array.type(promoted_type)
        )

    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)

    return dndarray.DNDarray(
        result, output_shape, types.heat_type_of(result), output_split, output_device, output_comm
    )


def __cum_op(x, partial_op, exscan_op, final_op, neutral, axis, dtype, out):
    """
    Generic wrapper for cumulative operations, i.e. cumsum(), cumprod(). Performs a three-stage cumulative operation. First, a partial
    cumulative operation is performed node-local that is combined into a global cumulative result via an MPI_Op and a final local
    reduction add or mul operation.

    Parameters
    ----------
    x : ht.DNDarray
        The heat DNDarray on which to perform the cumulative operation
    partial_op: function
        The function performing a partial cumulative operation on the process-local data portion, e.g. cumsum().
    exscan_op: mpi4py.MPI.Op
        The MPI operator for performing the exscan based on the results returned by the partial_op function.
    final_op: function
        The local operation for the final result, e.g. add() for cumsum().
    neutral: scalar
        Neutral element for the cumulative operation, i.e. an element that does not change the reductions operations
        result.
    axis: int
        The axis direction of the cumulative operation
    dtype: ht.type
        The type of the result tensor.
    out: ht.DNDarray
        The explicitly returned output tensor.

    Returns
    -------
    result: ht.DNDarray
        A DNDarray containing the result of the reduction operation

    Raises
    ------
    TypeError
        If the input or optional output parameter are not of type ht.DNDarray
    ValueError
        If the shape of the optional output parameters does not match the shape of the input
    NotImplementedError
        Numpy's behaviour of axis is None is not supported as of now
    RuntimeError
        If the split or device parameters do not match the parameters of the input
    """
    # perform sanitation
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("expected x to be a ht.DNDarray, but was {}".format(type(x)))
    if out is not None and not isinstance(out, dndarray.DNDarray):
        raise TypeError("expected out to be None or an ht.DNDarray, but was {}".format(type(out)))

    if axis is None:
        raise NotImplementedError("axis = None is not supported")
    axis = stride_tricks.sanitize_axis(x.shape, axis)

    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    if out is not None:
        if out.shape != x.shape:
            raise ValueError("out and a have different shapes {} != {}".format(out.shape, x.shape))
        if out.split != x.split:
            raise RuntimeError(
                "out and a have different splits {} != {}".format(out.split, x.split)
            )
        if out.device != x.device:
            raise RuntimeError(
                "out and a have different devices {} != {}".format(out.device, x.device)
            )
        dtype = out.dtype

    cumop = partial_op(
        x._DNDarray__array,
        axis,
        out=None if out is None else out._DNDarray__array,
        dtype=None if dtype is None else dtype.torch_type(),
    )

    if x.split is not None and axis == x.split:
        indices = torch.tensor([cumop.shape[axis] - 1])
        send = (
            torch.index_select(cumop, axis, indices)
            if indices[0] >= 0
            else torch.full(
                cumop.shape[:axis] + torch.Size([1]) + cumop.shape[axis + 1 :],
                neutral,
                dtype=cumop.dtype,
            )
        )
        recv = torch.full(
            cumop.shape[:axis] + torch.Size([1]) + cumop.shape[axis + 1 :],
            neutral,
            dtype=cumop.dtype,
        )

        x.comm.Exscan(send, recv, exscan_op)
        final_op(cumop, recv, out=cumop)

    if out is not None:
        return out

    return factories.array(
        cumop, dtype=x.dtype if dtype is None else dtype, is_split=x.split, device=x.device
    )


def __local_op(operation, x, out, no_cast=False, **kwargs):
    """
    Generic wrapper for local operations, which do not require communication. Accepts the actual operation function as
    argument and takes only care of buffer allocation/writing. This function is intended to work on an element-wise bases
    WARNING: the gshape of the result will be the same as x

    Parameters
    ----------
    operation : function
        A function implementing the element-wise local operation, e.g. torch.sqrt
    x : ht.DNDarray
        The value for which to compute 'operation'.
    no_cast : bool
        Flag to avoid casting to floats
    out : ht.DNDarray or None
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Returns
    -------
    result : ht.DNDarray
        A tensor of the same shape as x, containing the result of 'operation' for each element in x. If out was
        provided, result is a reference to it.

    Raises
    -------
    TypeError
        If the input is not a tensor or the output is not a tensor or None.
    """
    # perform sanitation
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("expected x to be a ht.DNDarray, but was {}".format(type(x)))
    if out is not None and not isinstance(out, dndarray.DNDarray):
        raise TypeError("expected out to be None or an ht.DNDarray, but was {}".format(type(out)))

    # infer the output type of the tensor
    # we need floating point numbers here, due to PyTorch only providing sqrt() implementation for float32/64
    if not no_cast:
        promoted_type = types.promote_types(x.dtype, types.float32)
        torch_type = promoted_type.torch_type()
    else:
        torch_type = x._DNDarray__array.dtype

    # no defined output tensor, return a freshly created one
    if out is None:
        result = operation(x._DNDarray__array.type(torch_type), **kwargs)
        return dndarray.DNDarray(
            result, x.gshape, types.canonical_heat_type(result.dtype), x.split, x.device, x.comm
        )

    # output buffer writing requires a bit more work
    # we need to determine whether the operands are broadcastable and the multiple of the broadcasting
    # reason: manually repetition for each dimension as PyTorch does not conform to numpy's broadcast semantic
    # PyTorch always recreates the input shape and ignores broadcasting for too large buffers
    broadcast_shape = stride_tricks.broadcast_shape(x.lshape, out.lshape)
    padded_shape = (1,) * (len(broadcast_shape) - len(x.lshape)) + x.lshape
    multiples = [int(a / b) for a, b in zip(broadcast_shape, padded_shape)]
    needs_repetition = builtins.any(multiple > 1 for multiple in multiples)

    # do an inplace operation into a provided buffer
    casted = x._DNDarray__array.type(torch_type)
    operation(
        casted.repeat(multiples) if needs_repetition else casted, out=out._DNDarray__array, **kwargs
    )

    return out


def __reduce_op(x, partial_op, reduction_op, neutral=None, **kwargs):
    """
    Generic wrapper for reduction operations, e.g. sum(), prod() etc. Performs a two-stage reduction. First, a partial
    reduction is performed node-local that is combined into a global reduction result via an MPI_Op.

    Parameters
    ----------
    x : ht.DNDarray
        The heat DNDarray on which to perform the reduction operation

    partial_op: function
        The function performing a partial reduction on the process-local data portion, e.g. sum() for implementing a
        distributed mean() operation.

    reduction_op: mpi4py.MPI.Op
        The MPI operator for performing the full reduction based on the results returned by the partial_op function.

    neutral: scalar
        Neutral element, i.e. an element that does not change the result of the reduction operation. Needed for
        those cases where 'x.gshape[x.split] < x.comm.rank', that is, the shape of the distributed tensor is such
        that one or more processes will be left without data.

    Returns
    -------
    result: ht.DNDarray
        A DNDarray containing the result of the reduction operation

    Raises
    ------
    TypeError
        If the input or optional output parameter are not of type ht.DNDarray
    ValueError
        If the shape of the optional output parameters does not match the shape of the reduced result
    """
    # perform sanitation
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("expected x to be a ht.DNDarray, but was {}".format(type(x)))
    out = kwargs.get("out")
    if out is not None and not isinstance(out, dndarray.DNDarray):
        raise TypeError("expected out to be None or an ht.DNDarray, but was {}".format(type(out)))

    # no further checking needed, sanitize axis will raise the proper exceptions
    axis = stride_tricks.sanitize_axis(x.shape, kwargs.get("axis"))
    if isinstance(axis, int):
        axis = (axis,)
    keepdim = kwargs.get("keepdim")
    split = x.split

    # if local tensor is empty, replace it with the identity element
    if 0 in x.lshape and (axis is None or (x.split in axis)):
        if neutral is None:
            neutral = float("nan")
        neutral_shape = x.lshape[:split] + (1,) + x.lshape[split + 1 :]
        partial = torch.full(neutral_shape, fill_value=neutral, dtype=x._DNDarray__array.dtype)
    else:
        partial = x._DNDarray__array

    # apply the partial reduction operation to the local tensor
    if axis is None:
        partial = partial_op(partial).reshape(-1)
        output_shape = (1,)
    else:
        output_shape = x.gshape
        for dim in axis:
            partial = partial_op(partial, dim=dim, keepdim=True)
            output_shape = output_shape[:dim] + (1,) + output_shape[dim + 1 :]
        if not keepdim and not len(partial.shape) == 1:
            gshape_losedim = tuple(x.gshape[dim] for dim in range(len(x.gshape)) if dim not in axis)
            lshape_losedim = tuple(x.lshape[dim] for dim in range(len(x.lshape)) if dim not in axis)
            output_shape = gshape_losedim
            # Take care of special cases argmin and argmax: keep partial.shape[0]
            if 0 in axis and partial.shape[0] != 1:
                lshape_losedim = (partial.shape[0],) + lshape_losedim
            if 0 not in axis and partial.shape[0] != x.lshape[0]:
                lshape_losedim = (partial.shape[0],) + lshape_losedim[1:]
            partial = partial.reshape(lshape_losedim)

    # Check shape of output buffer, if any
    if out is not None and out.shape != output_shape:
        raise ValueError(
            "Expecting output buffer of shape {}, got {}".format(output_shape, out.shape)
        )

    # perform a reduction operation in case the tensor is distributed across the reduction axis
    if x.split is not None and (axis is None or (x.split in axis)):
        split = None
        if x.comm.is_distributed():
            x.comm.Allreduce(MPI.IN_PLACE, partial, reduction_op)

    # if reduction_op is a Boolean operation, then resulting tensor is bool
    tensor_type = bool if reduction_op in __BOOLEAN_OPS else partial.dtype

    if out is not None:
        out._DNDarray__array = partial
        out._DNDarray__dtype = types.canonical_heat_type(tensor_type)
        out._DNDarray__split = split
        out._DNDarray__device = x.device
        out._DNDarray__comm = x.comm

        return out

    return dndarray.DNDarray(
        partial,
        output_shape,
        types.canonical_heat_type(tensor_type),
        split=split,
        device=x.device,
        comm=x.comm,
    )
