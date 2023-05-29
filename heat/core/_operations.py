"""Generalized operations for DNDarray"""

import builtins
import numpy as np
import torch
import warnings

from .communication import MPI, MPI_WORLD
from . import factories
from . import stride_tricks
from . import sanitation
from . import statistics
from .dndarray import DNDarray
from . import types

from typing import Callable, Optional, Type, Union, Dict

__all__ = []
__BOOLEAN_OPS = [MPI.LAND, MPI.LOR, MPI.BAND, MPI.BOR]


def __binary_op(
    operation: Callable,
    t1: Union[DNDarray, int, float],
    t2: Union[DNDarray, int, float],
    out: Optional[DNDarray] = None,
    where: Optional[DNDarray] = None,
    fn_kwargs: Optional[Dict] = {},
) -> DNDarray:
    """
    Generic wrapper for element-wise binary operations of two operands (either can be tensor or scalar).
    Takes the operation function and the two operands involved in the operation as arguments.

    Parameters
    ----------
    operation : function
        The operation to be performed. Function that performs operation elements-wise on the involved tensors,
        e.g. add values from other to self
    t1: DNDarray or scalar
        The first operand involved in the operation.
    t2: DNDarray or scalar
        The second operand involved in the operation.
    out: DNDarray, optional
        Output buffer in which the result is placed. If not provided, a freshly allocated array is returned.
    where: DNDarray, optional
        Condition to broadcast over the inputs. At locations where the condition is True, the `out` array
        will be set to the result of the operation. Elsewhere, the `out` array will retain its original
        value. If an uninitialized `out` array is created via the default `out=None`, locations within
        it where the condition is False will remain uninitialized. If distributed, the split axis (after
        broadcasting if required) must match that of the `out` array.
    fn_kwargs: Dict, optional
        keyword arguments used for the given operation
        Default: {} (empty dictionary)

    Returns
    -------
    result: ht.DNDarray
        A DNDarray containing the results of element-wise operation.

    Warning
    -------
    If both operands are distributed, they must be distributed along the same dimension, i.e. `t1.split = t2.split`.

    MPI communication is necessary when both operands are distributed along the same dimension, but the distribution maps do not match. E.g.:
    ```
    a =  ht.ones(10000, split=0)
    b = ht.zeros(10000, split=0)
    c = a[:-1] + b[1:]
    ```
    In such cases, one of the operands is redistributed OUT-OF-PLACE to match the distribution map of the other operand.
    The operand determining the resulting distribution is chosen as follows:
    1) split is preferred to no split
    2) no (shape)-broadcasting in the split dimension if not necessary
    3) t1 is preferred to t2
    """
    # Check inputs
    if not np.isscalar(t1) and not isinstance(t1, DNDarray):
        raise TypeError(
            f"Only DNDarrays and numeric scalars are supported, but input was {type(t1)}"
        )
    if not np.isscalar(t2) and not isinstance(t2, DNDarray):
        raise TypeError(
            f"Only DNDarrays and numeric scalars are supported, but input was {type(t2)}"
        )
    promoted_type = types.result_type(t1, t2).torch_type()

    # Make inputs Dndarrays
    if np.isscalar(t1) and np.isscalar(t2):
        try:
            t1 = factories.array(t1)
            t2 = factories.array(t2)
        except (ValueError, TypeError):
            raise TypeError(f"Data type not supported, inputs were {type(t1)} and {type(t2)}")
    elif np.isscalar(t1) and isinstance(t2, DNDarray):
        try:
            t1 = factories.array(t1, device=t2.device, comm=t2.comm)
        except (ValueError, TypeError):
            raise TypeError(f"Data type not supported, input was {type(t1)}")
    elif isinstance(t1, DNDarray) and np.isscalar(t2):
        try:
            t2 = factories.array(t2, device=t1.device, comm=t1.comm)
        except (ValueError, TypeError):
            raise TypeError(f"Data type not supported, input was {type(t2)}")

    # Make inputs have the same dimensionality
    output_shape = stride_tricks.broadcast_shape(t1.shape, t2.shape)
    if where is not None:
        output_shape = stride_tricks.broadcast_shape(where.shape, output_shape)
        while len(where.shape) < len(output_shape):
            where = where.expand_dims(axis=0)
    # Broadcasting allows additional empty dimensions on the left side
    # TODO simplify this once newaxis-indexing is supported to get rid of the loops
    while len(t1.shape) < len(output_shape):
        t1 = t1.expand_dims(axis=0)
    while len(t2.shape) < len(output_shape):
        t2 = t2.expand_dims(axis=0)
    # t1 = t1[tuple([None] * (len(output_shape) - t1.ndim))]
    # t2 = t2[tuple([None] * (len(output_shape) - t2.ndim))]
    # print(t1.lshape, t2.lshape)

    def __get_out_params(target, other=None, map=None):
        """
        Getter for the output parameters of a binary operation with target distribution.
        If `other` is provided, its distribution will be matched to `target` or, if provided,
        redistributed according to `map`.

        Parameters
        ----------
        target : DNDarray
            DNDarray determining the parameters
        other : DNDarray
            DNDarray to be adapted
        map : Tensor
            lshape_map `other` should be matched to. Defaults to `target.lshape_map`

        Returns
        -------
        Tuple
            split, device, comm, balanced, [other]
        """
        if other is not None:
            if out is None:
                other = sanitation.sanitize_distribution(other, target=target, diff_map=map)
            return target.split, target.device, target.comm, target.balanced, other
        return target.split, target.device, target.comm, target.balanced

    if t1.split is not None and t1.shape[t1.split] == output_shape[t1.split]:  # t1 is "dominant"
        output_split, output_device, output_comm, output_balanced, t2 = __get_out_params(t1, t2)
    elif t2.split is not None and t2.shape[t2.split] == output_shape[t2.split]:  # t2 is "dominant"
        output_split, output_device, output_comm, output_balanced, t1 = __get_out_params(t2, t1)
    elif t1.split is not None:
        # t1 is split but broadcast -> only on one rank; manipulate lshape_map s.t. this rank has 'full' data
        lmap = t1.lshape_map
        idx = lmap[:, t1.split].nonzero(as_tuple=True)[0]
        lmap[idx.item(), t1.split] = output_shape[t1.split]
        output_split, output_device, output_comm, output_balanced, t2 = __get_out_params(
            t1, t2, map=lmap
        )
    elif t2.split is not None:
        # t2 is split but broadcast -> only on one rank; manipulate lshape_map s.t. this rank has 'full' data
        lmap = t2.lshape_map
        idx = lmap[:, t2.split].nonzero(as_tuple=True)[0]
        lmap[idx.item(), t2.split] = output_shape[t2.split]
        output_split, output_device, output_comm, output_balanced, t1 = __get_out_params(
            t2, other=t1, map=lmap
        )
    else:  # both are not split
        output_split, output_device, output_comm, output_balanced = __get_out_params(t1)

    if out is not None:
        sanitation.sanitize_out(out, output_shape, output_split, output_device, output_comm)
        t1, t2 = sanitation.sanitize_distribution(t1, t2, target=out)

    result = operation(t1.larray.to(promoted_type), t2.larray.to(promoted_type), **fn_kwargs)

    if out is None and where is None:
        return DNDarray(
            result,
            output_shape,
            types.heat_type_of(result),
            output_split,
            device=output_device,
            comm=output_comm,
            balanced=output_balanced,
        )

    if where is not None:
        if out is None:
            out = factories.empty(
                output_shape,
                dtype=promoted_type,
                split=output_split,
                device=output_device,
                comm=output_comm,
            )
        if where.split != out.split:
            where = sanitation.sanitize_distribution(where, target=out)
        result = torch.where(where.larray, result, out.larray)

    out.larray.copy_(result)
    return out


def __cum_op(
    x: DNDarray,
    partial_op: Callable,
    exscan_op: Callable,
    final_op: Callable,
    neutral: Union[int, float],
    axis: Union[int, float],
    dtype: Union[str, Type[types.datatype]],
    out: Optional[DNDarray] = None,
) -> DNDarray:
    """
    Generic wrapper for cumulative operations. Performs a three-stage cumulative operation. First, a partial
    cumulative operation is performed node-local that is combined into a global cumulative result via an MPI_Op and a final local
    reduction add or mul operation.

    Parameters
    ----------
    x : DNDarray
        The heat DNDarray on which to perform the cumulative operation
    partial_op: function
        The function performing a partial cumulative operation on the process-local data portion, e.g. :func:`cumsum() <heat.arithmetics.cumsum>`.
    exscan_op: mpi4py.MPI.Op
        The MPI operator for performing the exscan based on the results returned by the partial_op function.
    final_op: function
        The local operation for the final result, e.g. :func:`add() <heat.arithmetics.add>` for :func:`cumsum() <heat.arithmetics.cumsum>`.
    neutral: scalar
        Neutral element for the cumulative operation, i.e. an element that does not change the reductions operations
        result.
    axis: int
        The axis direction of the cumulative operation
    dtype: datatype
        The type of the result tensor.
    out: DNDarray, optional
        The explicitly returned output tensor.

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
    sanitation.sanitize_in(x)

    if axis is None:
        raise NotImplementedError("axis = None is not supported")
    axis = stride_tricks.sanitize_axis(x.shape, axis)

    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    if out is not None:
        sanitation.sanitize_out(out, x.shape, x.split, x.device)
        dtype = out.dtype

    cumop = partial_op(
        x.larray,
        axis,
        out=None if out is None else out.larray,
        dtype=None if dtype is None else dtype.torch_type(),
    )

    if x.split is not None and axis == x.split:
        indices = torch.tensor([cumop.shape[axis] - 1], device=cumop.device)
        send = (
            torch.index_select(cumop, axis, indices)
            if indices[0] >= 0
            else torch.full(
                cumop.shape[:axis] + torch.Size([1]) + cumop.shape[axis + 1 :],
                neutral,
                dtype=cumop.dtype,
                device=cumop.device,
            )
        )
        recv = torch.full(
            cumop.shape[:axis] + torch.Size([1]) + cumop.shape[axis + 1 :],
            neutral,
            dtype=cumop.dtype,
            device=cumop.device,
        )

        x.comm.Exscan(send, recv, exscan_op)
        final_op(cumop, recv, out=cumop)

    if out is not None:
        return out

    return factories.array(
        cumop,
        dtype=x.dtype if dtype is None else dtype,
        is_split=x.split,
        device=x.device,
        comm=x.comm,
    )


def __local_op(
    operation: Callable,
    x: DNDarray,
    out: Optional[DNDarray] = None,
    no_cast: Optional[bool] = False,
    **kwargs,
) -> DNDarray:
    """
    Generic wrapper for local operations, which do not require communication. Accepts the actual operation function as
    argument and takes only care of buffer allocation/writing. This function is intended to work on an element-wise bases

    Parameters
    ----------
    operation : function
        A function implementing the element-wise local operation, e.g. torch.sqrt
    x : DNDarray
        The value for which to compute 'operation'.
    no_cast : bool
        Flag to avoid casting to floats
    out : DNDarray, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Warning
    -------
    The gshape of the result DNDarray will be the same as that of x

    Raises
    -------
    TypeError
        If the input is not a tensor or the output is not a tensor or None.
    """
    # perform sanitation
    sanitation.sanitize_in(x)
    if out is not None and not isinstance(out, DNDarray):
        raise TypeError(f"expected out to be None or an ht.DNDarray, but was {type(out)}")

    # infer the output type of the tensor
    # we need floating point numbers here, due to PyTorch only providing sqrt() implementation for float32/64
    if not no_cast:
        promoted_type = types.promote_types(x.dtype, types.float32)
        torch_type = promoted_type.torch_type()
    else:
        torch_type = x.larray.dtype

    # no defined output tensor, return a freshly created one
    if out is None:
        result = operation(x.larray.type(torch_type), **kwargs)
        return DNDarray(
            result,
            x.gshape,
            types.canonical_heat_type(result.dtype),
            x.split,
            x.device,
            x.comm,
            x.balanced,
        )

    # output buffer writing requires a bit more work
    # we need to determine whether the operands are broadcastable and the multiple of the broadcasting
    # reason: manually repetition for each dimension as PyTorch does not conform to numpy's broadcast semantic
    # PyTorch always recreates the input shape and ignores broadcasting for too large buffers
    broadcast_shape = stride_tricks.broadcast_shape(x.lshape, out.lshape)
    padded_shape = (1,) * (len(broadcast_shape) - len(x.lshape)) + x.lshape
    multiples = [(int(a / b) if b > 0 else 0) for a, b in zip(broadcast_shape, padded_shape)]
    needs_repetition = builtins.any(multiple > 1 for multiple in multiples)

    # do an inplace operation into a provided buffer
    casted = x.larray.type(torch_type)
    operation(casted.repeat(multiples) if needs_repetition else casted, out=out.larray, **kwargs)

    return out


def __reduce_op(
    x: DNDarray,
    partial_op: Callable,
    reduction_op: Callable,
    neutral: Optional[Union[int, float]] = None,
    **kwargs,
) -> DNDarray:
    """
    Generic wrapper for reduction operations, e.g. :func:`sum() <heat.arithmetics.sum>`, :func:`prod() <heat.arithmetics.prod>`
    etc. Performs a two-stage reduction. First, a partial reduction is performed node-local that is combined into a
    global reduction result via an MPI_Op.

    Parameters
    ----------
    x : DNDarray
        The DNDarray on which to perform the reduction operation
    partial_op: function
        The function performing a partial reduction on the process-local data portion, e.g. sum() for implementing a
        distributed mean() operation.
    reduction_op: mpi4py.MPI.Op
        The MPI operator for performing the full reduction based on the results returned by the partial_op function.
    neutral: scalar
        Neutral element, i.e. an element that does not change the result of the reduction operation. Needed for
        those cases where 'x.gshape[x.split] < x.comm.rank', that is, the shape of the distributed tensor is such
        that one or more processes will be left without data.

    Raises
    ------
    TypeError
        If the input or optional output parameter are not of type ht.DNDarray
    ValueError
        If the shape of the optional output parameters does not match the shape of the reduced result
    """
    # perform sanitation
    sanitation.sanitize_in(x)

    # no further checking needed, sanitize axis will raise the proper exceptions
    axis = stride_tricks.sanitize_axis(x.shape, kwargs.get("axis"))
    if isinstance(axis, int):
        axis = (axis,)
    keepdims = kwargs.get("keepdims")
    out = kwargs.get("out")
    split = x.split
    balanced = x.balanced

    # if local tensor is empty, replace it with the identity element
    if x.is_distributed() and 0 in x.lshape and (axis is None or split in axis):
        if neutral is None:
            neutral = float("nan")
        neutral_shape = x.gshape[:split] + (1,) + x.gshape[split + 1 :]
        partial = torch.full(
            neutral_shape,
            fill_value=neutral,
            dtype=x.dtype.torch_type(),
            device=x.device.torch_device,
        )
    else:
        partial = x.larray

    # apply the partial reduction operation to the local tensor
    if axis is None:
        partial = partial_op(partial).reshape(-1)
        output_shape = (1,)
        balanced = True
    else:
        output_shape = x.gshape
        for dim in axis:
            if not (
                partial.shape.numel() == 0 and partial_op.__name__ in ("local_max", "local_min")
            ):  # no neutral element for max/min
                partial = partial_op(partial, dim=dim, keepdim=True)
            output_shape = output_shape[:dim] + (1,) + output_shape[dim + 1 :]
        if not keepdims and len(partial.shape) != 1:
            gshape_losedim = tuple(x.gshape[dim] for dim in range(len(x.gshape)) if dim not in axis)
            lshape_losedim = tuple(x.lshape[dim] for dim in range(len(x.lshape)) if dim not in axis)
            output_shape = gshape_losedim
            # Take care of special cases argmin and argmax: keep partial.shape[0]
            if 0 in axis and partial.shape[0] != 1:
                lshape_losedim = (partial.shape[0],) + lshape_losedim
            if 0 not in axis and partial.shape[0] != x.lshape[0]:
                lshape_losedim = (partial.shape[0],) + lshape_losedim[1:]
            if len(lshape_losedim) > 0:
                partial = partial.reshape(lshape_losedim)
    # perform a reduction operation in case the tensor is distributed across the reduction axis
    if x.split is not None:
        if axis is None or (x.split in axis):
            split = None
            balanced = True
            if x.comm.is_distributed():
                x.comm.Allreduce(MPI.IN_PLACE, partial, reduction_op)
        elif axis is not None and not keepdims:
            down_dims = len(tuple(dim for dim in axis if dim < x.split))
            split -= down_dims
            balanced = x.balanced

    ARG_OPS = [statistics.MPI_ARGMAX, statistics.MPI_ARGMIN]
    arg_op = False
    if reduction_op in ARG_OPS:
        arg_op = True
        partial = partial.chunk(2)[-1].type(torch.int64)
        if partial.ndim > 1:
            partial = partial.squeeze(dim=0)

    # if reduction_op is a Boolean operation, then resulting tensor is bool
    tensor_type = bool if reduction_op in __BOOLEAN_OPS else partial.dtype

    if out is not None:
        # sanitize out
        sanitation.sanitize_out(out, output_shape, split, x.device)
        if arg_op and out.dtype != types.canonical_heat_type(partial.dtype):
            raise TypeError(
                f"Data type mismatch: out.dtype should be {types.canonical_heat_type(partial.dtype)}, is {out.dtype}"
            )
        out._DNDarray__array = partial
        return out

    return DNDarray(
        partial,
        output_shape,
        types.canonical_heat_type(tensor_type),
        split=split,
        device=x.device,
        comm=x.comm,
        balanced=balanced,
    )
