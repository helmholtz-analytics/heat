import builtins
import numpy as np
import torch
import warnings

from .communication import MPI
from . import factories
from . import stride_tricks
from . import tensor
from . import types

__all__ = [
    'all',
    'allclose',
    'any'
]


def all(x, axis=None, out=None, keepdim=False):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters:
    -----------

    x : ht.Tensor
        Input array or object that can be converted to an array.

    axis : None or int, optional #TODO: tuple of ints, issue #67
        Axis or along which a logical AND reduction is performed. The default (axis = None) is to perform a 
        logical AND over all the dimensions of the input array. axis may be negative, in which case it counts 
        from the last to the first axis.

    out : ht.Tensor, optional
        Alternate output array in which to place the result. It must have the same shape as the expected output 
        and its type is preserved.

    Returns:	
    --------
    all : ht.Tensor, bool

    A new boolean or ht.Tensor is returned unless out is specified, in which case a reference to out is returned.

    Examples:
    ---------
    >>> import heat as ht
    >>> a = ht.random.randn(4, 5)
    >>> a
    tensor([[ 0.5370, -0.4117, -3.1062,  0.4897, -0.3231],
            [-0.5005, -1.7746,  0.8515, -0.9494, -0.2238],
            [-0.0444,  0.3388,  0.6805, -1.3856,  0.5422],
            [ 0.3184,  0.0185,  0.5256, -1.1653, -0.1665]])
    >>> x = a < 0.5
    >>> x
    tensor([[0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1]], dtype=ht.uint8)
    >>> ht.all(x)
    tensor([0], dtype=ht.uint8)
    >>> ht.all(x, axis=0)
    tensor([[0, 1, 0, 1, 0]], dtype=ht.uint8)
    >>> ht.all(x, axis=1)
    tensor([[0],
            [0],
            [0],
            [0]], dtype=ht.uint8)

    Write out to predefined buffer:
    >>> out = ht.zeros((1, 5))
    >>> ht.all(x, axis=0, out=out)
    >>> out
    tensor([[0, 1, 0, 1, 0]], dtype=ht.uint8)
    """
    # TODO: make me more numpy API complete. Issue #101
    def local_all(t, *args, **kwargs):
        return torch.all(t != 0, *args, **kwargs)

    return __reduce_op(x, local_all, MPI.LAND, axis=axis, out=out, keepdim=keepdim)


def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Test whether two tensors are element-wise equal within a tolerance. Returns True if |x - y| <= atol + rtol * |y|
    for all elements of x and y, False otherwise

    Parameters:
    -----------
    x : ht.Tensor
        First tensor to compare
    y : ht.Tensor
        Second tensor to compare
    atol: float, optional
        Absolute tolerance. Default is 1e-08
    rtol: float, optional
        Relative tolerance (with respect to y). Default is 1e-05
    equal_nan: bool, optional
        Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal to NaN’s in b in the output
        array.

    Returns:
    --------
    allclose : bool
        True if the two tensors are equal within the given tolerance; False otherwise.

    Examples:
    ---------
    >>> a = ht.float32([[2, 2], [2, 2]])
    >>> ht.allclose(a, a)
    True

    >>> b = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])
    >>> ht.allclose(a, b)
    False
    >>> ht.allclose(a, b, atol=1e-04)
    True

    """
    if not isinstance(x, tensor.Tensor):
        raise TypeError('Expected x to be a ht.Tensor, but was {}'.format(type(x)))

    if not isinstance(y, tensor.Tensor):
        raise TypeError('Expected y to be a ht.Tensor, but was {}'.format(type(y)))

    return torch.allclose(x._Tensor__array, y._Tensor__array, rtol, atol, equal_nan)


def any(x, axis=None, out=None):
    """
    Test whether any array element along a given axis evaluates to True.
    The returning tensor is one dimensional unless axis is not None.

    Parameters:
    -----------
    x : tensor
        Input tensor
    axis : int, optional
        Axis along which a logic OR reduction is performed. With axis=None, the logical OR is performed over all
        dimensions of the tensor.
    out : tensor, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output.
        The output is a tensor with dtype=bool.

    Returns:
    --------
    boolean_tensor : tensor of type bool
        Returns a tensor of booleans that are 1, if any non-zero values exist on this axis, 0 otherwise.

    Examples:
    ---------
    >>> import heat as ht
    >>> t = ht.float32([[0.3, 0, 0.5]])
    >>> t.any()
    tensor([1], dtype=torch.uint8)
    >>> t.any(axis=0)
    tensor([[1, 0, 1]], dtype=torch.uint8)
    >>> t.any(axis=1)
    tensor([[1]], dtype=torch.uint8)

    >>> t = ht.int32([[0, 0, 1], [0, 0, 0]])
    >>> res = ht.zeros((1, 3), dtype=ht.bool)
    >>> t.any(axis=0, out=res)
    tensor([[0, 0, 1]], dtype=torch.uint8)
    >>> res
    tensor([[0, 0, 1]], dtype=torch.uint8)
    """
    def local_any(t, *args, **kwargs):
        return torch.any(t != 0, *args, **kwargs)

    return __reduce_op(x, local_any, MPI.LOR, axis=axis, out=out, keepdim=False)


def __local_operation(operation, x, out):
    """
    Generic wrapper for local operations, which do not require communication. Accepts the actual operation function as
    argument and takes only care of buffer allocation/writing.

    Parameters
    ----------
    operation : function
        A function implementing the element-wise local operation, e.g. torch.sqrt
    x : ht.Tensor
        The value for which to compute 'operation'.
    out : ht.Tensor or None
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Returns
    -------
    result : ht.Tensor
        A tensor of the same shape as x, containing the result of 'operation' for each element in x. If out was
        provided, result is a reference to it.

    Raises
    -------
    TypeError
        If the input is not a tensor or the output is not a tensor or None.
    """
    # perform sanitation
    if not isinstance(x, tensor.Tensor):
        raise TypeError('expected x to be a ht.Tensor, but was {}'.format(type(x)))
    if out is not None and not isinstance(out, tensor.Tensor):
        raise TypeError('expected out to be None or an ht.Tensor, but was {}'.format(type(out)))

    # infer the output type of the tensor
    # we need floating point numbers here, due to PyTorch only providing sqrt() implementation for float32/64
    promoted_type = types.promote_types(x.dtype, types.float32)
    torch_type = promoted_type.torch_type()

    # no defined output tensor, return a freshly created one
    if out is None:
        result = operation(x._Tensor__array.type(torch_type))
        return tensor.Tensor(result, x.gshape, promoted_type, x.split, x.device, x.comm)

    # output buffer writing requires a bit more work
    # we need to determine whether the operands are broadcastable and the multiple of the broadcasting
    # reason: manually repetition for each dimension as PyTorch does not conform to numpy's broadcast semantic
    # PyTorch always recreates the input shape and ignores broadcasting for too large buffers
    broadcast_shape = stride_tricks.broadcast_shape(x.lshape, out.lshape)
    padded_shape = (1,) * (len(broadcast_shape) - len(x.lshape)) + x.lshape
    multiples = [int(a / b) for a, b in zip(broadcast_shape, padded_shape)]
    needs_repetition = builtins.any(multiple > 1 for multiple in multiples)

    # do an inplace operation into a provided buffer
    casted = x._Tensor__array.type(torch_type)
    operation(casted.repeat(multiples) if needs_repetition else casted, out=out._Tensor__array)

    return out


def __reduce_op(x, partial_op, reduction_op, **kwargs):
    # TODO: document me Issue #102
    # perform sanitation
    if not isinstance(x, tensor.Tensor):
        raise TypeError('expected x to be a ht.Tensor, but was {}'.format(type(x)))
    out = kwargs.get('out')
    if out is not None and not isinstance(out, tensor.Tensor):
        raise TypeError('expected out to be None or an ht.Tensor, but was {}'.format(type(out)))

    # no further checking needed, sanitize axis will raise the proper exceptions
    axis = stride_tricks.sanitize_axis(x.shape, kwargs.get('axis'))
    split = x.split

    if axis is None:
        partial = partial_op(x._Tensor__array).reshape(-1)
        output_shape = (1,)
    else:
        partial = partial_op(x._Tensor__array, dim=axis, keepdim=True)
        shape_keepdim = x.gshape[:axis] + (1,) + x.gshape[axis + 1:]
        shape_losedim = x.gshape[:axis] + x.gshape[axis + 1:]
        output_shape = shape_keepdim if kwargs.get('keepdim') else shape_losedim

    # Check shape of output buffer, if any
    if out is not None and out.shape != output_shape:
        raise ValueError('Expecting output buffer of shape {}, got {}'.format(output_shape, out.shape))

    # perform a reduction operation in case the tensor is distributed across the reduction axis
    if x.split is not None and (axis is None or axis == x.split):
        split = None
        if x.comm.is_distributed():
            x.comm.Allreduce(MPI.IN_PLACE, partial, reduction_op)

    # if reduction_op is a Boolean operation, then resulting tensor is bool
    boolean_ops = [MPI.LAND, MPI.LOR, MPI.BAND, MPI.BOR]
    tensor_type = bool if reduction_op in boolean_ops else partial[0].dtype

    if out is not None:
        out._Tensor__array = partial
        out._Tensor__dtype = types.canonical_heat_type(tensor_type)
        out._tensor__split = split
        out._tensor__device = x.device
        out._tensor__comm = x.comm

        return out

    return tensor.Tensor(
        partial,
        output_shape,
        types.canonical_heat_type(tensor_type),
        split=split,
        device=x.device,
        comm=x.comm
    )


def __binary_op(operation, t1, t2):
    """
    Generic wrapper for element-wise binary operations of two operands (either can be tensor or scalar).
    Takes the operation function and the two operands involved in the operation as arguments.

    Parameters
    ----------
    operation : function
        The operation to be performed. Function that performs operation elements-wise on the involved tensors,
        e.g. add values from other to self

    t1: tensor or scalar
        The first operand involved in the operation,

    t2: tensor or scalar
        The second operand involved in the operation,

    Returns
    -------
    result: ht.Tensor
        A tensor containing the results of element-wise operation.
    """
    if np.isscalar(t1):
        try:
            t1 = factories.array([t1])
        except (ValueError, TypeError,):
            raise TypeError('Data type not supported, input was {}'.format(type(t1)))

        if np.isscalar(t2):
            try:
                t2 = factories.array([t2])
            except (ValueError, TypeError,):
                raise TypeError('Only numeric scalars are supported, but input was {}'.format(type(t2)))
            output_shape = (1,)
            output_split = None
            output_device = None
            output_comm = None
        elif isinstance(t2, tensor.Tensor):
            output_shape = t2.shape
            output_split = t2.split
            output_device = t2.device
            output_comm = t2.comm
        else:
            raise TypeError('Only tensors and numeric scalars are supported, but input was {}'.format(type(t2)))

        if t1.dtype != t2.dtype:
            t1 = t1.astype(t2.dtype)

    elif isinstance(t1, tensor.Tensor):
        if np.isscalar(t2):
            try:
                t2 = factories.array([t2])
                output_shape = t1.shape
                output_split = t1.split
                output_device = t1.device
                output_comm = t1.comm
            except (ValueError, TypeError,):
                raise TypeError('Data type not supported, input was {}'.format(type(t2)))

        elif isinstance(t2, tensor.Tensor):
            # TODO: implement complex NUMPY rules
            if t2.split is None or t2.split == t1.split:
                output_shape = stride_tricks.broadcast_shape(t1.shape, t2.shape)
                output_split = t1.split
                output_device = t1.device
                output_comm = t1.comm
            else:
                # It is NOT possible to perform binary operations on tensors with different splits, e.g. split=0
                # and split=1
                raise NotImplementedError('Not implemented for other splittings')

            # ToDo: Fine tuning in case of comm.size>t1.shape[t1.split]. Send torch tensors only to ranks, that will hold data.
            if t1.split is not None:
                if t1.shape[t1.split] == 1 and t1.comm.is_distributed():
                    warnings.warn('Broadcasting requires transferring data of first operator between MPI ranks!')
                    if t1.comm.rank > 0:
                        t1._Tensor__array = torch.zeros(t1.shape, dtype=t1.dtype.torch_type())
                    t1.comm.Bcast(t1)

            if t2.split is not None:
                if t2.shape[t2.split] == 1 and t2.comm.is_distributed():
                    warnings.warn('Broadcasting requires transferring data of second operator between MPI ranks!')
                    if t2.comm.rank > 0:
                        t2._Tensor__array = torch.zeros(t2.shape, dtype=t2.dtype.torch_type())
                    t2.comm.Bcast(t2)

        else:
            raise TypeError('Only tensors and numeric scalars are supported, but input was {}'.format(type(t2)))

        if t2.dtype != t1.dtype:
            t2 = t2.astype(t1.dtype)

    else:
        raise NotImplementedError('Not implemented for non scalar')

    if t1.split is not None:
        if t1.lshape[t1.split] == 0:
            result = t1
        else:
            result = operation(t1._Tensor__array, t2._Tensor__array)
    elif t1.split is not None:
        if t2.lshape[t2.split] == 0:
            result = t2
        else:
            result = operation(t1._Tensor__array, t2._Tensor__array)
    else:
        result = operation(t1._Tensor__array, t2._Tensor__array)

    return tensor.Tensor(result, output_shape, t1.dtype, output_split, output_device, output_comm)
