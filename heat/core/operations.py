import itertools
import torch
import numpy as np


from .communication import MPI
from . import stride_tricks
from . import types
from . import tensor

__all__ = [
    'all',
    'allclose',
    'argmin',
    'clip',
    'copy',
    'transpose',
    'tril',
    'triu'
]




def all(x, axis=None, out=None):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters:
    -----------

    x : ht.tensor
        Input array or object that can be converted to an array.

    axis : None or int, optional #TODO: tuple of ints, issue #67
        Axis or along which a logical AND reduction is performed. The default (axis = None) is to perform a 
        logical AND over all the dimensions of the input array. axis may be negative, in which case it counts 
        from the last to the first axis.

    out : ht.tensor, optional
        Alternate output array in which to place the result. It must have the same shape as the expected output 
        and its type is preserved.

    Returns:	
    --------
    all : ht.tensor, bool

    A new boolean or ht.tensor is returned unless out is specified, in which case a reference to out is returned.

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
    return __reduce_op(x, lambda t, *args, **kwargs: t.byte().all(*args, **kwargs), MPI.LAND, axis, out=out)


def allclose(x, y, rtol = 1e-05, atol = 1e-08, equal_nan = False):
    """
    Test whether two tensors are element-wise equal within a tolerance. Returns True if |x - y| <= atol + rtol * |y| for all elements of x and y, False otherwise

    Parameters:
    -----------

    x : ht.tensor
        First tensor to compare

    y : ht.tensor
        Second tensor to compare

    atol: float, optional
        Absolute tolerance. Default is 1e-08

    rtol: float, optional
        Relative tolerance (with respect to y). Default is 1e-05

    equal_nan: bool, optional
        Whether to compare NaN’s as equal. If True, NaN’s in a will be considered equal to NaN’s in b in the output array.

    Returns:
    --------
    allclose : bool
    True if the two tensors are equal within the given tolerance; False otherwise.

    Examples:
    ---------
    >>> a = ht.float32([[2, 2], [2, 2]])
    >>> ht.allclose(a,a)
    True

    >>> b = ht.float32([[2.00005,2.00005],[2.00005,2.00005]])
    >>> ht.allclose(a,b)
    False
    >>> ht.allclose(a,b, atol=1e-04)
    True

    """

    if not isinstance(x, tensor.tensor):
        raise TypeError('Expected x to be a ht.tensor, but was {}'.format(type(x)))

    if not isinstance(y, tensor.tensor):
        raise TypeError('Expected y to be a ht.tensor, but was {}'.format(type(y)))

    return torch.allclose(x._tensor__array, y._tensor__array, rtol, atol, equal_nan)

def argmin(x, axis=None, out=None):
    """
    Returns the indices of the minimum values along an axis.

    Parameters:
    ----------
    x : ht.tensor
        Input array.
    axis : int, optional
        By default, the index is into the flattened tensor, otherwise along the specified axis.
    # TODO out : ht.tensor, optional. Issue #100
        If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

    Returns:
    -------
    index_tensor : ht.tensor of ints
        Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

    Examples:
    --------
    >>> a = ht.randn(3,3)
    >>> a
    tensor([[-1.7297,  0.2541, -0.1044],
            [ 1.0865, -0.4415,  1.3716],
            [-0.0827,  1.0215, -2.0176]])
    >>> ht.argmin(a)
    tensor([8])
    >>> ht.argmin(a, axis=0)
    tensor([[0, 1, 2]])
    >>> ht.argmin(a, axis=1)
    tensor([[0],
            [1],
            [2]])
    """
    axis = stride_tricks.sanitize_axis(x.shape, axis)

    if axis is None:
        # TEMPORARY SOLUTION! TODO: implementation for axis=None, distributed tensor Issue #100
        # perform sanitation
        if not isinstance(x, tensor.tensor):
            raise TypeError(
                'expected x to be a ht.tensor, but was {}'.format(type(x)))

        out = torch.reshape(torch.argmin(x._tensor__array), (1,))
        return tensor.tensor(out, out.shape, types.canonical_heat_type(out.dtype), None, x.device, x.comm)

    out = __reduce_op(x, torch.min, MPI.MIN, axis, out=None)._tensor__array[1]

    return tensor.tensor(out, out.shape, types.canonical_heat_type(out.dtype), x.split, x.device, x.comm)


def clip(a, a_min, a_max, out=None):
    """
    Parameters
    ----------
    a : ht.tensor
        Array containing elements to clip.
    a_min : scalar or None
        Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of a_min and
        a_max may be None.
    a_max : scalar or None
        Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of a_min and
        a_max may be None.
    out : ht.tensor, optional
        The results will be placed in this array. It may be the input array for in-place clipping. out must be of
        the right shape to hold the output. Its type is preserved.

    Returns
    -------
    clipped_values : ht.tensor
        A tensor with the elements of this tensor, but where values < a_min are replaced with a_min, and those >
        a_max with a_max.
    """
    if not isinstance(a, tensor.tensor):
        raise TypeError('a must be a tensor')
    if a_min is None and a_max is None:
        raise ValueError('either a_min or a_max must be set')

    if out is None:
        return tensor.tensor(a._tensor__array.clamp(a_min, a_max), a.shape, a.dtype, a.split, a.device, a.comm)
    if not isinstance(out, tensor.tensor):
        raise TypeError('out must be a tensor')

    return a._tensor__array.clamp(a_min, a_max, out=out._tensor__array) and out


def copy(a):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : ht.tensor
        Input data to be copied.

    Returns
    -------
    copied : ht.tensor
        A copy of the original
    """
    if not isinstance(a, tensor.tensor):
        raise TypeError('input needs to be a tensor')
    return tensor.tensor(a._tensor__array.clone(), a.shape, a.dtype, a.split, a.device, a.comm)



def transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : None or list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes according to the values given.

    Returns
    -------
    p : ht.tensor
        a with its axes permuted.
    """
    # type check the input tensor
    if not isinstance(a, tensor.tensor):
        raise TypeError(
            'a must be of type ht.tensor, but was {}'.format(type(a)))

    # set default value for axes permutations
    dimensions = len(a.shape)
    if axes is None:
        axes = tuple(reversed(range(dimensions)))
    # if given, sanitize the input
    else:
        try:
            # convert to a list to allow index access
            axes = list(axes)
        except TypeError:
            raise ValueError('axes must be an iterable containing ints')

        if len(axes) != dimensions:
            raise ValueError('axes do not match tensor shape')
        for index, axis in enumerate(axes):
            if not isinstance(axis, int):
                raise TypeError(
                    'axis must be an integer, but was {}'.format(type(axis)))
            elif axis < 0:
                axes[index] = axis + dimensions

    # infer the new split axis, it is the position of the split axis within the new axes permutation
    try:
        transposed_split = axes.index(a.split) if a.split is not None else None
    except ValueError:
        raise ValueError('axes do not match tensor shape')

    # try to rearrange the tensor and return a new transposed variant
    try:
        transposed_data = a._tensor__array.permute(*axes)
        transposed_shape = tuple(a.shape[axis] for axis in axes)

        return tensor.tensor(transposed_data, transposed_shape, a.dtype, transposed_split, a.device, a.comm)
    # if not possible re- raise any torch exception as ValueError
    except RuntimeError as exception:
        raise ValueError(str(exception))


# statically allocated index slices for non-iterable dimensions in triangular operations
__index_base = (slice(None), slice(None),)


def __tri_op(m, k, op):
    """
    Generic implementation of triangle operations on tensors. It takes care of input sanitation and non-standard
    broadcast behavior of the 2D triangle-operators.

    Parameters
    ----------
    m : ht.tensor
        Input tensor for which to compute the triangle operator.
    k : int, optional
        Diagonal above which to apply the triangle operator, k<0 is below and k>0 is above.
    op : callable
        Implementation of the triangle operator.

    Returns
    -------
    triangle_tensor : ht.tensor
        Tensor with the applied triangle operation

    Raises
    ------
    TypeError
        If the input is not a tensor or the diagonal offset cannot be converted to an integral value.
    """
    if not isinstance(m, tensor.tensor):
        raise TypeError('Expected m to be a tensor but was {}'.format(type(m)))

    try:
        k = int(k)
    except ValueError:
        raise TypeError(
            'Expected k to be integral, but was {}'.format(type(k)))

    # chunk the global shape of the tensor to obtain the offset compared to the other ranks
    offset, _, _ = m.comm.chunk(m.shape, m.split)
    dimensions = len(m.shape)

    # manually repeat the input for vectors
    if dimensions == 1:
        triangle = op(m._tensor__array.expand(m.shape[0], -1), k - offset)
        return tensor.tensor(
            triangle,
            (m.shape[0], m.shape[0],),
            m.dtype,
            None if m.split is None else 1,
            m.device,
            m.comm
        )

    original = m._tensor__array
    output = original.clone()

    # modify k to account for tensor splits
    if m.split is not None:
        if m.split + 1 == dimensions - 1:
            k += offset
        elif m.split == dimensions - 1:
            k -= offset

    # in case of two dimensions we can just forward the call to the callable
    if dimensions == 2:
        op(original, k, out=output)
    # more than two dimensions: iterate over all but the last two to realize 2D broadcasting
    else:
        ranges = [range(elements) for elements in m.lshape[:-2]]
        for partial_index in itertools.product(*ranges):
            index = partial_index + __index_base
            op(original[index], k, out=output[index])

    return tensor.tensor(output, m.shape, m.dtype, m.split, m.device, m.comm)


def tril(m, k=0):
    """
    Returns the lower triangular part of the tensor, the other elements of the result tensor are set to 0.

    The lower triangular part of the tensor is defined as the elements on and below the diagonal.

    The argument k controls which diagonal to consider. If k=0, all elements on and below the main diagonal are
    retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
    value excludes just as many diagonals below the main diagonal.

    Parameters
    ----------
    m : ht.tensor
        Input tensor for which to compute the lower triangle.
    k : int, optional
        Diagonal above which to zero elements. k=0 (default) is the main diagonal, k<0 is below and k>0 is above.

    Returns
    -------
    lower_triangle : ht.tensor
        Lower triangle of the input tensor.
    """
    return __tri_op(m, k, torch.tril)


def triu(m, k=0):
    """
    Returns the upper triangular part of the tensor, the other elements of the result tensor are set to 0.

    The upper triangular part of the tensor is defined as the elements on and below the diagonal.

    The argument k controls which diagonal to consider. If k=0, all elements on and below the main diagonal are
    retained. A positive value includes just as many diagonals above the main diagonal, and similarly a negative
    value excludes just as many diagonals below the main diagonal.

    Parameters
    ----------
    m : ht.tensor
        Input tensor for which to compute the upper triangle.
    k : int, optional
        Diagonal above which to zero elements. k=0 (default) is the main diagonal, k<0 is below and k>0 is above.

    Returns
    -------
    upper_triangle : ht.tensor
        Upper triangle of the input tensor.
    """
    return __tri_op(m, k, torch.triu)


def __local_operation(operation, x, out):
    """
    Generic wrapper for local operations, which do not require communication. Accepts the actual operation function as
    argument and takes only care of buffer allocation/writing.

    Parameters
    ----------
    operation : function
        A function implementing the element-wise local operation, e.g. torch.sqrt
    x : ht.tensor
        The value for which to compute 'operation'.
    out : ht.tensor or None
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Returns
    -------
    result : ht.tensor
        A tensor of the same shape as x, containing the result of 'operation' for each element in x. If out was
        provided, result is a reference to it.

    Raises
    -------
    TypeError
        If the input is not a tensor or the output is not a tensor or None.
    """
    # perform sanitation
    if not isinstance(x, tensor.tensor):
        raise TypeError('expected x to be a ht.tensor, but was {}'.format(type(x)))
    if out is not None and not isinstance(out, tensor.tensor):
        raise TypeError('expected out to be None or an ht.tensor, but was {}'.format(type(out)))

    # infer the output type of the tensor
    # we need floating point numbers here, due to PyTorch only providing sqrt() implementation for float32/64
    promoted_type = types.promote_types(x.dtype, types.float32)
    torch_type = promoted_type.torch_type()

    # no defined output tensor, return a freshly created one
    if out is None:
        result = operation(x._tensor__array.type(torch_type))
        return tensor.tensor(result, x.gshape, promoted_type, x.split, x.device, x.comm)

    # output buffer writing requires a bit more work
    # we need to determine whether the operands are broadcastable and the multiple of the broadcasting
    # reason: manually repetition for each dimension as PyTorch does not conform to numpy's broadcast semantic
    # PyTorch always recreates the input shape and ignores broadcasting/too large buffers
    broadcast_shape = stride_tricks.broadcast_shape(x.lshape, out.lshape)
    padded_shape = (1,) * (len(broadcast_shape) - len(x.lshape)) + x.lshape
    multiples = [int(a / b) for a, b in zip(broadcast_shape, padded_shape)]
    needs_repetition = any(multiple > 1 for multiple in multiples)

    # do an inplace operation into a provided buffer
    casted = x._tensor__array.type(torch_type)
    operation(casted.repeat(multiples)
              if needs_repetition else casted, out=out._tensor__array)
    return out


def __reduce_op(x, partial_op, reduction_op, axis, out):
    # TODO: document me Issue #102
    # perform sanitation
    if not isinstance(x, tensor.tensor):
        raise TypeError(
            'expected x to be a ht.tensor, but was {}'.format(type(x)))
    if out is not None and not isinstance(out, tensor.tensor):
        raise TypeError(
            'expected out to be None or an ht.tensor, but was {}'.format(type(out)))

    # no further checking needed, sanitize axis will raise the proper exceptions
    axis = stride_tricks.sanitize_axis(x.shape, axis)
    split = x.split

    if axis is None:
        partial = partial_op(x._tensor__array).reshape((1,))
        output_shape = (1,)
    else:
        partial = partial_op(x._tensor__array, axis, keepdim=True)
        output_shape = x.gshape[:axis] + (1,) + x.gshape[axis + 1:]

    # Check shape of output buffer, if any
    if out is not None and out.shape != output_shape:
        raise ValueError('Expecting output buffer of shape {}, got {}'.format(
            output_shape, out.shape))

    # perform a reduction operation in case the tensor is distributed across the reduction axis
    if x.split is not None and (axis is None or axis == x.split):
        split = None
        if x.comm.is_distributed():
            x.comm.Allreduce(MPI.IN_PLACE, partial[0], reduction_op)

    # if reduction_op is a Boolean operation, then resulting tensor is bool
    boolean_ops = [MPI.LAND, MPI.LOR, MPI.BAND, MPI.BOR]
    tensor_type = bool if reduction_op in boolean_ops else partial[0].dtype

    if out is not None:
        out._tensor__array = partial
        out._tensor__dtype = types.canonical_heat_type(tensor_type)
        out._tensor__split = split
        out._tensor__device = x.device
        out._tensor__comm = x.comm

        return out

    return tensor.tensor(
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
    result: ht.tensor
    A tensor containing the results of element-wise operation.
    """

    if np.isscalar(t1):

        try:
            t1 = tensor.array([t1])
        except (ValueError, TypeError,):
            raise TypeError('Data type not supported, input was {}'.format(type(t1)))

        if np.isscalar(t2):
            try:
                t2 = tensor.array([t2])
            except (ValueError, TypeError,):
                raise TypeError('Only numeric scalars are supported, but input was {}'.format(type(t2)))

            output_shape = (1,)
            output_split = None
            output_device = None
            output_comm = None

        elif isinstance(t2, tensor.tensor):
            output_shape = t2.shape
            output_split = t2.split
            output_device = t2.device
            output_comm = t2.comm

        else:
            raise TypeError('Only tensors and numeric scalars are supported, but input was {}'.format(type(t2)))

        if t1.dtype != t2.dtype:
            t1 = t1.astype(t2.dtype)


    elif isinstance(t1, tensor.tensor):

        if np.isscalar(t2):
            try:
                t2 = tensor.array([t2])
            except (ValueError, TypeError,):
                raise TypeError('Data type not supported, input was {}'.format(type(t2)))

        elif isinstance(t2, tensor.tensor):

            output_shape = stride_tricks.broadcast_shape(t1.shape, t2.shape)

            # TODO: implement complex NUMPY rules
            if t2.split is None or t2.split == t1.split:
                pass

            else:
                # It is NOT possible to perform binary operations on tensors with different splits, e.g. split=0 and split=1
                raise NotImplementedError('Not implemented for other splittings')


        else:
            raise TypeError('Only tensors and numeric scalars are supported, but input was {}'.format(type(t2)))

        if t2.dtype != t1.dtype:
            t2 = t2.astype(t1.dtype)

        output_shape = t1.shape
        output_split = t1.split
        output_device = t1.device
        output_comm = t1.comm


    else:
        raise NotImplementedError('Not implemented for non scalar')

    result = operation(t1._tensor__array, t2._tensor__array)

    return tensor.tensor(result, output_shape, t1.dtype, output_split, output_device, output_comm)

