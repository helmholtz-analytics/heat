import itertools
import torch

from .communication import MPI
from . import stride_tricks
from . import types
from . import tensor

__all__ = [
    'abs',
    'absolute',
    'all',
    'argmin',
    'clip',
    'copy',
    'exp',
    'floor',
    'log',
    'max',
    'min',
    'sin',
    'sqrt',
    'sum',
    'tril',
    'triu'
]


def abs(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : ht.tensor
        The values for which the compute the absolute value.
    out : ht.tensor, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.tensor
        A tensor containing the absolute value of each element in x.
    """
    if dtype is not None and not issubclass(dtype, types.generic):
        raise TypeError('dtype must be a heat data type')

    absolute_values = __local_operation(torch.abs, x, out)
    if dtype is not None:
        absolute_values._tensor__array = absolute_values._tensor__array.type(
            dtype.torch_type())
        absolute_values._tensor__dtype = dtype

    return absolute_values


def absolute(x, out=None, dtype=None):
    """
    Calculate the absolute value element-wise.

    np.abs is a shorthand for this function.

    Parameters
    ----------
    x : ht.tensor
        The values for which the compute the absolute value.
    out : ht.tensor, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated array is returned.
    dtype : ht.type, optional
        Determines the data type of the output array. The values are cast to this type with potential loss of
        precision.

    Returns
    -------
    absolute_values : ht.tensor
        A tensor containing the absolute value of each element in x.
    """
    return abs(x, out, dtype)


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
    >>> a = ht.random.randn(4,5)
    >>> a
    tensor([[ 0.5370, -0.4117, -3.1062,  0.4897, -0.3231],
            [-0.5005, -1.7746,  0.8515, -0.9494, -0.2238],
            [-0.0444,  0.3388,  0.6805, -1.3856,  0.5422],
            [ 0.3184,  0.0185,  0.5256, -1.1653, -0.1665]])
    >>> x = a<0.5
    >>> x
    tensor([[0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [1,1, 0, 1, 1]], dtype=torch.uint8)
    >>> ht.all(x)
    tensor([0], dtype=torch.uint8)
    >>> ht.all(x, axis=0)
    tensor([[0, 1, 0, 1, 0]], dtype=torch.uint8)
    >>> ht.all(x, axis=1)
    tensor([[0],
            [0],
            [0],
            [0]], dtype=torch.uint8)

    Write out to predefined buffer:
    >>> out = ht.zeros((1,5))
    >>> ht.all(x,axis=0,out=out)
    >>> out
    tensor([[0, 1, 0, 1, 0]], dtype=torch.uint8)
    """
    # TODO: make me more numpy API complete. Issue #101

    sum_of_ones = __reduce_op((x == 1), torch.sum, MPI.SUM, axis, out=None)

    # calculate shape of output
    if axis is None:
        numel = x._tensor__array.numel()
        output_shape = (1,)
    else:
        numel = x.gshape[axis]
        output_shape = x.gshape[:axis] + (1,) + x.gshape[axis + 1:]

    if out is not None:
        # perform sanitation
        if not isinstance(out, tensor.tensor):
            raise TypeError(
                'expected out to be None or an ht.tensor, but was {}'.format(type(out)))
        out = stride_tricks.sanitize_out(out, output_shape)
        # write to "out"
        out._tensor__array = tensor.tensor((sum_of_ones == numel), output_shape,
                                           types.canonical_heat_type(sum_of_ones.dtype), split=out.split, comm=x.comm)
        return out
    # TODO: distributed calculation of (sum_of_ones == numel) Issue #99
    return tensor.tensor((sum_of_ones == numel), output_shape,
                         types.canonical_heat_type(sum_of_ones.dtype), split=sum_of_ones.split, comm=x.comm)


def argmin(x, axis=None, out=None):
    '''
    Returns the indices of the minimum values along an axis.

    Parameters:
    ----------

    x : ht.tensor
    Input array.

    axis : int, optional
    By default, the index is into the flattened tensor, otherwise along the specified axis.

    out : ht.tensor, optional.
    If provided, the result will be inserted into this tensor. Must be of the same shape
    and buffer length as the expected output.


    Returns:
    -------

    index_tensor : ht.tensor of ints
    Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

    Examples:
    --------

    >>> a = ht.random.randn(3,3)
    >>> a
    tensor([[ 0.2604, -0.2292,  1.2846],
        [-0.6166,  0.4549, -0.1443],
        [-1.0306, -1.1125,  0.5686]])
    >>> ht.argmin(a)
    tensor([7])
    >>> ht.argmin(a, axis=0)
    tensor([[2, 2, 1]])
    >>> ht.argmin(a, axis=1)
    tensor([[1],
        [0],
        [1]])
    >>> out = ht.zeros((1, 3))
    >>> ht.argmin(a, axis=0, out=out)
    >>> out
    tensor([[2, 2, 1]])

    '''
    partial_op = torch.min if x.comm.is_distributed() else torch.argmin
    if axis is None:
        # flatten tensor
        x_flat_torch = torch.reshape(
            x._tensor__array, (x._tensor__array.numel(),))
        x_flat = tensor.tensor(x_flat_torch, x_flat_torch.shape,
                               types.canonical_heat_type(x.dtype), x.split, x.comm)
        return __reduce_op(x_flat, partial_op, MPI.MINLOC, axis=0, out=out)
    else:
        return __reduce_op(x, partial_op, MPI.MINLOC, axis, out=out)


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
        return tensor.tensor(a._tensor__array.clamp(a_min, a_max), a.shape, a.dtype, a.split, a.comm)
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
    return tensor.tensor(a._tensor__array.clone(), a.shape, a.dtype, a.split, a.comm)


def exp(x, out=None):
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the exponential.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    exponentials : ht.tensor
        A tensor of the same shape as x, containing the positive exponentials of each element in this tensor. If out
        was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.exp(ht.arange(5))
    tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981])
    """
    return __local_operation(torch.exp, x, out)


def floor(x, out=None):
    """
    Return the floor of the input, element-wise.

    The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lfloor x \rfloor.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the floored values.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    floored : ht.tensor
        A tensor of the same shape as x, containing the floored valued of each element in this tensor. If out was
        provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.floor(ht.arange(-2.0, 2.0, 0.4))
    tensor([-2., -2., -2., -1., -1.,  0.,  0.,  0.,  1.,  1.])
    """
    return __local_operation(torch.floor, x, out)


def log(x, out=None):
    """
    Natural logarithm, element-wise.

    The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x. The natural
    logarithm is logarithm in base e.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the logarithm.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    logarithms : ht.tensor
        A tensor of the same shape as x, containing the positive logarithms of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, logarithms is a reference to it.

    Examples
    --------
    >>> ht.log(ht.arange(5))
    tensor([  -inf, 0.0000, 0.6931, 1.0986, 1.3863])
    """
    return __local_operation(torch.log, x, out)


def max(x, axis=None, out=None):
    """"
    Return a tuple containing:
        - the maximum of an array or maximum along an axis;
        - indices of maxima

    Parameters
    ----------
    a : ht.tensor
    Input data.

    axis : None or int, optional
    Axis or axes along which to operate. By default, flattened input is used.

    out : ht.tensor, optional
    Tuple of two output tensors (max, max_indices). Must be of the same shape and buffer length as the expected output.

    # TODO: initial : scalar, optional Issue #101
    The minimum value of an output element. Must be present to allow computation on empty slice.

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
    (tensor([[10., 11., 12.]]), tensor([[3, 3, 3]]))
    >>> ht.min(a, axis=1)
    (tensor([[ 3.],
        [ 6.],
        [ 9.],
        [12.]]), tensor([[2],
        [2],
        [2],
        [2]]))
    """
    return __reduce_op(x, torch.max, MPI.MAX, axis, out)


def min(x, axis=None, out=None):
    """"
    Return a tuple containing:
        - the minimum of an array or minimum along an axis;
        - indices of minima

    Parameters
    ----------
    a : ht.tensor
    Input data.

    axis : None or int
    Axis or axes along which to operate. By default, flattened input is used.


    out : ht.tensor, optional
    Tuple of two output tensors (min, min_indices). Must be of the same shape and buffer length as the expected output.


    # TODO: initial : scalar, optional Issue #101
    The maximum value of an output element. Must be present to allow computation on empty slice.

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
    (tensor([[1., 2., 3.]]), tensor([[0, 0, 0]]))
    >>> ht.min(a, axis=1)
    (tensor([[ 1.],
        [ 4.],
        [ 7.],
        [10.]]), tensor([[0],
        [0],
        [0],
        [0]]))
    """
    return __reduce_op(x, torch.min, MPI.MIN, axis, out)


def sin(x, out=None):
    """
    Return the trigonometric sine, element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the trigonometric sine.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided
        or set to None, a fresh tensor is allocated.

    Returns
    -------
    sine : ht.tensor
        A tensor of the same shape as x, containing the trigonometric sine of each element in this tensor.
        Negative input elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sin(ht.arange(-6, 7, 2))
    tensor([ 0.2794,  0.7568, -0.9093,  0.0000,  0.9093, -0.7568, -0.2794])
    """
    return __local_operation(torch.sin, x, out)


def sqrt(x, out=None):
    """
    Return the non-negative square-root of a tensor element-wise.

    Parameters
    ----------
    x : ht.tensor
        The value for which to compute the square-roots.
    out : ht.tensor or None, optional
        A location in which to store the results. If provided, it must have a broadcastable shape. If not provided or
        set to None, a fresh tensor is allocated.

    Returns
    -------
    square_roots : ht.tensor
        A tensor of the same shape as x, containing the positive square-root of each element in x. Negative input
        elements are returned as nan. If out was provided, square_roots is a reference to it.

    Examples
    --------
    >>> ht.sqrt(ht.arange(5))
    tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000])
    >>> ht.sqrt(ht.arange(-5, 0))
    tensor([nan, nan, nan, nan, nan])
    """
    return __local_operation(torch.sqrt, x, out)


def sum(x, axis=None, out=None):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    x : ht.tensor
        Input data.

    axis : None or int, optional
        Axis along which a sum is performed. The default, axis=None, will sum
        all of the elements of the input array. If axis is negative it counts
        from the last to the first axis.

    Returns
    -------
    sum_along_axis : ht.tensor
        An array with the same shape as self.__array except for the specified axis which
        becomes one, e.g. a.shape = (1,2,3) => ht.ones((1,2,3)).sum(axis=1).shape = (1,1,3)

    Examples
    --------
    >>> ht.sum(ht.ones(2))
    tensor([2.])

    >>> ht.sum(ht.ones((3,3)))
    tensor([9.])

    >>> ht.sum(ht.ones((3,3)).astype(ht.int))
    tensor([9])

    >>> ht.sum(ht.ones((3,2,1)), axis=-3)
    tensor([[[3.],
            [3.]]])
    """

    # TODO: make me more numpy API complete Issue #101

    return __reduce_op(x, torch.sum, MPI.SUM, axis, out)


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
        raise TypeError(
            'expected x to be a ht.tensor, but was {}'.format(type(x)))
    if out is not None and not isinstance(out, tensor.tensor):
        raise TypeError(
            'expected out to be None or an ht.tensor, but was {}'.format(type(out)))

    # infer the output type of the tensor
    # we need floating point numbers here, due to PyTorch only providing sqrt() implementation for float32/64
    promoted_type = types.promote_types(x.dtype, types.float32)
    torch_type = promoted_type.torch_type()

    # no defined output tensor, return a freshly created one
    if out is None:
        return tensor.tensor(operation(x._tensor__array.type(torch_type)), x.gshape, promoted_type, x.split, x.comm)

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
        return tensor.tensor(triangle, (m.shape[0], m.shape[0],), m.dtype, None if m.split is None else 1, m.comm)

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

    return tensor.tensor(output, m.shape, m.dtype, m.split, m.comm)


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


def __reduce_op(x, partial_op, op, axis, out):
    # TODO: document me Issue #102

    # perform sanitation of input
    if not isinstance(x, tensor.tensor):
        raise TypeError(
            'expected x to be a ht.tensor, but was {}'.format(type(x)))
    if out is not None and not isinstance(out, tensor.tensor):
        raise TypeError(
            'expected out to be None or an ht.tensor, but was {}'.format(type(out)))

    # no further checking needed, sanitize_axis and sanitize_out will raise the proper exceptions
    axis = stride_tricks.sanitize_axis(x.shape, axis)

    if axis is None:
        partial = torch.reshape(partial_op(x._tensor__array), (1,))
        output_shape = (1,)
    else:
        partial = partial_op(x._tensor__array, axis, keepdim=True)
        output_shape = x.gshape[:axis] + (1,) + x.gshape[axis + 1:]

    # sanitize output buffer, if any
    if out is not None:
        out = stride_tricks.sanitize_out(out, output_shape)

    if x.comm.is_distributed() and (axis is None or axis == x.split):
        if op in [MPI.MINLOC, MPI.MAXLOC]:  # axis is never None here
            offset = x.comm.chunk(x.gshape, x.split)[0]
            partial[1][0] = partial[1][0] + offset
            total = x.comm.allreduce(partial, op)
            if out is not None:
                out._tensor__array = tensor.tensor(total[1], output_shape, types.canonical_heat_type(
                    total[1].dtype), split=out.split, comm=x.comm)
                return out
            return tensor.tensor(total[1], output_shape, types.canonical_heat_type(total[1].dtype), split=None, comm=x.comm)

        x.comm.Allreduce(MPI.IN_PLACE, partial[0], op)
        if out is not None:
            out._tensor__array = tensor.tensor(partial, output_shape, types.canonical_heat_type(
                partial[0].dtype), split=out.split, comm=x.comm)
            return out
        return tensor.tensor(partial, output_shape, types.canonical_heat_type(partial[0].dtype), split=None, comm=x.comm)
    else:
        if out is not None:
            out._tensor__array = tensor.tensor(partial, output_shape, types.canonical_heat_type(
                partial[0].dtype), split=out.split, comm=x.comm)
            return out
        return tensor.tensor(partial, output_shape, types.canonical_heat_type(partial[0].dtype), split=None, comm=x.comm)
