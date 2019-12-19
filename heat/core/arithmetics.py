import torch

from .communication import MPI
from . import dndarray
from . import operations
from . import stride_tricks

__all__ = [
    "add",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "diff",
    "div",
    "divide",
    "floordiv",
    "floor_divide",
    "fmod",
    "mod",
    "mul",
    "multiply",
    "pow",
    "prod",
    "power",
    "remainder",
    "sub",
    "subtract",
    "sum",
]


def add(t1, t2):
    """
    Element-wise addition of values from two operands, commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be added as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand involved in the addition
    t2: tensor or scalar
        The second operand involved in the addition

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the results of element-wise addition of t1 and t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.add(1.0, 4.0)
    tensor([5.])

    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.add(T1, T2)
    tensor([[3., 4.],
            [5., 6.]])

    >>> s = 2.0
    >>> ht.add(T1, s)
    tensor([[3., 4.],
            [5., 6.]])

    """
    return operations.__binary_op(torch.add, t1, t2)


def bitwise_and(t1, t2):
    """
    Compute the bit-wise AND of two arrays element-wise.

    Parameters
    ----------
    t1, t2: tensor or scalar
        Only integer and boolean types are handled. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the results of element-wise AND of t1 and t2.

    Examples:
    ---------
    import heat as ht
    >>> ht.bitwise_and(13, 17)
    tensor([1])
    >>> np.bitwise_and(14, 13)
    tensor([12])

    >>> ht.bitwise_and(ht.array([14,3]), 13)
    tensor([12,  1])

    >>> ht.bitwise_and(ht.array([11,7]), ht.array([4,25]))
    tensor([0, 1])
    >>> ht.bitwise_and(ht.array([2,5,255]), ht.array([3,14,16]))
    tensor([ 2,  4, 16])

    >>> ht.bitwise_and(ht.array([True, True]), ht.array([False, True]))
    tensor([False,  True])
    """
    return operations.__binary_bit_op("__and__", t1, t2)


def bitwise_or(t1, t2):
    """
    Compute the bit-wise OR of two arrays element-wise.

    Parameters
    ----------
    t1, t2: tensor or scalar
       Only integer and boolean types are handled. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Returns
    -------
    result: ht.DNDArray
       A tensor containing the results of element-wise OR of t1 and t2.

    Examples:
    ---------
    import heat as ht
    >>> ht.bitwise_or(13, 16)
    tensor([29])

    >>> ht.bitwise_or(32, 2)
    tensor([34])
    >>> ht.bitwise_or(ht.array([33, 4]), 1)
    tensor([33,  5])
    >>> ht.bitwise_or(ht.array([33, 4]), ht.array([1, 2]))
    tensor([33,  6])

    >>> ht.bitwise_or(ht.array([2, 5, 255]), ht.array([4, 4, 4]))
    tensor([  6,   5, 255])
    >>> ht.bitwise_or(ht.array([2, 5, 255, 2147483647], dtype=ht.int32),
    ...               ht.array([4, 4, 4, 2147483647], dtype=ht.int32))
    tensor([         6,          5,        255, 2147483647])
    >>> ht.bitwise_or(ht.array([True, True]), ht.array([False, True]))
    tensor([ True,  True])
    """
    return operations.__binary_bit_op("__or__", t1, t2)


def bitwise_xor(t1, t2):
    """
    Compute the bit-wise XOR of two arrays element-wise.

    Parameters
    ----------
    t1, t2: tensor or scalar
       Only integer and boolean types are handled. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Returns
    -------
    result: ht.DNDArray
       A tensor containing the results of element-wise OR of t1 and t2.

    Examples:
    ---------
    import heat as ht
    >>> ht.bitwise_xor(13, 17)
    tensor([28])

    >>> ht.bitwise_xor(31, 5)
    tensor([26])
    >>> ht.bitwise_xor(ht.array[31,3], 5)
    tensor([26,  6])

    >>> ht.bitwise_xor(ht.array([31,3]), ht.array([5,6]))
    tensor([26,  5])
    >>> ht.bitwise_xor(ht.array([True, True]), ht.array([False, True]))
    tensor([ True, False])
    """
    return operations.__binary_bit_op("__xor__", t1, t2)


def diff(a, n=1, axis=-1):
    """
    Calculate the n-th discrete difference along the given axis.
    The first difference is given by out[i] = a[i+1] - a[i] along the given axis, higher differences are calculated by using diff recursively.

    a : DNDarray
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
        Default value is 1
        n=2 is equivalent to ht.diff(ht.diff(a))
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.

    Returns
    -------
    diff : DNDarray
        The n-th differences. The shape of the output is the same as a except along axis where the dimension is smaller by n.
        The type of the output is the same as the type of the difference between any two elements of a.
        The split does not change. The outpot array is balanced.
    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError("diff requires that n be a positive number, got {}".format(n))
    if not isinstance(a, dndarray.DNDarray):
        raise TypeError("'a' must be a DNDarray")

    axis = stride_tricks.sanitize_axis(a.gshape, axis)

    if not a.is_distributed():
        ret = a.copy()
        for _ in range(n):
            axis_slice = [slice(None)] * len(ret.shape)
            axis_slice[axis] = slice(1, None, None)
            axis_slice_end = [slice(None)] * len(ret.shape)
            axis_slice_end[axis] = slice(None, -1, None)
            ret = ret[axis_slice] - ret[axis_slice_end]
        return ret

    size = a.comm.size
    rank = a.comm.rank
    ret = a.copy()
    # work loop, runs n times. using the result at the end of the loop as the starting values for each loop
    for _ in range(n):
        axis_slice = [slice(None)] * len(ret.shape)
        axis_slice[axis] = slice(1, None, None)
        axis_slice_end = [slice(None)] * len(ret.shape)
        axis_slice_end[axis] = slice(None, -1, None)

        # build the slice for the first element on the specified axis
        arb_slice = [slice(None)] * len(a.shape)
        arb_slice[axis] = 0
        # send the first element of the array to rank - 1
        if rank > 0:
            snd = ret.comm.Isend(ret.lloc[arb_slice].clone(), dest=rank - 1, tag=rank)

        # standard logic for the diff with the next element
        dif = ret.lloc[axis_slice] - ret.lloc[axis_slice_end]
        # need to slice out to select the proper elements of out
        diff_slice = [slice(x) for x in dif.shape]
        ret.lloc[diff_slice] = dif

        if rank > 0:
            snd.wait()  # wait for the send to finish
        if rank < size - 1:
            cr_slice = [slice(None)] * len(a.shape)
            # slice of 1 element in the selected axis for the shape creation
            cr_slice[axis] = 1
            recv_data = torch.ones(
                ret.lloc[cr_slice].shape, dtype=ret.dtype.torch_type(), device=a.device.torch_device
            )
            rec = ret.comm.Irecv(recv_data, source=rank + 1, tag=rank + 1)
            axis_slice_end = [slice(None)] * len(a.shape)
            # select the last elements in the selected axis
            axis_slice_end[axis] = slice(-1, None)
            rec.wait()
            # diff logic
            ret.lloc[axis_slice_end] = (
                recv_data.reshape(ret.lloc[axis_slice_end].shape) - ret.lloc[axis_slice_end]
            )

    axis_slice_end = [slice(None)] * len(a.shape)
    axis_slice_end[axis] = slice(None, -1 * n, None)
    ret = ret[axis_slice_end]  # slice of the last element on the array (nonsense data)
    ret.balance_()  # balance the array before returning
    return ret


def div(t1, t2):
    """
    Element-wise true division of values of operand t1 by values of operands t2 (i.e t1 / t2), not commutative.
    Takes the two operands (scalar or tensor) whose elements are to be divided (operand 1 by operand 2)
    as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand whose values are divided
    t2: tensor or scalar
        The second operand by whose values is divided

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the results of element-wise true division (i.e. floating point values) of t1 by t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.div(2.0, 2.0)
    tensor([1.])

    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.div(T1, T2)
    tensor([[0.5000, 1.0000],
            [1.5000, 2.0000]])

    >>> s = 2.0
    >>> ht.div(s, T1)
    tensor([[2.0000, 1.0000],
            [0.6667, 0.5000]])
    """
    return operations.__binary_op(torch.div, t1, t2)


# Alias in compliance with numpy API
divide = div


def fmod(t1, t2):
    """
    Element-wise division remainder of values of operand t1 by values of operand t2 (i.e. C Library function fmod), not commutative.
    Takes the two operands (scalar or tensor, both may contain floating point number) whose elements are to be
    divided (operand 1 by operand 2) as arguments.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand whose values are divided (may be floats)
    t2: tensor or scalar
        The second operand by whose values is divided (may be floats)

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the remainder of the element-wise division (i.e. floating point values) of t1 by t2.
        It has the sign as the dividend t1.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.fmod(2.0, 2.0)
    tensor([0.])

    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.fmod(T1, T2)
    tensor([[1., 0.],
            [1., 0.]])

    >>> s = 2.0
    >>> ht.fmod(s, T1)
    tensor([[0., 0.]
            [2., 2.]])
    """
    return operations.__binary_op(torch.fmod, t1, t2)


def floordiv(t1, t2):
    """
    Element-wise floor division of value of operand t1 by values of operands t2 (i.e. t1 // t2), not commutative.
    Takes the two operands (scalar or tensor) whose elements are to be divided (operand 1 by operand 2) as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand whose values are divided
    t2: tensor or scalar
        The second operand by whose values is divided

    Return
    ------
    result: ht.DNDarray
        A tensor containing the results of element-wise floor division (integer values) of t1 by t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> T1 = ht.float32([[1.7, 2.0], [1.9, 4.2]])
    >>> ht.floordiv(T1, 1)
    tensor([[1., 2.],
            [1., 4.]])
    >>> T2 = ht.float32([1.5, 2.5])
    >>> ht.floordiv(T1, T2)
    tensor([[1., 0.],
            [1., 1.]])
    """
    return operations.__binary_op(lambda a, b: torch.div(a, b).floor(), t1, t2)


# Alias in compliance with numpy API
floor_divide = floordiv


def mod(t1, t2):
    """
    Element-wise division remainder of values of operand t1 by values of operand t2 (i.e. t1 % t2), not commutative.
    Takes the two operands (scalar or tensor) whose elements are to be divided (operand 1 by operand 2) as arguments.

    Currently t1 and t2 are just passed to remainder.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand whose values are divided
    t2: tensor or scalar
        The second operand by whose values is divided

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the remainder of the element-wise division of t1 by t2.
        It has the same sign as the devisor t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.mod(2, 2)
    tensor([0])

    >>> T1 = ht.int32([[1, 2], [3, 4]])
    >>> T2 = ht.int32([[2, 2], [2, 2]])
    >>> ht.mod(T1, T2)
    tensor([[1, 0],
            [1, 0]], dtype=torch.int32)

    >>> s = 2
    >>> ht.mod(s, T1)
    tensor([[0, 0]
            [2, 2]], dtype=torch.int32)
    """
    return remainder(t1, t2)


def mul(t1, t2):
    """
    Element-wise multiplication (NOT matrix multiplication) of values from two operands, commutative.
    Takes the first and second operand (scalar or tensor) whose elements are to be multiplied as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand involved in the multiplication
    t2: tensor or scalar
        The second operand involved in the multiplication

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the results of element-wise multiplication of t1 and t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.mul(2.0, 4.0)
    tensor([8.])

    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> s = 3.0
    >>> ht.mul(T1, s)
    tensor([[3., 6.],
            [9., 12.]])

    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.mul(T1, T2)
    tensor([[2., 4.],
            [6., 8.]])

    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.mul(T1, T2)
    tensor([[2., 4.],
            [6., 8.]])
    """
    return operations.__binary_op(torch.mul, t1, t2)


# Alias in compliance with numpy API
multiply = mul


def pow(t1, t2):
    """
    Element-wise exponential function of values of operand t1 to the power of values of operand t2 (i.e t1 ** t2),
    not commutative. Takes the two operands (scalar or tensor) whose elements are to be involved in the exponential
    function(operand 1 to the power of operand 2)
    as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand whose values represent the base
    t2: tensor or scalar
        The second operand by whose values represent the exponent

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the results of element-wise exponential function.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.pow (3.0, 2.0)
    tensor([9.])

    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[3, 3], [2, 2]])
    >>> ht.pow(T1, T2)
    tensor([[1., 8.],
            [9., 16.]])
    >>> s = 3.0
    >>> ht.pow(T1, s)
    tensor([[1., 8.],
            [27., 64.]])
    """
    return operations.__binary_op(torch.pow, t1, t2)


# Alias in compliance with numpy API
power = pow


def remainder(t1, t2):
    """
    Element-wise division remainder of values of operand t1 by values of operand t2 (i.e. t1 % t2), not commutative.
    Takes the two operands (scalar or tensor) whose elements are to be divided (operand 1 by operand 2) as arguments.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand whose values are divided
    t2: tensor or scalar
        The second operand by whose values is divided

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the remainder of the element-wise division of t1 by t2.
        It has the same sign as the devisor t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.mod(2, 2)
    tensor([0])

    >>> T1 = ht.int32([[1, 2], [3, 4]])
    >>> T2 = ht.int32([[2, 2], [2, 2]])
    >>> ht.mod(T1, T2)
    tensor([[1, 0],
            [1, 0]], dtype=torch.int32)

    >>> s = 2
    >>> ht.mod(s, T1)
    tensor([[0, 0]
            [2, 2]], dtype=torch.int32)
    """
    return operations.__binary_op(torch.remainder, t1, t2)


def prod(x, axis=None, out=None, keepdim=None):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    x : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed. The default, axis=None, will calculate the product of all the
        elements in the input array. If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a product is performed on all of the axes specified in the tuple instead of a single
        axis or all the axes as before.
    out : ndarray, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output, but
        the type of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

    Returns
    -------
    product_along_axis : ht.DNDarray
        An array shaped as a but with the specified axis removed. Returns a reference to out if specified.

    Examples
    --------
    >>> import heat as ht
    >>> ht.prod([1.,2.])
    ht.tensor([2.0])

    >>> ht.prod([
        [1.,2.],
        [3.,4.]
    ])
    ht.tensor([24.0])

    >>> ht.prod([
        [1.,2.],
        [3.,4.]
    ], axis=1)
    ht.tensor([  2.,  12.])
    """
    return operations.__reduce_op(x, torch.prod, MPI.PROD, axis=axis, out=out, keepdim=keepdim)


def sub(t1, t2):
    """
    Element-wise subtraction of values of operand t2 from values of operands t1 (i.e t1 - t2), not commutative.
    Takes the two operands (scalar or tensor) whose elements are to be subtracted (operand 2 from operand 1)
    as argument.

    Parameters
    ----------
    t1: tensor or scalar
        The first operand from which values are subtracted
    t2: tensor or scalar
        The second operand whose values are subtracted

    Returns
    -------
    result: ht.DNDarray
        A tensor containing the results of element-wise subtraction of t1 and t2.

    Examples:
    ---------
    >>> import heat as ht
    >>> ht.sub(4.0, 1.0)
    tensor([3.])

    >>> T1 = ht.float32([[1, 2], [3, 4]])
    >>> T2 = ht.float32([[2, 2], [2, 2]])
    >>> ht.sub(T1, T2)
    tensor([[-1., 0.],
            [1., 2.]])

    >>> s = 2.0
    >>> ht.sub(s, T1)
    tensor([[ 1.,  0.],
            [-1., -2.]])
    """
    return operations.__binary_op(torch.sub, t1, t2)


# Alias in compliance with numpy API
subtract = sub


def sum(x, axis=None, out=None, keepdim=None):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    x : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis along which a sum is performed. The default, axis=None, will sum
        all of the elements of the input array. If axis is negative it counts
        from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes specified
        in the tuple instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output, but
        the type of the output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

    Returns
    -------
    sum_along_axis : ht.DNDarray
        An array with the same shape as self.__array except for the specified axis which
        becomes one, e.g. a.shape = (1, 2, 3) => ht.ones((1, 2, 3)).sum(axis=1).shape = (1, 1, 3)

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
    return operations.__reduce_op(x, torch.sum, MPI.SUM, axis=axis, out=out, keepdim=keepdim)
