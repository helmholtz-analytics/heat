import itertools
import torch
import warnings
import numpy as np

from .communication import MPI
from . import stride_tricks
from . import types
from . import tensor

__all__ = [
    'abs',
    'absolute',
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


def argmin(x, axis=None):
    '''
    Returns the indices of the minimum values along an axis.

    Parameters:
    ----------

    x : ht.tensor
    Input array.

    axis : int, optional
    By default, the index is into the flattened tensor, otherwise along the specified axis.

    # TODO out : array, optional
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
    '''

    if axis is None:
        # TEMPORARY SOLUTION! TODO: implementation for axis=None, distributed tensor
        # perform sanitation
        if not isinstance(x, tensor.tensor):
            raise TypeError(
                'expected x to be a ht.tensor, but was {}'.format(type(x)))
        axis = stride_tricks.sanitize_axis(x.shape, axis)
        out = torch.reshape(torch.argmin(x._tensor__array), (1,))
        return tensor.tensor(out, out.shape, types.canonical_heat_type(out.dtype), split=None, comm=x.comm)

    out = __reduce_op(x, torch.min, MPI.MIN, axis)._tensor__array[1]
    return tensor.tensor(out, out.shape, types.canonical_heat_type(out.dtype), x._tensor__split, comm=x.comm)


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


def max(x, axis=None):
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

    # TODO: out : ht.tensor, optional
    Tuple of two output tensors (max, max_indices). Must be of the same shape and buffer length as the expected output.

    # TODO: initial : scalar, optional
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
    return __reduce_op(x, torch.max, MPI.MAX, axis)


def min(x, axis=None):
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


    # TODO: out : ht.tensor, optional
    Tuple of two output tensors (min, min_indices). Must be of the same shape and buffer length as the expected output.


    # TODO: initial : scalar, optional
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
    return __reduce_op(x, torch.min, MPI.MIN, axis)


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


# def sum(x, axis=None):
#     # TODO: document me
#     axis = stride_tricks.sanitize_axis(x.shape, axis)
#     if axis is not None:
#         sum_axis = x._tensor__array.sum(axis, keepdim=True)
#     else:
#         sum_axis = torch.reshape(x._tensor__array.sum(), (1,))
#         if not x.comm.is_distributed():
#             return tensor.tensor(sum_axis, (1,), types.canonical_heat_type(sum_axis.dtype), None, x.comm)

#     return __reduce_op(x, sum_axis, MPI.SUM, axis)


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


def sum(x, axis=None):
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

    # TODO: make me more numpy API complete

    return __reduce_op(x, torch.sum, MPI.SUM, axis)


def moments(x, axis=None, direction=None,  continuous_axes=False, merge_axes=False):
    """
    Find the central moment of a data set
    m=1 is the true mean (not 0)

    Parameters
    ----------
    x : ht.tensor
        Input data

    m : int
        Moment to be calculated (0 = 1, 1 = mean, 2 = var (std^2), 3 = skewness, 4 = kurtosis)

    axis : None, int, multiple (1D list, 1D array, 1D tensor, 1D tuple), optional
           axis along which to find the central moment (negative counts from end to 0
           If None:
               calculate the moments for all elements along the specified axis
           if int / multiple:
               calculate the moment/s of the selected elements along the specified axis

    direction : None, int
          direction for which the moment is calculated in (i.e. row or column)
          Default = 0 (column)

    continuous_axes: bool
                     (used in moment calculation)
                     default = False
                     if true and if two axes are given, they will be treated as a range and the moments of that data will be
                     returned

    merge_axes: bool
                (used in moment calculation)
                default = False
                if true, will treat the given axes as composing one matrix for which the moments will be returned

    Returns
    -------
    array of the length of the number of indices given with the calculated moment for each column/row

    Examples
    --------
    TODO: do this...
    # >>> ht.moment(ht.ones(2))
    """

    if len(x.shape) != 2:
        raise IndexError('Input array has too many dimensions, must be 2D was {}'.format(x.shape))
    if not direction:
        direction = 0

    if axis:
        if not isinstance(axis, (int, float, list, tuple, torch.Tensor, np.ndarray)):
            raise TypeError('axis must be either None, int, float, list, tuple, torch.Tensor, or ndarray')

    # this requires a *torch* tensor so it must be the tensor which is only on node
    # todo: does this need to be in the other loop?
    # moms = __local_operation(proc_moment, x, **{'axis': axis, 'dir': direction})
    # todo: does this work? will this do the operation on only the local data?
    # x.comm.Barrier()

    if x.comm.is_distributed():
        # need to access the correct data on each node

        merge = x.comm.Op.Create(merge_moments, communte=True)
        # todo: fix the call for the allreduce
        moms = x.comm.allreduce(x, proc_moment, op=merge, out=None, **{'axis': axis, 'direction': direction, 'continuous_axes': continuous_axes,
                                                                       'merge_axes': merge_axes})
        # print(moms)
        return moms

        # todo: test speed of allgather vs the loop below
        # wrld_sz = x.comm.Get_size()
        # # need a accurate while loop here
        # rem1 = 0
        # rem2 = 0
        # while True:  # this loop will loop pairwise over the world size and do pairwise updates
        #     # until the final result is on the 0th process, then it will be broadcast to all processes
        #     # then the function will exit
        #
        #     odd_flag = 0
        #     if wrld_sz % 2 != 0:  # test if the length is an odd number
        #         if rem1 != 0 and rem2 == 0:
        #             rem2 = wrld_sz-1
        #             odd_flag = 1
        #         elif rem1 == 0:
        #             rem1 = wrld_sz-1
        #             odd_flag = 1
        #     splt = int(wrld_sz / 2)
        #     # use 'rank in range(splt, wrld_sz-oddFlag
        #     if x.comm.rank in range(splt, wrld_sz-odd_flag):
        #         x.comm.Send(moms, dest=splt-x.comm.rank, tag=x.comm.rank)
        #     if x.comm.rank < splt:
        #         snt = torch.zeros(5)
        #         x.comm.Recv(snt, source=splt-x.comm.rank, tag=x.comm.rank)
        #         moms = merge_moments(moms, snt)
        #     if rem1 and rem2:
        #         if x.comm.rank == rem1:
        #             x.comm.Send(moms, dest=rem2, tag=x.comm.rank)
        #         if x.comm.rank == rem2:
        #             snt = torch.zeros(5)
        #             x.comm.Recv(snt, source=rem1, tag=x.comm.rank)
        #             moms = merge_moments(moms, snt)
        #         rem1 = rem2
        #         rem2 = 0
        #     wrld_sz = splt
        #     if wrld_sz == 1:
        #         if rem1:
        #             if x.comm.rank == rem1:
        #                 x.comm.Send(moms, dest=0, tag=x.comm.rank)
        #             if x.comm.rank == 0:
        #                 snt = [0., 0., 0., 0., 0]
        #                 x.comm.Recv(snt, source=rem1, tag=x.comm.rank)
        #                 moms = merge_moments(moms, snt)
        #         if x.comm.rank == 0:
        #             x.comm.Bcast(moms, root=0)
        #             return moms
        #     x.comm.Barrier()
    else:
        moms = __local_operation(proc_moment, x, out=None, **{'axis': axis, 'direction': direction, 'continuous_axes': continuous_axes,
                                                              'merge_axes': merge_axes})
        return moms


def proc_moment(x, axis, direction, continuous_axes=False, merge_axes=False):
    """
    Function which will be distributed to each node to find the moments

    Parameters:
    -----------
    x:      torch tensor
        data input (2D)
    axis:   int
            which axis of x to find the moments for
    dir:    int
            which direction to find the sum (columns (1)/rows (0))

    Returns:
    --------
    list of mu, std, m3, m4, number of elements
    """
    # if direction == 0:
    #     x1 = x[axis, :]
    # elif direction == 1:
    #     x1 = x[:, axis]
    # else:
    #     raise ValueError("dir must be 0 or 1")

    # get size of the data, if > 100 000 will chunk it
    # max_size = max(x.size())
    # spts = int(max_size / 1000000) + 1
    #
    # if spts:
    #     moms = torch.zeros(len(axis), 5)
    #     first = True
    #     chunk_dim = list(max(x.size())).index(max_size)
    #     for ch in torch.chunk(x, spts, dim=chunk_dim):
    #         # todo: fix the chunk so it doesnt happen in the same dimension as direction
    #         # calc the moment for each chunk then merge them together with the previous iteration
    #         #todo: if the chunk is not the direction
    #         tmp_moms = calc_moments(ch, axis=axis, dimen=direction, continuous_axes=continuous_axes, merge_axes=merge_axes)
    #         if first:
    #             moms = merge_moments(moms, tmp_moms)
    #             first = False
    #         else:
    #             moms = tmp_moms
    #     return moms
    # else:
    #     return calc_moments(x, axis, dimen=direction, continuous_axes=continuous_axes, merge_axes=merge_axes)
    # todo: either find a way to chunk the data here or do in in the calc_moments function, either way this whole function is useless right now
    return calc_moments(x, axis, dimen=direction, continuous_axes=continuous_axes, merge_axes=merge_axes)


def calc_moments(data, axis=None, dimen=0, continuous_axes=False, merge_axes=False):
    """
    Calculate the mean, variance, skewness and Kurtosis of the given data set

    Parameters:
    -----------
    data: pytorch tensor
          data input
          data must be of the size X by 1 where X is a natural number

    axis: int, float, list, tuple, torch.Tensor, np.ndarray
          default = None -> will sum over the whole dataset
          axis/axes for which the moments will be calculated for
          if multiple given: will either return the moments of each column unless otherwise specified

    dimen: int
           default = None -> dimension 0
           the dimension of the data for which the moments are calculated for

    continuous_axes: bool
                     default = False
                     if true and if two axes are given, they will be treated as a range and the moments of that data will be
                     returned

    merge_axes: bool
                default = False
                if true, will treat the given axes as composing one matrix for which the moments will be returned

    Returns:
    --------
    list containing mean, *variance*, skewness, kurtosis, number of entries (n-1)
        (to get unbiased estimator of varience -> var = M2/(n-1))

    Examples:
    ---------
    """
    # for all members of the tensor
    # todo: implement multi-threading
    if not axis:
        n = 0.
        m1 = 0.
        m2 = 0.
        m3 = 0.
        m4 = 0.
        mu = 0.

        for j in data.view(data.numel()):
            n += 1.
            d = j - m1
            d_n = d / n

            # mu_old = mu
            mu += (j - mu) / n
            m1 += d_n
            m4 += (6 * m3 * d_n ** 2) - (4 * m3 * d_n) + (d * d_n ** 3 * (n - 1) * (n ** 2 - 3 * n + 3))
            m3 += (d * d_n ** 2 * (n - 3) * (n - 2)) - (3 * d_n * m2)
            m2 += d * d_n * (n - 1)
            # m2 += (d - mu) * (d - mu_old)
        return torch.FloatTensor([mu, m2/(n-1), m3, m4, n - 1])

    if axis:
        if not isinstance(axis, (int, float, list, tuple, torch.Tensor, np.ndarray)):
            raise TypeError('axis must be either none, int, float, list, tuple, torch.Tensor, or ndarray')

    try:
        axis = tuple(axis)
        if continuous_axes and axis[0] >= axis[1]:
            raise ValueError("axis[1] must be > axis[0] for the continuous case")
        once_flag = False
    except TypeError:
        axis = tuple((axis, 1))
        once_flag = True

    if not isinstance(dimen, int):
        raise TypeError("Dimension must be an int, currently {}".format(type(dimen)))

    if continuous_axes or once_flag:
        out = torch.zeros(1, 5)
    else:
        out = torch.zeros(len(axis), 5)

    #   make nested for loop -> todo: fix the data call for the 1D senario
    if continuous_axes or once_flag:
        n = 0.
        m1 = 0.
        m2 = 0.
        m3 = 0.
        m4 = 0.
        mu = 0.
        length = axis[1] - axis[0] if axis[1] - axis[0] > 0 else 1
        narr = data.narrow(dimen, axis[0], length).contiguous().view(data.narrow(dimen, axis[0], length).numel())
        for dat in narr:  # needs to be parrelellized
            n += 1.
            d = dat - m1
            d_n = d / n

            # mu_old = mu
            mu += (dat - mu) / n
            m1 += d_n
            m4 += (6 * m3 * d_n ** 2) - (4 * m3 * d_n) + (d * d_n ** 3 * (n - 1) * (n ** 2 - 3 * n + 3))
            m3 += (d * d_n ** 2 * (n - 3) * (n - 2)) - (3 * d_n * m2)
            m2 += d * d_n * (n - 1)
            # m2 += (d - mu)*(d - mu_old)
        return torch.FloatTensor([mu, m2/(n-1), m3, m4, n - 1])

    # for non continuous case, need to use narrow multiple times
    # todo: multithread this
    else:
        for count, a in enumerate(axis):
            n = 0.
            m1 = 0.
            m2 = 0.
            m3 = 0.
            m4 = 0.
            mu = 0.
            narr = data.narrow(dimen, a, 1).contiguous().view(data.narrow(dimen, a, 1).numel())
            for dat in narr:  # needs to be parrelellized

                n += 1.
                d = dat - m1
                d_n = d / n

                # mu_old = mu
                mu += (dat - mu) / n
                m1 += d_n
                m4 += (6 * m3 * d_n ** 2) - (4 * m3 * d_n) + (d * d_n ** 3 * (n - 1) * (n ** 2 - 3 * n + 3))
                m3 += (d * d_n ** 2 * (n - 3) * (n - 2)) - (3 * d_n * m2)
                m2 += d * d_n * (n - 1)
                # m2 += (d - mu) * (d - mu_old)
            out[count] = torch.FloatTensor([mu, m2/(n-1), m3, m4, n - 1])
        if merge_axes:
            sz = len(axis)
            rem1 = 0
            rem2 = 0
            while True:  # this loop will loop pairwise over the whole process and do pairwise updates
                if sz % 2 != 0:  # test if the length is an odd number
                    if rem1 != 0 and rem2 == 0:
                        rem2 = sz - 1
                    elif rem1 == 0:
                        rem1 = sz - 1
                splt = int(sz / 2)
                for i in range(splt):
                    out[i] = merge_moments(out[i], out[i + splt])
                if rem1 and rem2:
                    out[rem1] = merge_moments(out[rem1], out[rem2])
                    rem1 = rem2
                    rem2 = 0
                sz = splt
                if sz == 1:
                    if rem1:
                        out[0] = merge_moments(out[0], out[rem1])
                    return out[0]
        return out


def merge_moments(mo1, mo2):
    """
    Merge two moment calculations of the form: mu, std, m3, m4, n

    :param mo1: itterable/tensor -> [mu, m2, m3, m4, number of entries] see output of the moments function
    :param mo2:
    :return: [mu_out, m2_out, m3_out, m4_out, n]
    """
    # dont need the axis here: only need to merge them if the sizes are greater than 1x5
    print(mo1.size(), mo2.size())
    if mo1.size()[-1] != 5 or mo2.size()[-1] != 5:
        raise ValueError('Tensors for moment calculations must be of second dim 5 (A x 5)')
    if mo1.size()[0] != mo2.size()[0] or mo1.size()[0] == mo2.size()[0]:
        if mo1.size()[0] != mo2.size()[0]:
            warnings.warn('Moment arrays of difference sizes, merging 0th of '
                          'both and assuming zeros for the other array', UserWarning)
            try:
                z = torch.zeros(mo1.size()[0]-mo2.size()[0], 5)
                mo2 = torch.cat((mo2, z))
            except RuntimeError:
                try:
                    z = torch.zeros(mo2.size()[0] - mo1.size()[0], 5)
                    mo1 = torch.cat((mo1, z))
                except RuntimeError:
                    raise RuntimeError("invalid memory size. likely a negative axis call in zeroes")

        # loop over the axes here to get the merged moments for each axes and return the aX5 array
    out = torch.zeros(mo1.size())
    for ax in range(mo1.size()[0]):
        try:
            oneD_flag = False
            mu1 = mo1[ax][0]
            m2_1 = mo1[ax][1]
            m3_1 = mo1[ax][2]
            m4_1 = mo1[ax][3]
            n1 = mo1[ax][-1] + 1.

            mu2 = mo2[ax][0]
            m2_2 = mo2[ax][1]
            m3_2 = mo2[ax][2]
            m4_2 = mo2[ax][3]
            n2 = mo2[ax][-1] + 1.
        except IndexError:
            oneD_flag = True
            mu1 = mo1[0]
            m2_1 = mo1[1]
            m3_1 = mo1[2]
            m4_1 = mo1[3]
            n1 = mo1[-1] + 1.

            mu2 = mo2[0]
            m2_2 = mo2[1]
            m3_2 = mo2[2]
            m4_2 = mo2[3]
            n2 = mo2[-1] + 1.
        n = n1 + n2

        delta = mu2 - mu1

        mu_out = mu1 + n2 * (delta / n)
        m2_out = m2_1 + m2_2 + (delta ** 2) * n1 * n2 / n
        m3_out = m3_1 + m3_2 + 3 * delta * ((n1 / n) * m2_2 - (n2 / n) * m2_1) + (
                    (delta ** 3) / (n ** 2)) * n1 * n2 * (n1 - n2)
        m4_out = m4_1 + m4_2 + 4 * delta * (((n1 / n) * m3_2) - (n2 / n) * m3_1) \
                 + 6 * (delta ** 2) * (((n2 / n) ** 2 * m2_1) + (n1 / n) ** 2 * m2_2) \
                 + (delta ** 4 / n ** 3) * n1 * n2 * (n1 ** 2 - n1 * n2 + n2 ** 2)
        if oneD_flag:
            out = torch.FloatTensor([mu_out, m2_out/(n-2), m3_out, m4_out, n - 2])
        else:
            out[ax] = torch.FloatTensor([mu_out, m2_out/(n-2), m3_out, m4_out, n - 2])
    return out


def mean(x, axis=None, dimen=0, continuous_axes=False, merge_axes=False):
    '''
    Function for the sum of a tensor
    :param x:       ht tensor       data
    :param axis:    int/multiple ints   axes for which to get the mean of, if none (default: all)
    :return:
    '''
    def merge_means(mu1, n1, mu2, n2, datatype):
        delta = mu2 - mu1
        return mu1 + n2 * (delta / (n1+n2)), n1 + n2

    if dimen:
        if not isinstance(dimen, int):
            raise TypeError("Dimension (dimen) must be an int, currently is {}".format(type(dimen)))


    if axis:
        if not isinstance(axis, (int, float, list, tuple, torch.Tensor, np.ndarray)):
            raise TypeError('axis must be either None, int, float, list, tuple, torch.Tensor, or ndarray')
        if axis >= any(x.shape):
            raise ValueError('axis {} is out of bounds for shape {}'.format(axis, x.shape))
    else:
        # case for full matrix calculation
        mu = __local_operation(torch.mean, x, out=None)
        if not x.comm.is_distributed():
            return mu

    try:
        axis = tuple(axis)
        if continuous_axes and axis[0] >= axis[1]:
            raise ValueError("axis[1] must be > axis[0] for the continuous case")
        once_flag = False
    except TypeError:
        axis = tuple((axis, 1))
        once_flag = True

    mu = __local_operation(torch.mean, x, out=None)
    n = __local_operation(torch.numel, x, out=None)
    mu_tot = torch.zeros((x.comm.Get_size(), 2))
    mu_proc = torch.zeros((x.comm.Get_size(), 2))
    mu_proc[x.comm.rank] = torch.Tensor(mu, n)

    # send all to an array with the index as the rank
    # mu_merge = x.comm.Op.Create(merge_means, commute=True)
    x.comm.allreduce(mu_tot, mu_proc, MPI.SUM)
    x.comm.barrier()

    rem1 = 0
    rem2 = 0
    sz = mu_tot.size()[0]
    while True:  # this loop will loop pairwise over the whole process and do pairwise updates
        if sz % 2 != 0:  # test if the length is an odd number
            if rem1 != 0 and rem2 == 0:
                rem2 = sz - 1
            elif rem1 == 0:
                rem1 = sz - 1
        splt = int(sz / 2)
        for i in range(splt):
            mu_tot[i] = merge_means(mu_tot[i][0], mu_tot[i][1], mu_tot[i+splt][0], mu_tot[i+splt][1])
        if rem1 and rem2:
            mu_tot[rem1] = merge_means(mu_tot[rem1][0], mu_tot[rem1][1], mu_tot[rem2][0], mu_tot[rem2][1])
            rem1 = rem2
            rem2 = 0
        sz = splt
        if sz == 1:
            if rem1:
                mu_tot[0] = merge_means(mu_tot[0][0], mu_tot[0][1], mu_tot[rem1][0], mu_tot[rem1][1])
            return mu_tot[0][0]
    # return x.comm.allreduce((mu, n), mu_tot, op=mu_merge)
    # return __reduce_op(x, torch.mean, mu_merge, axis)


def std(x, axis=None):
    '''

    :param x:
    :param axis:
    :return:
    :return:
    '''
    if axis:
        if not isinstance(axis, (int, float, list, tuple, torch.Tensor, np.ndarray)):
            raise TypeError('axis must be either an int, float, list, tuple, torch.Tensor, or ndarray, '
                            'currently is {}'.format(type(axis)))

    # if x.comm.is_distributed():
        # mutliple processes
    std = __local_operation(torch.std, x, out=None)
    if not x.comm.is_distributed():
        return std
    mu = __local_operation(torch.mean, x, out=None)
    n = __local_operation(torch.numel, x, out=None)
    std_tot = torch.zeros(x.comm.Get_size(), 3)
    std_proc = torch.zeros(x.comm.Get_size(), 3)
    std_proc[x.comm.rank] = torch.Tensor(std, mu, n)

    def merge_stds(std1, mu1, n1, std2, mu2, n2, datatype):
        n = n1 + n2
        delta = mu2 - mu1
        mu_out = mu1 + n2 * (delta / n)
        return std1 + std2 + (delta ** 2) * (n1 * n2) / n, mu_out, n

    # send all to an array with the index as the rank
    # mu_merge = x.comm.Op.Create(merge_means, commute=True)
    x.comm.allreduce(std_tot, std_proc, MPI.SUM)

    rem1 = 0
    rem2 = 0
    sz = std_tot.size()[0]
    while True:  # this loop will loop pairwise over the whole process and do pairwise updates
        if sz % 2 != 0:  # test if the length is an odd number
            if rem1 != 0 and rem2 == 0:
                rem2 = sz - 1
            elif rem1 == 0:
                rem1 = sz - 1
        splt = int(sz / 2)
        for i in range(splt):
            std_tot[i] = merge_stds(std_tot[i][0], std_tot[i][1], std_tot[i][2],
                                    std_tot[i + splt][0], std_tot[i + splt][1], std_tot[i + splt][2])
        if rem1 and rem2:
            std_tot[rem1] = merge_stds(std_tot[rem1][0], std_tot[rem1][1], std_tot[rem1][2],
                                       std_tot[rem2][0], std_tot[rem2][1], std_tot[rem2][2])
            rem1 = rem2
            rem2 = 0
        sz = splt
        if sz == 1:
            if rem1:
                std_tot[0] = merge_stds(std_tot[0][0], std_tot[0][1], std_tot[0][2],
                                        std_tot[rem1][0], std_tot[rem1][1], std_tot[rem1][2])
            return std_tot[0][0]
    # std_merge = x.comm.Op.Create(merge_stds, commute=True)
    # return __reduce_op(x, torch.std, std_merge, axis)


def __local_operation(operation, x, out, **kwargs):
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
        return tensor.tensor(operation(x._tensor__array.type(torch_type), **kwargs),
                             x.gshape, promoted_type, x.split, x.comm)

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
    operation(casted.repeat(multiples) if needs_repetition else casted,
              out=out._tensor__array, **kwargs)
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


def __reduce_op(x, partial_op, op, axis, out=None, **kwargs):
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

    if axis is None:
        partial = torch.reshape(partial_op(x._tensor__array, **kwargs), (1,), **kwargs)
        output_shape = (1,)
    else:
        partial = partial_op(x._tensor__array, axis, keepdim=True, **kwargs)
        # TODO: verify if this works for negative split axis Issue #103
        output_shape = x.gshape[:axis] + (1,) + x.gshape[axis + 1:]

    # Check shape of output buffer, if any
    if out is not None and out.shape != output_shape:
        raise ValueError('Expecting output buffer of shape {}, got {}'.format(
            output_shape, out.shape))

    if x.comm.is_distributed() and (axis is None or axis == x.split):
        x.comm.Allreduce(MPI.IN_PLACE, partial[0], op)
        if out is not None:
            out._tensor__array = tensor.tensor(partial, output_shape, types.canonical_heat_type(
                partial[0].dtype), split=out.split, comm=x.comm)
            return out
        return tensor.tensor(partial, output_shape, types.canonical_heat_type(partial[0].dtype), split=None, comm=x.comm)
    if out is not None:
        out._tensor__array = tensor.tensor(partial, output_shape, types.canonical_heat_type(
            partial[0].dtype), split=out.split, comm=x.comm)
        return out
    return tensor.tensor(partial, output_shape, types.canonical_heat_type(partial[0].dtype), split=None, comm=x.comm)
