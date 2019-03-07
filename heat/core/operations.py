import itertools
import torch
import warnings
import numpy as np

from .communication import MPI
from . import stride_tricks
from . import types
from . import tensor
import builtins

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
    'mean',
    'sin',
    'sqrt',
    'std',
    'sum',
    'transpose',
    'tril',
    'triu',
    'var'
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
    # TODO: initial : scalar, optional Issue #101
    """
    Return the maximum along a given axis.

    Parameters
    ----------
    a : ht.tensor
        Input data.
    axis : None or int, optional
        Axis or axes along which to operate. By default, flattened input is used.
    out : ht.tensor, optional
        Tuple of two output tensors (max, max_indices). Must be of the same shape and buffer length as the expected
        output. The minimum value of an output element. Must be present to allow computation on empty slice.

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
    tensor([[10., 11., 12.]])
    >>> ht.min(a, axis=1)
    tensor([[ 3.],
        [ 6.],
        [ 9.],
        [12.]])
    """
    def local_max(*args, **kwargs):
        result = torch.max(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    return __reduce_op(x, local_max, MPI.MAX, axis, out)


def min(x, axis=None, out=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    a : ht.tensor
        Input data.
    axis : None or int
        Axis or axes along which to operate. By default, flattened input is used.
    out : ht.tensor, optional
        Tuple of two output tensors (min, min_indices). Must be of the same shape and buffer length as the expected
        output.The maximum value of an output element. Must be present to allow computation on empty slice.

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
    tensor([[1., 2., 3.]])
    >>> ht.min(a, axis=1)
    tensor([[ 1.],
        [ 4.],
        [ 7.],
        [10.]])
    """
    def local_min(*args, **kwargs):
        result = torch.min(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    return __reduce_op(x, local_min, MPI.MIN, axis, out)


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
    return __reduce_op(x, torch.sum, MPI.SUM, axis, out)


def merge_means(mu1, n1, mu2, n2):
    """
    Function to merge two means by pairwise update

    Parameters
    ----------
    mu1 : 1D ht.tensor or 1D torch.tensor
          Calculated mean
    n1 : 1D ht.tensor or 1D torch.tensor
         number of elements used to calculate mu1
    mu2 : 1D ht.tensor or 1D torch.tensor
          Calculated mean
    n2 : 1D ht.tensor or 1D torch.tensor
         number of elements used to calculate mu2

    Returns
    -------
    mean of combined set
    number of elements in the combined set

    References
    ----------
    [1] J. Bennett, R. Grout, P. Pebay, D. Roe, D. Thompson, Numerically stable, single-pass, parallel statistics algorithms,
        IEEE International Conference on Cluster Computing and Workshops, 2009, Oct 2009, New Orleans, LA, USA.
    """
    delta = mu2.item() - mu1.item()
    n1 = n1.item()
    n2 = n2.item()
    return mu1 + n2 * (delta / (n1 + n2)), n1 + n2


def mean(x, axis=None):
    """
    Calculates and returns the mean of a tensor.
    If a axis is given, the mean will be taken in that direction.

    Parameters
    ----------
    x : ht.tensor
        Values for which the mean is calculated for
    axis : None, Int, iterable
           axis which the mean is taken in.
           Default: None -> mean of all data calculated

    Returns
    -------
    ht.tensor containing the mean/s, if split, then split in the same direction as x.
    """

    def reduce_means_elementwise(output_shape_i):
        """
        Function to combine the calculated means together.
        This does an element-wise update of the calculated means to merge them together using the merge_means function.
        This function operates using x from the mean function paramters

        Parameters
        ----------
        output_shape_i : iterable
                        iterable with the dimensions of the output of the mean function

        Returns
        -------
        ht.tensor of the calculated means
        """
        if x.lshape[x.split] != 0:
            mu = __local_operation(torch.mean, x, out=None, dim=axis)
        else:
            mu = tensor.zeros(output_shape_i)

        mu_to_combine = tensor.zeros((x.comm.Get_size(),) + tuple(output_shape_i))
        mu_tot = tensor.zeros((x.comm.Get_size(),) + tuple(output_shape_i))

        n_for_merge = tensor.zeros(x.comm.Get_size())
        n2 = tensor.zeros(x.comm.Get_size())
        mu_to_combine[x.comm.Get_rank()] = mu
        n2[x.comm.Get_rank()] = x.lshape[x.split]
        x.comm.Allreduce(mu_to_combine, mu_tot, MPI.SUM)
        x.comm.Allreduce(n2, n_for_merge, MPI.SUM)

        sz = x.comm.Get_size()
        rem1 = rem2 = 0
        while True:
            if sz % 2 != 0:  # test if the length is an odd number
                if rem1 != 0 and rem2 == 0:
                    rem2 = sz - 1
                elif rem1 == 0:
                    rem1 = sz - 1
            splt = int(sz / 2)
            for sp_it in range(splt):  # todo: multithread perfect opp for GPU parrallelizm
                mu_reshape = __local_operation(torch.reshape, mu_tot[sp_it], out=None, shape=(1, int(mu_tot[sp_it].lnumel)))[0]
                for en, (el1, el2) in enumerate(zip(mu_reshape,
                                                    __local_operation(torch.reshape, mu_tot[sp_it+splt], out=None,
                                                                      shape=(1, int(mu_tot[sp_it+splt].lnumel)))[0])):
                    try:
                        mu_reshape[en], n = merge_means(el1, n_for_merge[sp_it], el2, n_for_merge[sp_it+splt])
                    except IndexError:
                        mu_reshape, n = merge_means(el1, n_for_merge[sp_it], el2, n_for_merge[sp_it + splt])
                n_for_merge[sp_it] = n  # todo: need to update the n_for_merge here not previously!!
                mu_tot[sp_it] = __local_operation(torch.reshape, mu_reshape, out=None, shape=output_shape_i)
            if rem1 and rem2:
                mu_reshape = __local_operation(torch.reshape, mu_tot[rem1], out=None, shape=(1, int(mu_tot[rem1].lnumel)))[0]
                for en, (el1, el2) in enumerate(zip(mu_reshape,
                                                    __local_operation(torch.reshape, mu_tot[rem2], out=None,
                                                                      shape=(1, int(mu_tot[rem2].lnumel)))[0])):
                    mu_reshape[en], n = merge_means(el1, n_for_merge[rem1], el2, n_for_merge[rem2])
                n_for_merge[rem2] = n
                mu_tot[rem2] = __local_operation(torch.reshape, mu_reshape, out=None, shape=output_shape_i)

                rem1 = rem2
                rem2 = 0
            sz = splt
            if sz == 1 or sz == 0:
                if rem1:
                    mu_reshape = __local_operation(torch.reshape, mu_tot[0], out=None, shape=(1, int(mu_tot[0].lnumel)))[0]
                    for en, (el1, el2) in enumerate(zip(mu_reshape,
                                                        __local_operation(torch.reshape, mu_tot[rem1], out=None,
                                                                          shape=(1, int(mu_tot[rem1].lnumel)))[0])):
                        mu_reshape[en], _ = merge_means(el1, n_for_merge[0], el2, n_for_merge[rem1])

                    mu_tot[0] = __local_operation(torch.reshape, mu_reshape, out=None, shape=output_shape_i)
                return mu_tot[0]
    # ------------------------------------------------------------------------------------------------------------------
    if axis is None:
        # full matrix calculation
        if x.split:
            # if x is distributed and no axis is given: return mean of the whole set
            # print("axis is None, distributed case", axis, x.split)
            if x.lshape[x.split] != 0:
                mu_in = __local_operation(torch.mean, x, out=None)
            else:
                mu_in = 0
            n = x.lnumel
            mu_tot = tensor.zeros((x.comm.Get_size(), 2))
            mu_proc = tensor.zeros((x.comm.Get_size(), 2))
            mu_proc[x.comm.rank][0] = mu_in
            mu_proc[x.comm.rank][1] = float(n)
            x.comm.Allreduce(mu_proc, mu_tot, MPI.SUM)

            rem1 = 0
            rem2 = 0
            sz = mu_tot.shape[0]
            while True:  # this loop will loop pairwise over the whole process and do pairwise updates
                if sz % 2 != 0:  # test if the length is an odd number
                    if rem1 != 0 and rem2 == 0:
                        rem2 = sz - 1
                    elif rem1 == 0:
                        rem1 = sz - 1
                splt = int(sz / 2)
                for i in range(splt):  # todo: make this multithreaded, perfect opportunity for GPU usage but this might be smaller than the gains
                    merged = merge_means(mu_tot[i, 0], mu_tot[i, 1], mu_tot[i + splt, 0], mu_tot[i + splt, 1])
                    for enum, m in enumerate(merged):
                        mu_tot[i, enum] = m
                if rem1 and rem2:
                    merged = merge_means(mu_tot[rem1, 0], mu_tot[rem1, 1], mu_tot[rem2, 0], mu_tot[rem2, 1])
                    for enum, m in enumerate(merged):
                        mu_tot[rem2, enum] = m
                    rem1 = rem2
                    rem2 = 0
                sz = splt
                if sz == 1 or sz == 0:
                    if rem1:
                        merged = merge_means(mu_tot[0, 0], mu_tot[0, 1], mu_tot[rem1, 0], mu_tot[rem1, 1])
                        for enum, m in enumerate(merged):
                            mu_tot[0, enum] = m
                    return mu_tot[0][0]
        else:
            # if x is not distributed do a torch.mean on x
            # print('axis==None, not distributed', axis, x.split)
            return __local_operation(torch.mean, x, out=None)
    else:
        # if isinstance(axis, int):
        #     if axis >= len(x.shape):
        #         raise ValueError("axis (axis) must be < {}, currently is {}".format(len(x.shape), axis))
        #     axis = axis if axis > 0 else axis % len(x.shape)
        # else:
        #     raise TypeError("axis (axis) must be an int, currently is {}".format(type(axis)))
        output_shape = list(x.shape)
        if isinstance(axis, (list, tuple, tensor.tensor, torch.Tensor)):
            if any([not isinstance(j, int) for j in axis]):
                raise ValueError("items in axis itterable must be integers, axes: {}".format([type(q) for q in axis]))
            if any(d > len(x.shape) for d in axis):
                raise ValueError("axes (axis) must be < {}, currently are {}".format(len(x.shape), axis))
            if any(d < 0 for d in axis):
                axis = [j % len(x.shape) for j in axis]

            # multiple dimensions
            if x.split is None:
                # print("not split, given sigular axis", axis, x.split)
                return __local_operation(torch.mean, x, out=None, **{'dim': axis})
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it not in axis]
            if x.split in axis:
                # print('multiple dimensions, split {} in dimensions given {}'.format(x.split, axis))
                # merge in the direction of the split
                return reduce_means_elementwise(output_shape)

            else:
                # multiple dimensions which does *not* include the split axis
                # combine along the split axis
                try:
                    return tensor.array(__local_operation(torch.mean, x, out=None, dim=axis), split=x.split, comm=x.comm)
                except ValueError:
                    return __local_operation(torch.mean, x, out=None, dim=axis)
        elif isinstance(axis, int):
            if axis >= len(x.shape):
                raise ValueError("axis (axis) must be < {}, currently is {}".format(len(x.shape), axis))
            axis = axis if axis > 0 else axis % len(x.shape)

            # only one axis given
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]

            if x.split is None:
                # print("not split, given sigular axis", axis, x.split)
                return __local_operation(torch.mean, x, out=None, **{'dim': axis})
            if axis == x.split:
                # print('singular axis == x.split', axis, x.split)
                return reduce_means_elementwise(output_shape)
            else:
                # singular axis given (axis) not equal to split direction (x.split)
                # local operation followed by array creation to create the full tensor of the means
                try:
                    return tensor.array(__local_operation(torch.mean, x, out=None, dim=axis), split=x.split, comm=x.comm)
                except ValueError:
                    return __local_operation(torch.mean, x, out=None, dim=axis)
        else:
            raise TypeError("axis (axis) must be an int or a list, ht.tensor, torch.Tensor, or tuple, currently is {}".format(type(axis)))


def merge_vars(var1, mu1, n1, var2, mu2, n2, bessel=True):
    """
    Function to merge two variances by pairwise update
    **Note** this is a parallel of the merge_means function

    Parameters
    ----------
    mu1 : 1D ht.tensor or 1D torch.tensor
          Calculated mean
    n1 : 1D ht.tensor or 1D torch.tensor
         number of elements used to calculate mu1
    mu2 : 1D ht.tensor or 1D torch.tensor
          Calculated mean
    n2 : 1D ht.tensor or 1D torch.tensor
         number of elements used to calculate mu2

    Returns
    -------
    mean of combined set
    number of elements in the combined set

    References
    ----------
    [1] J. Bennett, R. Grout, P. Pebay, D. Roe, D. Thompson, Numerically stable, single-pass, parallel statistics algorithms,
        IEEE International Conference on Cluster Computing and Workshops, 2009, Oct 2009, New Orleans, LA, USA.
    """
    n1 = n1.item()
    n2 = n2.item()
    n = n1 + n2
    delta = mu2.item() - mu1.item()
    if bessel:
        return (var1 * (n1 - 1) + var2 * (n2 - 1) + (delta ** 2) * n1 * n2 / n) / (n - 1), mu1 + n2 * (delta / (n1 + n2)), n
    else:
        return (var1 * n1 + var2 * n2 + (delta ** 2) * n1 * n2 / n) / n, mu1 + n2 * (delta / (n1 + n2)), n


def var(x, axis=None, bessel=True):
    """
    Calculates and returns the variance of a tensor.
    If a axis is given, the variance will be taken in that direction.

    Parameters
    ----------
    x : ht.tensor
        Values for which the mean is calculated for
    axis : None, Int
            axis which the mean is taken in.
            Default: None -> var of all data calculated
            NOTE -> if multidemensional var is implemented in pytorch, this can be an iterable. Only thing which muse be changed is the raise
    all_procs : Bool
                Flag to distribute the data to all processes
                If True: will split the result in the same direction as x
                Default: False (var of the whole dataset still calculated but not available on every node)
    bessel : Bool
             Default: True
             use the bessel correction when calculating the varaince/std
             toggle between unbiased and biased calculation of the std

    Returns
    -------
    ht.tensor containing the var/s, if split, then split in the same direction as x.
    """

    if not isinstance(bessel, bool):
        raise TypeError("bessel must be a boolean, currently is {}".format(type(bessel)))

    def reduce_vars_elementwise(output_shape_i):
        """
        Function to combine the calculated vars together.
        This does an element-wise update of the calculated vars to merge them together using the merge_vars function.
        This function operates using x from the var function paramters

        Parameters
        ----------
        output_shape_i : iterable
                        iterable with the dimensions of the output of the var function

        Returns
        -------
        ht.tensor of the calculated vars
        """

        if x.lshape[x.split] != 0:
            mu = __local_operation(torch.mean, x, out=None, dim=axis)
            var = __local_operation(torch.var, x, out=None, dim=axis, unbiased=bessel)
        else:
            mu = tensor.zeros(output_shape_i)
            var = tensor.zeros(output_shape_i)

        mu_to_combine = tensor.zeros((x.comm.Get_size(),) + tuple(output_shape_i))
        mu_tot = tensor.zeros((x.comm.Get_size(),) + tuple(output_shape_i))
        var_to_combine = tensor.zeros((x.comm.Get_size(),) + tuple(output_shape_i))
        var_tot = tensor.zeros((x.comm.Get_size(),) + tuple(output_shape_i))

        n_for_merge = tensor.zeros(x.comm.Get_size())
        n2 = tensor.zeros(x.comm.Get_size())
        mu_to_combine[x.comm.Get_rank()] = mu
        var_to_combine[x.comm.Get_rank()] = var
        n2[x.comm.Get_rank()] = x.lshape[x.split]
        x.comm.Allreduce(mu_to_combine, mu_tot, MPI.SUM)
        x.comm.Allreduce(var_to_combine, var_tot, MPI.SUM)
        x.comm.Allreduce(n2, n_for_merge, MPI.SUM)

        sz = x.comm.Get_size()
        rem1 = rem2 = 0
        while True:
            if sz % 2 != 0:  # test if the length is an odd number
                if rem1 != 0 and rem2 == 0:
                    rem2 = sz - 1
                elif rem1 == 0:
                    rem1 = sz - 1
            splt = int(sz / 2)
            for i in range(splt):  # todo: multithread for GPU
                mu_reshape = __local_operation(torch.reshape, mu_tot[i], out=None, shape=(1, int(mu_tot[i].lnumel)))[0]
                var_reshape = __local_operation(torch.reshape, var_tot[i], out=None, shape=(1, int(mu_tot[i].lnumel)))[0]
                for en, (mu1, var1, mu2, var2) in enumerate(zip(mu_reshape, var_reshape,
                                                                __local_operation(torch.reshape, mu_tot[i+splt], out=None,
                                                                                  shape=(1, int(mu_tot[i+splt].lnumel)))[0],
                                                                __local_operation(torch.reshape, var_tot[i+splt], out=None,
                                                                                  shape=(1, int(var_tot[i+splt].lnumel)))[0])):
                    # print(i, en, mu1, var1, mu2, var2)
                    try:
                        var_reshape[en], mu_reshape[en], n = merge_vars(var1, mu1, n_for_merge[i], var2, mu2, n_for_merge[i+splt], bessel)
                    except ValueError:
                        var_reshape, mu_reshape, n = merge_vars(var1, mu1, n_for_merge[i], var2, mu2, n_for_merge[i + splt], bessel)
                n_for_merge[i] = n
                mu_tot[i] = __local_operation(torch.reshape, mu_reshape, out=None, shape=output_shape_i)
                var_tot[i] = __local_operation(torch.reshape, var_reshape, out=None, shape=output_shape_i)

            if rem1 and rem2:
                mu_reshape = __local_operation(torch.reshape, mu_tot[rem1], out=None, shape=(1, int(mu_tot[rem1].lnumel)))[0]
                var_reshape = __local_operation(torch.reshape, var_tot[rem1], out=None, shape=(1, int(mu_tot[rem1].lnumel)))[0]
                for en, (mu1, var1, mu2, var2) in enumerate(zip(mu_reshape, var_reshape,
                                                                __local_operation(torch.reshape, mu_tot[rem2], out=None,
                                                                                  shape=(1, int(mu_tot[rem2].lnumel)))[0],
                                                                __local_operation(torch.reshape, var_tot[rem2], out=None,
                                                                                  shape=(1, int(var_tot[rem2].lnumel)))[0])):
                    var_reshape[en], mu_reshape[en], n = merge_vars(var1, mu1, n_for_merge[rem1], var2, mu2, n_for_merge[rem2], bessel)
                n_for_merge[rem2] = n
                mu_tot[rem2] = __local_operation(torch.reshape, mu_reshape, out=None, shape=output_shape_i)
                var_tot[rem2] = __local_operation(torch.reshape, var_reshape, out=None, shape=output_shape_i)

                rem1 = rem2
                rem2 = 0
            sz = splt
            if sz == 1 or sz == 0:
                if rem1:
                    mu_reshape = __local_operation(torch.reshape, mu_tot[0], out=None, shape=(1, int(mu_tot[0].lnumel)))[0]
                    var_reshape = __local_operation(torch.reshape, var_tot[0], out=None, shape=(1, int(mu_tot[0].lnumel)))[0]
                    for en, (mu1, var1, mu2, var2) in enumerate(zip(mu_reshape, var_reshape,
                                                                    __local_operation(torch.reshape, mu_tot[rem1], out=None,
                                                                                      shape=(1, int(mu_tot[rem1].lnumel)))[0],
                                                                    __local_operation(torch.reshape, var_tot[rem1], out=None,
                                                                                      shape=(1, int(var_tot[rem1].lnumel)))[0])):
                        var_reshape[en], mu_reshape[en], n = merge_vars(var1, mu1, n_for_merge[0], var2, mu2, n_for_merge[rem1], bessel)
                    mu_tot[0] = __local_operation(torch.reshape, mu_reshape, out=None, shape=output_shape_i)
                    var_tot[0] = __local_operation(torch.reshape, var_reshape, out=None, shape=output_shape_i)
                return var_tot[0]
    # ----------------------------------------------------------------------------------------------------
    if axis is None:
        # case for full matrix calculation (axis is None)
        if x.split is not None:
            # print("axis is None, distributed case", axis, x.split)
            if x.lshape[x.split] != 0:
                mu_in = __local_operation(torch.mean, x, out=None)
                var_in = __local_operation(torch.var, x, out=None, unbiased=bessel)
            else:
                mu_in = 0
                var_in = 0
            n = x.lnumel
            var_tot = tensor.zeros((x.comm.Get_size(), 3))
            var_proc = tensor.zeros((x.comm.Get_size(), 3))
            var_proc[x.comm.rank][0] = var_in
            var_proc[x.comm.rank][1] = mu_in
            var_proc[x.comm.rank][2] = float(n)
            x.comm.Allreduce(var_proc, var_tot, MPI.SUM)

            rem1 = 0
            rem2 = 0
            sz = var_tot.shape[0]
            while True:  # this loop will loop pairwise over the whole process and do pairwise updates
                if sz % 2 != 0:  # test if the length is an odd number
                    if rem1 != 0 and rem2 == 0:
                        rem2 = sz - 1
                    elif rem1 == 0:
                        rem1 = sz - 1
                splt = int(sz / 2)
                for i in range(splt):
                    merged = merge_vars(var_tot[i, 0], var_tot[i, 1], var_tot[i, 2],
                                        var_tot[i + splt, 0], var_tot[i + splt, 1], var_tot[i + splt, 2], bessel)
                    for enum, m in enumerate(merged):
                        var_tot[i, enum] = m
                if rem1 and rem2:
                    merged = merge_vars(var_tot[rem1, 0], var_tot[rem1, 1], var_tot[rem1, 2],
                                        var_tot[rem2, 0], var_tot[rem2, 1], var_tot[rem2, 2], bessel)
                    for enum, m in enumerate(merged):
                        var_tot[rem2, enum] = m
                    rem1 = rem2
                    rem2 = 0
                sz = splt
                if sz == 1 or sz == 0:
                    if rem1:
                        merged = merge_vars(var_tot[0, 0], var_tot[0, 1], var_tot[0, 2],
                                            var_tot[rem1, 0], var_tot[rem1, 1], var_tot[rem1, 2], bessel)
                        for enum, m in enumerate(merged):
                            var_tot[0, enum] = m
                    return var_tot[0][0]
        else:
            # full matrix on one node
            return __local_operation(torch.var, x, out=None, unbiased=bessel)
    else:
        # case for mean in one dimension
        output_shape = list(x.shape)
        # if isinstance(axis, (list, tuple, tensor.tensor, torch.Tensor)):
        #     if any(d > len(x.shape) for d in axis):
        #         raise ValueError("Axis (axis) must be < {}, currently are {}".format(len(x.shape), axis))
        #     if any(d < 0 for d in axis):
        #         axis = [j % len(x.shape) for j in axis]
        if isinstance(axis, int):
            if axis >= len(x.shape):
                raise ValueError("axis must be < {}, currently is {}".format(len(x.shape), axis))
            axis = axis if axis > 0 else axis % len(x.shape)
            # only one axis given
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]
            if x.split is None:
                return __local_operation(torch.var, x, out=None, dim=axis, unbiased=bessel)
            elif axis == x.split:
                return reduce_vars_elementwise(output_shape)
            else:
                try:
                    return tensor.array(__local_operation(torch.var, x, out=None, dim=axis, unbiased=bessel), split=x.split, comm=x.comm)
                except ValueError:
                    return __local_operation(torch.var, x, out=None, dim=axis, unbiased=bessel)
        else:
            raise TypeError("Axis (axis) must be an int, currently is {}".format(type(axis)))
            # TODO: when multi dim support is abalable from pytorch this can be uncommented and the raise above can be changed
            # # multiple dimensions
            # output_shape = [output_shape[it] for it in range(len(output_shape)) if it not in axis]
            # if x.split in axis:
            #     # merge in the direction of the split
            #     return reduce_vars_elementwise(output_shape)
            #
            # else:
            #     # multiple dimensions which does *not* include the split axis
            #     # combine along the split axis
            #     try:
            #         return tensor.array(__local_operation(torch.var, x, out=None, dim=axis, unbiased=bessel), split=x.split, comm=x.comm)
            #     except ValueError:
            #         return __local_operation(torch.var, x, out=None, dim=axis, unbiased=bessel)



def std(x, axis=None, bessel=True):
    """
    Calculates and returns the standard deviation of a tensor with the bessel correction
    If a axis is given, the variance will be taken in that direction.

    Parameters
    ----------
    x : ht.tensor
        Values for which the mean is calculated for
    axis : None, Int
            axis which the mean is taken in.
            Default: None -> var of all data calculated
            NOTE -> if multidemensional var is implemented in pytorch, this can be an iterable. Only thing which muse be changed is the raise
    all_procs : Bool
                Flag to distribute the data to all processes
                If True: will split the result in the same direction as x
                Default: False (var of the whole dataset still calculated but not available on every node)
    bessel : Bool
             Default: True
             use the bessel correction when calculating the varaince/std
             toggle between unbiased and biased calculation of the std

    Returns
    -------
    ht.tensor containing the std/s, if split, then split in the same direction as x.
    """
    if not axis:
        return np.sqrt(var(x, axis, bessel))
    else:
        return sqrt(var(x, axis, bessel), out=None)


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
        result = operation(x._tensor__array.type(torch_type), **kwargs)
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
    operation(casted.repeat(multiples) if needs_repetition else casted, out=out._tensor__array, **kwargs)
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
