import numpy as np
import torch
import warnings

from .communication import MPI
from . import exponential
from . import factories
from . import operations
from . import dndarray
from . import types


__all__ = [
    'argmax',
    'argmin',
    'max',
    'mean',
    'min',
    'std',
    'var'
]


def argmax(x, axis=None, out=None, **kwargs):
    """
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    x : ht.DNDarray
        Input array.
    axis : int, optional
        By default, the index is into the flattened tensor, otherwise along the specified axis.
    out : ht.DNDarray, optional.
        If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

    Returns
    -------
    index_tensor : ht.DNDarray of ints
        Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

    Examples
    --------
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    >>> a = ht.random.randn(3,3)
    >>> a
    tensor([[-0.5631, -0.8923, -0.0583],
            [-0.1955, -0.9656,  0.4224],
            [ 0.2673, -0.4212, -0.5107]])
    >>> ht.argmax(a)
    tensor([5])
    >>> ht.argmax(a, axis=0)
    tensor([[2, 2, 1]])
    >>> ht.argmax(a, axis=1)
    tensor([[2],
            [2],
            [0]])
    """
    def local_argmax(*args, **kwargs):
        axis = kwargs.get('dim', -1)
        shape = x.shape

        # case where the argmin axis is set to None
        # argmin will be the flattened index, computed standalone and the actual minimum value obtain separately
        if len(args) <= 1 and axis < 0:
            indices = torch.argmax(*args, **kwargs).reshape(1)
            maxima = args[0].flatten()[indices]

            # artificially flatten the input tensor shape to correct the offset computation
            axis = x.split
            shape = [np.prod(shape)]
        # usual case where indices and maximum values are both returned. Axis is not equal to None
        else:
            maxima, indices = torch.max(*args, **kwargs)

        # add offset of data chunks if reduction is computed across split axis
        if axis == x.split:
            offset, _, _ = x.comm.chunk(shape, x.split)
            indices += offset

        return torch.cat([maxima.double(), indices.double()])

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError('axis must be None or int, but was {}'.format(type(axis)))

    # perform the global reduction
    reduced_result = operations.__reduce_op(x, local_argmax, MPI_ARGMAX, axis=axis, out=None, **kwargs)

    # correct the tensor
    reduced_result._DNDarray__array = reduced_result._DNDarray__array.chunk(2)[-1].type(torch.int64)
    reduced_result._DNDarray__dtype = types.int64

    # address lshape/gshape mismatch when axis is 0
    if axis is not None:
        if isinstance(axis, int):
            axis = (axis,)
        if 0 in axis:
            reduced_result._DNDarray__gshape = (1,) + reduced_result._DNDarray__gshape
            if not kwargs.get('keepdim'):
                reduced_result = reduced_result.squeeze(axis=0)

    # set out parameter correctly, i.e. set the storage correctly
    if out is not None:
        if out.shape != reduced_result.shape:
            raise ValueError('Expecting output buffer of shape {}, got {}'.format(reduced_result.shape, out.shape))
        out._DNDarray__array.storage().copy_(reduced_result._DNDarray__array.storage())
        out._DNDarray__array = out._DNDarray__array.type(torch.int64)
        out._DNDarray__dtype = types.int64
        return out

    return reduced_result


def argmin(x, axis=None, out=None, **kwargs):
    """
    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    x : ht.DNDarray
        Input array.
    axis : int, optional
        By default, the index is into the flattened tensor, otherwise along the specified axis.
    # out : ht.DNDarray, optional. Issue #100
        If provided, the result will be inserted into this tensor. It should be of the appropriate shape and dtype.

    Returns
    -------
    index_tensor : ht.DNDarray of ints
        Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed.

    Examples
    --------
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    >>> a = ht.random.randn(3,3)
    >>> a
    tensor([[-0.5631, -0.8923, -0.0583],
    [-0.1955, -0.9656,  0.4224],
    [ 0.2673, -0.4212, -0.5107]])
    >>> ht.argmin(a)
    tensor([4])
    >>> ht.argmin(a, axis=0)
    tensor([[0, 1, 2]])
    >>> ht.argmin(a, axis=1)
    tensor([[1],
            [1],
            [2]])
    """
    def local_argmin(*args, **kwargs):
        axis = kwargs.get('dim', -1)
        shape = x.shape

        # case where the argmin axis is set to None
        # argmin will be the flattened index, computed standalone and the actual minimum value obtain separately
        if len(args) <= 1 and axis < 0:
            indices = torch.argmin(*args, **kwargs).reshape(1)
            minimums = args[0].flatten()[indices]

            # artificially flatten the input tensor shape to correct the offset computation
            axis = x.split
            shape = [np.prod(shape)]
        # usual case where indices and minimum values are both returned. Axis is not equal to None
        else:
            minimums, indices = torch.min(*args, **kwargs)

        # add offset of data chunks if reduction is computed across split axis
        if axis == x.split:
            offset, _, _ = x.comm.chunk(shape, x.split)
            indices += offset

        return torch.cat([minimums.double(), indices.double()])

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError('axis must be None or int, but was {}'.format(type(axis)))

    # perform the global reduction
    reduced_result = operations.__reduce_op(x, local_argmin, MPI_ARGMIN, axis=axis, out=None, **kwargs)

    # correct the tensor
    reduced_result._DNDarray__array = reduced_result._DNDarray__array.chunk(2)[-1].type(torch.int64)
    reduced_result._DNDarray__dtype = types.int64

    # address lshape/gshape mismatch when axis is 0
    if axis is not None:
        if isinstance(axis, int):
            axis = (axis,)
        if 0 in axis:
            reduced_result._DNDarray__gshape = (1,) + reduced_result._DNDarray__gshape
            if not kwargs.get('keepdim'):
                reduced_result = reduced_result.squeeze(axis=0)

    # set out parameter correctly, i.e. set the storage correctly
    if out is not None:
        if out.shape != reduced_result.shape:
            raise ValueError('Expecting output buffer of shape {}, got {}'.format(reduced_result.shape, out.shape))
        out._DNDarray__array.storage().copy_(reduced_result._DNDarray__array.storage())
        out._DNDarray__array = out._DNDarray__array.type(torch.int64)
        out._DNDarray__dtype = types.int64
        return out

    return reduced_result


def max(x, axis=None, out=None, keepdim=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the maximum along a given axis.

    Parameters
    ----------
    a : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ht.DNDarray, optional
        Tuple of two output tensors (max, max_indices). Must be of the same shape and buffer length as the expected
        output. The minimum value of an output element. Must be present to allow computation on empty slice.

    Returns
    -------
    maximums : ht.DNDarray
        The maximum along a given axis.

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

    return operations.__reduce_op(x, local_max, MPI.MAX, axis=axis, out=out, keepdim=keepdim)


def mean(x, axis=None):
    """
    Calculates and returns the mean of a tensor.
    If a axis is given, the mean will be taken in that direction.

    Parameters
    ----------
    x : ht.DNDarray
        Values for which the mean is calculated for.
    axis : None, Int, iterable, defaults to None
        Axis which the mean is taken in. Default None calculates mean of all data items.

    Returns
    -------
    means : ht.DNDarray
        The mean/s, if split, then split in the same direction as x.

    Examples
    --------
    >>> a = ht.random.randn(1,3)
    >>> a
    tensor([[-1.2435,  1.1813,  0.3509]])
    >>> ht.mean(a)
    tensor(0.0962)

    >>> a = ht.random.randn(4,4)
    >>> a
    tensor([[ 0.0518,  0.9550,  0.3755,  0.3564],
            [ 0.8182,  1.2425,  1.0549, -0.1926],
            [-0.4997, -1.1940, -0.2812,  0.4060],
            [-1.5043,  1.4069,  0.7493, -0.9384]])
    >>> ht.mean(a, 1)
    tensor([ 0.4347,  0.7307, -0.3922, -0.0716])
    >>> ht.mean(a, 0)
    tensor([-0.2835,  0.6026,  0.4746, -0.0921])

    >>> a = ht.random.randn(4,4)
    >>> a
    tensor([[ 2.5893,  1.5934, -0.2870, -0.6637],
            [-0.0344,  0.6412, -0.3619,  0.6516],
            [ 0.2801,  0.6798,  0.3004,  0.3018],
            [ 2.0528, -0.1121, -0.8847,  0.8214]])
    >>> ht.mean(a, (0,1))
    tensor(0.4730)
    """

    def reduce_means_elementwise(output_shape_i):
        """
        Function to combine the calculated means together.
        This does an element-wise update of the calculated means to merge them together using the merge_means function.
        This function operates using x from the mean function paramters.

        Parameters
        ----------
        output_shape_i : iterable
            Iterable with the dimensions of the output of the mean function.

        Returns
        -------
        means : ht.DNDarray
            The calculated means.
        """
        if x.lshape[x.split] != 0:
            mu = torch.mean(x._DNDarray__array, out=None, dim=axis)
        else:
            mu = factories.zeros(output_shape_i)

        mu_shape = list(mu.shape) if list(mu.shape) else [1]

        mu_tot = factories.zeros(([x.comm.size] + mu_shape))
        n_tot = factories.zeros(x.comm.size)
        mu_tot[x.comm.rank, :] = mu
        n_tot[x.comm.rank] = float(x.lshape[x.split])
        x.comm.Allreduce(MPI.IN_PLACE, mu_tot, MPI.SUM)
        x.comm.Allreduce(MPI.IN_PLACE, n_tot, MPI.SUM)

        for i in range(1, x.comm.size):
            mu_tot[0, :],  n_tot[0] = merge_means(mu_tot[0, :], n_tot[0], mu_tot[i, :], n_tot[i])

        return mu_tot[0]

    if axis is None:
        # full matrix calculation
        if not x.is_distributed():
            # if x is not distributed do a torch.mean on x
            ret = torch.mean(x._DNDarray__array)
            return dndarray.DNDarray(ret, tuple(ret.shape), x.dtype, None, x.device, x.comm)
        else:
            # if x is distributed and no axis is given: return mean of the whole set
            if x.lshape[x.split] != 0:
                mu_in = operations.__local_op(torch.mean, x, out=None)
            else:
                mu_in = 0
            n = x.lnumel
            mu_tot = factories.zeros((x.comm.size, 2))
            mu_proc = factories.zeros((x.comm.size, 2))
            mu_proc[x.comm.rank][0] = mu_in
            mu_proc[x.comm.rank][1] = float(n)
            x.comm.Allreduce(mu_proc, mu_tot, MPI.SUM)

            for i in range(1, x.comm.size):
                merged = merge_means(mu_tot[0, 0], mu_tot[0, 1], mu_tot[i, 0], mu_tot[i, 1])
                mu_tot[0, 0] = merged[0]
                mu_tot[0, 1] = merged[1]

            return mu_tot[0][0]
    else:
        output_shape = list(x.shape)
        if isinstance(axis, (list, tuple, dndarray.DNDarray, torch.Tensor)):
            if any([not isinstance(j, int) for j in axis]):
                raise ValueError("items in axis itterable must be integers, axes: {}".format([type(q) for q in axis]))
            if any(d > len(x.shape) for d in axis):
                raise ValueError("axes (axis) must be < {}, currently are {}".format(len(x.shape), axis))
            if any(d < 0 for d in axis):
                axis = [j % len(x.shape) for j in axis]

            output_shape = [output_shape[it] for it in range(len(output_shape)) if it not in axis]
            # multiple dimensions
            if x.split is None:
                return dndarray.DNDarray(torch.mean(x._DNDarray__array, dim=axis),
                                         tuple(output_shape), x.dtype, x.split, x.device, x.comm)

            if x.split in axis:
                # merge in the direction of the split
                return reduce_means_elementwise(output_shape)
            else:
                # multiple dimensions which does *not* include the split axis
                # combine along the split axis
                return dndarray.DNDarray(torch.mean(x._DNDarray__array, dim=axis),
                                         tuple(output_shape), x.dtype,
                                         x.split if x.split < len(output_shape) else len(output_shape) - 1,
                                         x.device, x.comm)
        elif isinstance(axis, int):
            if axis >= len(x.shape):
                raise ValueError("axis (axis) must be < {}, currently is {}".format(len(x.shape), axis))
            axis = axis if axis > 0 else axis % len(x.shape)

            # only one axis given
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]
            output_shape = output_shape if output_shape else (1, )

            if x.split is None:
                # return operations.__local_op(torch.mean, x, out=None, **{'dim': axis})
                return dndarray.DNDarray(torch.mean(x._DNDarray__array, dim=axis),
                                         tuple(output_shape), x.dtype, x.split, x.device, x.comm)
            elif axis == x.split:
                return reduce_means_elementwise(output_shape)
            else:
                # singular axis given (axis) not equal to split direction (x.split)
                return dndarray.DNDarray(torch.mean(x._DNDarray__array, dim=axis), tuple(output_shape), x.dtype,
                                         x.split if x.split < len(output_shape) else len(output_shape) - 1, x.device, x.comm)
        else:
            raise TypeError(
                'axis (axis) must be an int or a list, ht.DNDarray, torch.Tensor, or tuple, but was {}'.format(type(axis)))


def merge_means(mu1, n1, mu2, n2):
    """
    Function to merge two means by pairwise update.
    **Note** all tensors/arrays must be either the same size or individual values (can be mixed, i.e. n can be a float)

    Parameters
    ----------
    mu1 : ht.DNDarray, torch.tensor, float, int
        Calculated mean
    n1 : ht.DNDarray, torch.tensor, float
        number of elements used to calculate mu1
    mu2 : ht.DNDarray, torch.tensor, float, int
        Calculated mean
    n2 : ht.DNDarray, torch.tensor, float
        number of elements used to calculate mu2

    Returns
    -------
    combined_set_count : int
        Number of elements in the combined set.

    References
    ----------
    [1] J. Bennett, R. Grout, P. Pebay, D. Roe, D. Thompson, Numerically stable, single-pass, parallel statistics
        algorithms, IEEE International Conference on Cluster Computing and Workshops, 2009, Oct 2009, New Orleans, LA,
        USA.
    """
    return mu1 + n2 * ((mu2 - mu1) / (n1 + n2)), n1 + n2


def merge_vars(var1, mu1, n1, var2, mu2, n2, bessel=True):
    """
    Function to merge two variances by pairwise update.
    **Note** this is a parallel of the merge_means function
    **Note pt2.** all tensors/arrays must be either the same size or individual values

    Parameters
    ----------
    var1 : ht.DNDarray, torch.tensor, float, int
        Variance.
    mu1 : ht.DNDarray, torch.tensor, float, int
        Calculated mean.
    n1 : ht.DNDarray, torch.tensor, float, int
        Number of elements used to calculate mu1.
    var2 : ht.DNDarray, torch.tensor, float, int
        Variance.
    mu2 : ht.DNDarray, torch.tensor, float, int
        Calculated mean.
    n2 : ht.DNDarray, torch.tensor, float, int
        Number of elements used to calculate mu2.
    bessel : bool
        Flag for the use of the bessel correction

    Returns
    -------
    combined_set_count : int
        Number of elements in the combined set.

    References
    ----------
    [1] J. Bennett, R. Grout, P. Pebay, D. Roe, D. Thompson, Numerically stable, single-pass, parallel statistics
        algorithms, IEEE International Conference on Cluster Computing and Workshops, 2009, Oct 2009, New Orleans, LA,
        USA.
    """
    n = n1 + n2
    delta = mu2 - mu1
    if bessel:
        return (var1 * (n1 - 1) + var2 * (n2 - 1) + (delta ** 2) * n1 * n2 / n) / (n - 1), mu1 + n2 * (delta / (n1 + n2)), n
    else:
        return (var1 * n1 + var2 * n2 + (delta ** 2) * n1 * n2 / n) / n, mu1 + n2 * (delta / (n1 + n2)), n


def min(x, axis=None, out=None, keepdim=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    a : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ht.DNDarray, optional
        Tuple of two output tensors (min, min_indices). Must be of the same shape and buffer length as the expected
        output. The maximum value of an output element. Must be present to allow computation on empty slice.

    Returns
    -------
    minimums : ht.DNDarray
        The minimums along a given axis.

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

    return operations.__reduce_op(x, local_min, MPI.MIN, axis=axis, out=out, keepdim=keepdim)


def mpi_argmax(a, b, _):
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0],), dim=1)
    indices = torch.stack((lhs.chunk(2)[1], rhs.chunk(2)[1],), dim=1)

    # determine the minimum value and select the indices accordingly
    max, max_indices = torch.max(values, dim=1)
    result = torch.cat((max, indices[torch.arange(values.shape[0]), max_indices],))

    rhs.copy_(result)


MPI_ARGMAX = MPI.Op.Create(mpi_argmax, commute=True)


def mpi_argmin(a, b, _):
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0],), dim=1)
    indices = torch.stack((lhs.chunk(2)[1], rhs.chunk(2)[1],), dim=1)

    # determine the minimum value and select the indices accordingly
    min, min_indices = torch.min(values, dim=1)
    result = torch.cat(
        (min, indices[torch.arange(values.shape[0]), min_indices],))

    rhs.copy_(result)


MPI_ARGMIN = MPI.Op.Create(mpi_argmin, commute=True)


def std(x, axis=None, bessel=True):
    """
    Calculates and returns the standard deviation of a tensor with the bessel correction.
    If a axis is given, the variance will be taken in that direction.

    Parameters
    ----------
    x : ht.DNDarray
        Values for which the std is calculated for.
    axis : None, int, defaults to None
        Axis which the mean is taken in. Default None calculates the standard deviation of all data items.
    bessel : bool, defaults to True
        Use the bessel correction when calculating the variance/std. Toggles between unbiased and biased calculation of
        the standard deviation.

    Returns
    -------
    stds : ht.DNDarray
        The std/s, if split, then split in the same direction as x.

    Examples
    --------
    >>> a = ht.random.randn(1,3)
    >>> a
    tensor([[ 0.3421,  0.5736, -2.2377]])
    >>> ht.std(a)
    tensor(1.5606)
    >>> a = ht.random.randn(4,4)
    >>> a
    tensor([[-1.0206,  0.3229,  1.1800,  1.5471],
            [ 0.2732, -0.0965, -0.1087, -1.3805],
            [ 0.2647,  0.5998, -0.1635, -0.0848],
            [ 0.0343,  0.1618, -0.8064, -0.1031]])
    >>> ht.std(a, 0)
    tensor([0.6157, 0.2918, 0.8324, 1.1996])
    >>> ht.std(a, 1)
    tensor([1.1405, 0.7236, 0.3506, 0.4324])
    >>> ht.std(a, 1, bessel=False)
    tensor([0.9877, 0.6267, 0.3037, 0.3745])
    """
    if not axis:
        return np.sqrt(var(x, axis, bessel))
    else:
        return exponential.sqrt(var(x, axis, bessel), out=None)


def var(x, axis=None, bessel=True):
    """
    Calculates and returns the variance of a tensor. If an axis is given, the variance will be taken in that direction.

    Parameters
    ----------
    x : ht.DNDarray
        Values for which the variance is calculated for.
    axis : None, int, defaults to None
        Axis which the variance is taken in. Default None calculates the variance of all data items.
    bessel : bool, defaults to True
        Use the bessel correction when calculating the variance/std. Toggles between unbiased and biased calculation of
        the variance.

    Returns
    -------
    variances : ht.DNDarray
        The var/s, if split, then split in the same direction as x.

    Examples
    --------
    >>> a = ht.random.randn(1,3)
    >>> a
    tensor([[-1.9755,  0.3522,  0.4751]])
    >>> ht.var(a)
    tensor(1.9065)

    >>> a = ht.random.randn(4,4)
    >>> a
    tensor([[-0.8665, -2.6848, -0.0215, -1.7363],
            [ 0.5886,  0.5712,  0.4582,  0.5323],
            [ 1.9754,  1.2958,  0.5957,  0.0418],
            [ 0.8196, -1.2911, -0.2026,  0.6212]])
    >>> ht.var(a, 1)
    tensor([1.3092, 0.0034, 0.7061, 0.9217])
    >>> ht.var(a, 0)
    tensor([1.3624, 3.2563, 0.1447, 1.2042])
    >>> ht.var(a, 0, bessel=True)
    tensor([1.3624, 3.2563, 0.1447, 1.2042])
    >>> ht.var(a, 0, bessel=False)
    tensor([1.0218, 2.4422, 0.1085, 0.9032])
    """
    if not isinstance(bessel, bool):
        raise TypeError('bessel must be a boolean, currently is {}'.format(type(bessel)))

    def reduce_vars_elementwise(output_shape_i):
        """
        Function to combine the calculated vars together. This does an element-wise update of the calculated vars to
        merge them together using the merge_vars function. This function operates using x from the var function
        parameters.

        Parameters
        ----------
        output_shape_i : iterable
            Iterable with the dimensions of the output of the var function.

        Returns
        -------
        variances : ht.DNDarray
            The calculated variances.
        """

        if x.lshape[x.split] != 0:
            mu = operations.__local_op(torch.mean, x, out=None, dim=axis)
            var = torch.var(x._DNDarray__array, out=None, dim=axis, unbiased=bessel)
        else:
            mu = factories.zeros(output_shape_i)
            var = factories.zeros(output_shape_i)

        n_for_merge = factories.zeros(x.comm.size)
        n2 = factories.zeros(x.comm.size)
        n2[x.comm.rank] = x.lshape[x.split]
        x.comm.Allreduce(n2, n_for_merge, MPI.SUM)

        var_shape = list(var.shape) if list(var.shape) else [1]

        var_tot = factories.zeros(([x.comm.size, 2] + var_shape))
        n_tot = factories.zeros(x.comm.size)
        var_tot[x.comm.rank, 0, :] = var
        var_tot[x.comm.rank, 1, :] = mu
        n_tot[x.comm.rank] = float(x.lshape[x.split])
        x.comm.Allreduce(MPI.IN_PLACE, var_tot, MPI.SUM)
        x.comm.Allreduce(MPI.IN_PLACE, n_tot, MPI.SUM)

        for i in range(1, x.comm.size):
            var_tot[0, 0, :], var_tot[0, 1, :], n_tot[0] = merge_vars(var_tot[0, 0, :], var_tot[0, 1, :], n_tot[0],
                                                                      var_tot[i, 0, :], var_tot[i, 1, :], n_tot[i])
        return var_tot[0, 0, :]
    # ----------------------------------------------------------------------------------------------------
    if axis is None:  # no axis given
        if not x.is_distributed():  # not distributed (full tensor on one node)
            ret = torch.var(x._DNDarray__array, unbiased=bessel)
            return dndarray.DNDarray(ret, tuple(ret.shape), x.dtype, None, x.device, x.comm)

        else:  # case for full matrix calculation (axis is None)
            if x.lshape[x.split] != 0:
                mu_in = operations.__local_op(torch.mean, x, out=None)
                var_in = operations.__local_op(torch.var, x, out=None, unbiased=bessel)
                if torch.isnan(var_in._DNDarray__array):
                    var_in = 0.0
            else:
                mu_in = 0
                var_in = 0
            n = x.lnumel
            var_tot = factories.zeros((x.comm.size, 3))
            var_proc = factories.zeros((x.comm.size, 3))
            var_proc[x.comm.rank][0] = var_in
            var_proc[x.comm.rank][1] = mu_in
            var_proc[x.comm.rank][2] = float(n)
            x.comm.Allreduce(var_proc, var_tot, MPI.SUM)

            for i in range(1, x.comm.size):
                merged = merge_vars(var_tot[0, 0], var_tot[0, 1], var_tot[0, 2], var_tot[i, 0], var_tot[i, 1], var_tot[i, 2])
                var_tot[0, 0] = merged[0]
                var_tot[0, 1] = merged[1]
                var_tot[0, 2] = merged[2]

            return var_tot[0][0]

    else:  # axis is given
        # case for var in one dimension
        output_shape = list(x.shape)
        if isinstance(axis, int):
            if axis >= len(x.shape):
                raise ValueError("axis must be < {}, currently is {}".format(len(x.shape), axis))
            axis = axis if axis > 0 else axis % len(x.shape)
            # only one axis given
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]
            output_shape = output_shape if output_shape else (1,)

            if x.split is None:  # x is *not* distributed -> no need to distributed
                return dndarray.DNDarray(torch.var(x._DNDarray__array, dim=axis, unbiased=bessel),
                                         tuple(output_shape), x.dtype, None, x.device, x.comm)
            elif axis == x.split:  # x is distributed and axis chosen is == to split
                return reduce_vars_elementwise(output_shape)
            else:
                # singular axis given (axis) not equal to split direction (x.split)
                lcl = torch.var(x._DNDarray__array, dim=axis, keepdim=True)
                return dndarray.DNDarray(lcl, tuple(output_shape), x.dtype, x.split if x.split < len(output_shape) else len(output_shape) - 1,
                                         x.device, x.comm)
        else:
            raise TypeError(
                'axis (axis) must be an int, currently is {}. Check if multidim var is available in PyTorch'.format(type(axis)))
