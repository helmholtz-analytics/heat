import torch

from .communication import MPI
from . import types
from . import tensor
from . import exponential
from .operations import __local_operation as local_op
from .operations import __reduce_op as reduce_op


__all__ = [
    'mean',
    'std',
    'sum',
    'var'
]


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
            mu = local_op(torch.mean, x, out=None, dim=axis)
        else:
            mu = tensor.zeros(output_shape_i)

        n_for_merge = tensor.zeros(x.comm.size)
        n2 = tensor.zeros(x.comm.size)
        n2[x.comm.rank] = x.lshape[x.split]
        x.comm.Allreduce(n2, n_for_merge, MPI.SUM)

        sz = x.comm.size
        rem1, rem2 = 0, 0

        mu_reshape = tensor.zeros((x.comm.size, int(np.prod(mu.lshape))))
        mu_reshape[x.comm.rank] = local_op(torch.reshape, mu, out=None, shape=(1, int(mu.lnumel)))
        mu_reshape_combi = tensor.zeros((x.comm.size, int(np.prod(mu.lshape))))
        x.comm.Allreduce(mu_reshape, mu_reshape_combi, MPI.SUM)

        while True:  # todo: multithread for GPU parrallelizm
            if sz % 2 != 0:
                if rem1 and not rem2:
                    rem2 = sz - 1
                elif not rem1:
                    rem1 = sz - 1
            splt = sz // 2
            for sp_it in range(splt):
                for en, (el1, el2) in enumerate(zip(mu_reshape_combi[sp_it, :], mu_reshape_combi[sp_it+splt, :])):
                    try:
                        mu_reshape_combi[sp_it, en], n = merge_means(el1, n_for_merge[sp_it], el2, n_for_merge[sp_it+splt])
                    except IndexError:
                        mu_reshape_combi, n = merge_means(el1, n_for_merge[sp_it], el2, n_for_merge[sp_it + splt])
                n_for_merge[sp_it] = n
            if rem1 and rem2:
                for en, (el1, el2) in enumerate(zip(mu_reshape_combi[rem1, :], mu_reshape_combi[rem2, :])):
                    mu_reshape_combi[rem2, en], n = merge_means(el1, n_for_merge[rem1], el2, n_for_merge[rem2])
                n_for_merge[rem2] = n

                rem1 = rem2
                rem2 = 0
            sz = splt
            if sz == 1 or sz == 0:
                if rem1:
                    for en, (el1, el2) in enumerate(zip(mu_reshape_combi[0, :], mu_reshape_combi[rem1, :])):
                        mu_reshape_combi[0, en], _ = merge_means(el1, n_for_merge[0], el2, n_for_merge[rem1])

                ret = local_op(torch.reshape, mu_reshape_combi[0], out=None, shape=output_shape_i)
                return ret
    # ------------------------------------------------------------------------------------------------------------------
    if axis is None:
        # full matrix calculation
        if x.split:
            # if x is distributed and no axis is given: return mean of the whole set
            if x.lshape[x.split] != 0:
                mu_in = local_op(torch.mean, x, out=None)
            else:
                mu_in = 0
            n = x.lnumel
            mu_tot = tensor.zeros((x.comm.size, 2))
            mu_proc = tensor.zeros((x.comm.size, 2))
            mu_proc[x.comm.rank][0] = mu_in
            mu_proc[x.comm.rank][1] = float(n)
            x.comm.Allreduce(mu_proc, mu_tot, MPI.SUM)

            rem1 = 0
            rem2 = 0
            sz = mu_tot.shape[0]
            while True:  # this loop will loop pairwise over the whole process and do pairwise updates
                # likely: do not need to parallelize: (likely) will not be worth it (can be tested)
                if sz % 2 != 0:
                    if rem1 and not rem2:
                        rem2 = sz - 1
                    elif not rem1:
                        rem1 = sz - 1
                splt = sz // 2
                for i in range(splt):
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
            return local_op(torch.mean, x, out=None)
    else:
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
                return local_op(torch.mean, x, out=None, **{'dim': axis})
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it not in axis]
            if x.split in axis:
                # merge in the direction of the split
                return reduce_means_elementwise(output_shape)

            else:
                # multiple dimensions which does *not* include the split axis
                # combine along the split axis
                try:
                    return tensor.array(local_op(torch.mean, x, out=None, dim=axis), split=x.split, comm=x.comm)
                except ValueError:
                    return local_op(torch.mean, x, out=None, dim=axis)
        elif isinstance(axis, int):
            if axis >= len(x.shape):
                raise ValueError("axis (axis) must be < {}, currently is {}".format(len(x.shape), axis))
            axis = axis if axis > 0 else axis % len(x.shape)

            # only one axis given
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]

            if x.split is None:
                return local_op(torch.mean, x, out=None, **{'dim': axis})
            if axis == x.split:
                return reduce_means_elementwise(output_shape)
            else:
                # singular axis given (axis) not equal to split direction (x.split)
                # local operation followed by array creation to create the full tensor of the means
                try:
                    return tensor.array(local_op(torch.mean, x, out=None, dim=axis), split=x.split, comm=x.comm)
                except ValueError:
                    return local_op(torch.mean, x, out=None, dim=axis)
        else:
            raise TypeError("axis (axis) must be an int or a list, ht.tensor, torch.Tensor, or tuple, currently is {}".format(type(axis)))


def merge_vars(var1, mu1, n1, var2, mu2, n2, bessel=True):
    """
    Function to merge two variances by pairwise update
    **Note** this is a parallel of the merge_means function

    Parameters
    ----------
    var1 : 1D ht.tensor or 1D torch.tensor
        variance
    mu1 : 1D ht.tensor or 1D torch.tensor
        Calculated mean
    n1 : 1D ht.tensor or 1D torch.tensor
        number of elements used to calculate mu1
    var2 : 1D ht.tensor or 1D torch.tensor
        variance
    mu2 : 1D ht.tensor or 1D torch.tensor
        Calculated mean
    n2 : 1D ht.tensor or 1D torch.tensor
        number of elements used to calculate mu2
    bessel : Bool
        flag for the use of the bessel correction

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
        Values for which the variance is calculated for
    axis : None, Int
        axis which the variance is taken in.
        Default: None -> var of all data calculated
        NOTE -> if multidemensional var is implemented in pytorch, this can be an iterable. Only thing which muse be changed is the raise
    bessel : Bool
        Default: True
        use the bessel correction when calculating the varaince/std
        toggle between unbiased and biased calculation of the std

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
            mu = local_op(torch.mean, x, out=None, dim=axis)
            var = local_op(torch.var, x, out=None, dim=axis, unbiased=bessel)
        else:
            mu = tensor.zeros(output_shape_i)
            var = tensor.zeros(output_shape_i)

        n_for_merge = tensor.zeros(x.comm.size)
        n2 = tensor.zeros(x.comm.size)
        n2[x.comm.rank] = x.lshape[x.split]
        x.comm.Allreduce(n2, n_for_merge, MPI.SUM)

        sz = x.comm.size
        rem1, rem2 = 0, 0

        mu_reshape = tensor.zeros((x.comm.size, int(np.prod(mu.lshape))))
        mu_reshape[x.comm.rank] = local_op(torch.reshape, mu, out=None, shape=(1, int(mu.lnumel)))
        mu_reshape_combi = tensor.zeros((x.comm.size, int(np.prod(mu.lshape))))
        x.comm.Allreduce(mu_reshape, mu_reshape_combi, MPI.SUM)

        var_reshape = tensor.zeros((x.comm.size, int(np.prod(var.lshape))))
        var_reshape[x.comm.rank] = local_op(torch.reshape, var, out=None, shape=(1, int(var.lnumel)))
        var_reshape_combi = tensor.zeros((x.comm.size, int(np.prod(var.lshape))))
        x.comm.Allreduce(var_reshape, var_reshape_combi, MPI.SUM)

        # TODO: gpu init:
        # if x.device == gpu or if gpu is available:
        #   (if the device is not on the gpu need to copy/create reshape_compi there)
        #   create a gpu tensor for the reshape_combi ones
        #   do the loops

        while True:  # todo: multithread for GPU
            if sz % 2 != 0:
                if rem1 and not rem2:
                    rem2 = sz - 1
                elif not rem1:
                    rem1 = sz - 1
            splt = sz // 2
            for i in range(splt):

                for en, (mu1, var1, mu2, var2) in enumerate(zip(mu_reshape_combi[i], var_reshape_combi[i], mu_reshape_combi[i + splt], var_reshape_combi[i + splt])):
                    try:
                        var_reshape_combi[i, en], mu_reshape_combi[i, en], n = merge_vars(var1, mu1, n_for_merge[i], var2, mu2, n_for_merge[i+splt], bessel)
                    except ValueError:
                        var_reshape_combi, mu_reshape_combi, n = merge_vars(var1, mu1, n_for_merge[i], var2, mu2, n_for_merge[i + splt], bessel)

                n_for_merge[i] = n
            if rem1 and rem2:
                for en, (mu1, var1, mu2, var2) in enumerate(zip(mu_reshape_combi[rem1], var_reshape_combi[rem1], mu_reshape_combi[rem2], var_reshape_combi[rem2])):
                    var_reshape_combi[rem2], mu_reshape_combi[rem2], n = merge_vars(var1, mu1, n_for_merge[rem1], var2, mu2, n_for_merge[rem2], bessel)
                n_for_merge[rem2] = n

                rem1 = rem2
                rem2 = 0
            sz = splt
            if sz == 1 or sz == 0:
                if rem1:
                    for en, (mu1, var1, mu2, var2) in enumerate(zip(mu_reshape_combi[0], var_reshape_combi[0], mu_reshape_combi[rem1], var_reshape_combi[rem1])):
                        var_reshape_combi[0], mu_reshape_combi[0], n = merge_vars(var1, mu1, n_for_merge[0], var2, mu2, n_for_merge[rem1], bessel)
                ret = local_op(torch.reshape, var_reshape_combi[0], out=None, shape=output_shape_i)
                # TODO: this must return a split tensor with the same split dimension
                # what should the split dimension be?
                return ret
    # ----------------------------------------------------------------------------------------------------
    if axis is None:
        # case for full matrix calculation (axis is None)
        if x.split is not None:
            if x.lshape[x.split] != 0:
                mu_in = local_op(torch.mean, x, out=None)
                var_in = local_op(torch.var, x, out=None, unbiased=bessel)
            else:
                mu_in = 0
                var_in = 0
            n = x.lnumel
            var_tot = tensor.zeros((x.comm.size, 3))
            var_proc = tensor.zeros((x.comm.size, 3))
            var_proc[x.comm.rank][0] = var_in
            var_proc[x.comm.rank][1] = mu_in
            var_proc[x.comm.rank][2] = float(n)
            x.comm.Allreduce(var_proc, var_tot, MPI.SUM)

            rem1 = 0
            rem2 = 0
            sz = var_tot.shape[0]
            while True:  # this loop will loop pairwise over the processes and do pairwise updates
                if sz % 2 != 0:
                    if rem1 and not rem2:
                        rem2 = sz - 1
                    elif not rem1:
                        rem1 = sz - 1
                splt = sz // 2
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
            ret = local_op(torch.var, x, out=None, unbiased=bessel)
    else:
        # case for var in one dimension
        output_shape = list(x.shape)
        if isinstance(axis, int):
            if axis >= len(x.shape):
                raise ValueError("axis must be < {}, currently is {}".format(len(x.shape), axis))
            axis = axis if axis > 0 else axis % len(x.shape)
            # only one axis given
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]
            if x.split != x.split:
                ret = local_op(torch.var, x, out=None, dim=axis, unbiased=bessel)
            else:
                ret = reduce_vars_elementwise(output_shape)
        else:
            raise TypeError("Axis (axis) must be an int, currently is {}. Check if multidim var is available in pyTorch".format(type(axis)))

    return tensor.tensor(ret, gshape=output_shape, dtype=types.float,
                         split=x.split, device=x.device, comm=x.comm)


def std(x, axis=None, bessel=True):
    """
    Calculates and returns the standard deviation of a tensor with the bessel correction
    If a axis is given, the variance will be taken in that direction.

    Parameters
    ----------
    x : ht.tensor
        Values for which the std is calculated for
    axis : None, Int
        axis which the mean is taken in.
        Default: None -> std of all data calculated
        NOTE -> if multidemensional var is implemented in pytorch, this can be an iterable. Only thing which muse be changed is the raise
    bessel : Bool
        Default: True
        use the bessel correction when calculating the varaince/std
        toggle between unbiased and biased calculation of the std

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

    Returns
    -------
    ht.tensor containing the std/s, if split, then split in the same direction as x.
    """
    if not axis:
        return np.sqrt(var(x, axis, bessel))
    else:
        return exponential.sqrt(var(x, axis, bessel), out=None)


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
    return reduce_op(x, torch.sum, MPI.SUM, axis, out)
