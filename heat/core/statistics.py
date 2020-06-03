import numpy as np
import torch

from .communication import MPI
from . import exponential
from . import factories
from . import linalg
from . import manipulations
from . import operations
from . import dndarray
from . import types
from . import stride_tricks
from . import logical
from . import constants


__all__ = [
    "argmax",
    "argmin",
    "average",
    "cov",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "std",
    "var",
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
        axis = kwargs.get("dim", -1)
        shape = x.shape

        # case where the argmax axis is set to None
        # argmax will be the flattened index, computed standalone and the actual maximum value obtain separately
        if len(args) <= 1 and axis < 0:
            indices = torch.argmax(*args, **kwargs).reshape(1)
            maxima = args[0].flatten()[indices]
            # artificially flatten the input tensor shape to correct the offset computation
            axis = 0
            shape = [np.prod(shape)]
        # usual case where indices and maximum values are both returned. Axis is not equal to None
        else:
            maxima, indices = torch.max(*args, **kwargs)

        # add offset of data chunks if reduction is computed across split axis
        if axis == x.split:
            offset, _, _ = x.comm.chunk(shape, x.split)
            indices += torch.tensor(offset, dtype=indices.dtype)

        return torch.cat([maxima.double(), indices.double()])

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError("axis must be None or int, but was {}".format(type(axis)))

    # perform the global reduction
    smallest_value = -constants.sanitize_infinity(x._DNDarray__array.dtype)
    reduced_result = operations.__reduce_op(
        x, local_argmax, MPI_ARGMAX, axis=axis, out=None, neutral=smallest_value, **kwargs
    )

    # correct the tensor
    reduced_result._DNDarray__array = reduced_result._DNDarray__array.chunk(2)[-1].type(torch.int64)
    reduced_result._DNDarray__dtype = types.int64

    # address lshape/gshape mismatch when axis is 0
    if axis is not None:
        if isinstance(axis, int):
            axis = (axis,)
        if 0 in axis:
            reduced_result._DNDarray__gshape = (1,) + reduced_result._DNDarray__gshape
            if not kwargs.get("keepdim"):
                reduced_result = reduced_result.squeeze(axis=0)

    # set out parameter correctly, i.e. set the storage correctly
    if out is not None:
        if out.shape != reduced_result.shape:
            raise ValueError(
                "Expecting output buffer of shape {}, got {}".format(
                    reduced_result.shape, out.shape
                )
            )
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
    out : ht.DNDarray, optional. Issue #100
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
        axis = kwargs.get("dim", -1)
        shape = x.shape

        # case where the argmin axis is set to None
        # argmin will be the flattened index, computed standalone and the actual minimum value obtain separately
        if len(args) <= 1 and axis < 0:
            indices = torch.argmin(*args, **kwargs).reshape(1)
            minimums = args[0].flatten()[indices]

            # artificially flatten the input tensor shape to correct the offset computation
            axis = 0
            shape = [np.prod(shape)]
        # usual case where indices and minimum values are both returned. Axis is not equal to None
        else:
            minimums, indices = torch.min(*args, **kwargs)

        # add offset of data chunks if reduction is computed across split axis
        if axis == x.split:
            offset, _, _ = x.comm.chunk(shape, x.split)
            indices += torch.tensor(offset, dtype=indices.dtype)

        return torch.cat([minimums.double(), indices.double()])

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError("axis must be None or int, but was {}".format(type(axis)))

    # perform the global reduction
    largest_value = constants.sanitize_infinity(x._DNDarray__array.dtype)
    reduced_result = operations.__reduce_op(
        x, local_argmin, MPI_ARGMIN, axis=axis, out=None, neutral=largest_value, **kwargs
    )

    # correct the tensor
    reduced_result._DNDarray__array = reduced_result._DNDarray__array.chunk(2)[-1].type(torch.int64)
    reduced_result._DNDarray__dtype = types.int64

    # address lshape/gshape mismatch when axis is 0
    if axis is not None:
        if isinstance(axis, int):
            axis = (axis,)
        if 0 in axis:
            reduced_result._DNDarray__gshape = (1,) + reduced_result._DNDarray__gshape
            if not kwargs.get("keepdim"):
                reduced_result = reduced_result.squeeze(axis=0)

    # set out parameter correctly, i.e. set the storage correctly
    if out is not None:
        if out.shape != reduced_result.shape:
            raise ValueError(
                "Expecting output buffer of shape {}, got {}".format(
                    reduced_result.shape, out.shape
                )
            )
        out._DNDarray__array.storage().copy_(reduced_result._DNDarray__array.storage())
        out._DNDarray__array = out._DNDarray__array.type(torch.int64)
        out._DNDarray__dtype = types.int64
        return out

    return reduced_result


def average(x, axis=None, weights=None, returned=False):
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    x : ht.tensor
        Tensor containing data to be averaged.

    axis : None or int or tuple of ints, optional
        Axis or axes along which to average x.  The default,
        axis=None, will average over all of the elements of the input tensor.
        If axis is negative it counts from the last to the first axis.

        #TODO Issue #351: If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.

    weights : ht.tensor, optional
        An tensor of weights associated with the values in x. Each value in
        x contributes to the average according to its associated weight.
        The weights tensor can either be 1D (in which case its length must be
        the size of x along the given axis) or of the same shape as x.
        If weights=None, then all data in x are assumed to have a
        weight equal to one, the result is equivalent to ht.mean(x).

    returned : bool, optional
        Default is False. If True, the tuple (average, sum_of_weights)
        is returned, otherwise only the average is returned.
        If weights=None, sum_of_weights is equivalent to the number of
        elements over which the average is taken.

    Returns
    -------
    average, [sum_of_weights] : ht.tensor or tuple of ht.tensors
        Return the average along the specified axis. When returned=True,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. sum_of_weights is of the
        same type as `average`.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero.

    TypeError
        When the length of 1D weights is not the same as the shape of x
        along axis.


    Examples
    --------
    >>> data = ht.arange(1,5, dtype=float)
    >>> data
    tensor([1., 2., 3., 4.])
    >>> ht.average(data)
    tensor(2.5000)
    >>> ht.average(ht.arange(1,11, dtype=float), weights=ht.arange(10,0,-1))
    tensor([4.])
    >>> data = ht.array([[0, 1],
                         [2, 3],
                        [4, 5]], dtype=float, split=1)
    >>> weights = ht.array([1./4, 3./4])
    >>> ht.average(data, axis=1, weights=weights)
    tensor([0.7500, 2.7500, 4.7500])
    >>> ht.average(data, weights=weights)
    Traceback (most recent call last):
        ...
    TypeError: Axis must be specified when shapes of x and weights differ.
    """

    # perform sanitation
    if not isinstance(x, dndarray.DNDarray):
        raise TypeError("expected x to be a ht.DNDarray, but was {}".format(type(x)))
    if weights is not None and not isinstance(weights, dndarray.DNDarray):
        raise TypeError("expected weights to be a ht.DNDarray, but was {}".format(type(x)))
    axis = stride_tricks.sanitize_axis(x.shape, axis)

    if weights is None:
        result = mean(x, axis)
        num_elements = x.gnumel / result.gnumel
        cumwgt = factories.empty(1, dtype=result.dtype)
        cumwgt._DNDarray__array = num_elements
    else:
        # Weights sanitation:
        # weights (global) is either same size as x (global), or it is 1D and same size as x along chosen axis
        if x.gshape != weights.gshape:
            if axis is None:
                raise TypeError("Axis must be specified when shapes of x and weights differ.")
            elif isinstance(axis, tuple):
                raise NotImplementedError("Weighted average over tuple axis not implemented yet.")
            if weights.numdims != 1:
                raise TypeError("1D weights expected when shapes of x and weights differ.")
            if weights.gshape[0] != x.gshape[axis]:
                raise ValueError("Length of weights not compatible with specified axis.")

            wgt_lshape = tuple(
                weights.lshape[0] if dim == axis else 1 for dim in list(range(x.numdims))
            )
            wgt_slice = [slice(None) if dim == axis else 0 for dim in list(range(x.numdims))]
            wgt_split = None if weights.split is None else axis
            wgt = factories.empty(wgt_lshape, dtype=weights.dtype, device=x.device)
            wgt._DNDarray__array[wgt_slice] = weights._DNDarray__array
            wgt = factories.array(wgt._DNDarray__array, is_split=wgt_split)
        else:
            if x.comm.is_distributed():
                if x.split is not None and weights.split != x.split and weights.numdims != 1:
                    # fix after Issue #425 is solved
                    raise NotImplementedError(
                        "weights.split does not match data.split: not implemented yet."
                    )
            wgt = factories.empty_like(weights, device=x.device)
            wgt._DNDarray__array = weights._DNDarray__array

        cumwgt = wgt.sum(axis=axis)

        if logical.any(cumwgt == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        result = (x * wgt).sum(axis=axis) / cumwgt

    if returned:
        if cumwgt.gshape != result.gshape:
            cumwgt._DNDarray__array = torch.broadcast_tensors(
                cumwgt._DNDarray__array, result._DNDarray__array
            )[0]
            cumwgt._DNDarray__gshape = result.gshape
            cumwgt._DNDarray__split = result.split
        return (result, cumwgt)

    return result


def cov(m, y=None, rowvar=True, bias=False, ddof=None):
    """
    Estimate the covariance matrix of some data, m. For more imformation on the algorithm please see the numpy function of the same name

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden. Note that ``ddof=1`` will return the unbiased estimate and
        ``ddof=0`` will return the simple average. The default value is ``None``.

    Returns
    -------
    cov : DNDarray
        the covariance matrix of the variables
    """
    if ddof is not None and not isinstance(ddof, int):
        raise TypeError("ddof must be integer")
    if not isinstance(m, dndarray.DNDarray):
        raise TypeError("m must be a DNDarray")
    if not m.is_balanced():
        raise RuntimeError("balance is required for cov(). use balance_() to balance m")
    if m.numdims > 2:
        raise ValueError("m has more than 2 dimensions")

    if m.numdims == 1:
        m = m.expand_dims(1)
    x = m.copy()
    if not rowvar and x.shape[0] != 1:
        x = x.T

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    if y is not None:
        if not isinstance(y, dndarray.DNDarray):
            raise TypeError("y must be a DNDarray")
        if y.numdims > 2:
            raise ValueError("y has too many dimensions, max=2")
        if y.numdims == 1:
            y = y.expand_dims(1)
        if not y.is_balanced():
            raise RuntimeError("balance is required for cov(). use balance_() to balance y")
        if not rowvar and y.shape[0] != 1:
            y = y.T

        x = manipulations.concatenate((x, y), axis=0)

    avg = mean(x, axis=1)
    norm = x.shape[1] - ddof
    # find normalization:
    if norm <= 0:
        raise ValueError("ddof >= number of elements in m, {} {}".format(ddof, m.gnumel))
    x -= avg.expand_dims(1)
    c = linalg.dot(x, x.T)
    c /= norm
    return c


def max(x, axis=None, out=None, keepdim=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the maximum along a given axis.

    Parameters
    ----------
    x : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ht.DNDarray, optional
        Tuple of two output tensors (max, max_indices). Must be of the same shape and buffer length as the expected
        output. The minimum value of an output element. Must be present to allow computation on empty slice.
    keepdim : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original arr.

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
            result = result[0]
        return result

    smallest_value = -constants.sanitize_infinity(x._DNDarray__array.dtype)
    return operations.__reduce_op(
        x, local_max, MPI.MAX, axis=axis, out=out, neutral=smallest_value, keepdim=keepdim
    )


def maximum(x1, x2, out=None):
    """
    Compares two tensors and returns a new tensor containing the element-wise maxima.
    If one of the elements being compared is a NaN, then that element is returned. TODO: Check this: If both elements are NaNs then the first is returned.
    The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.

    Parameters:
    -----------

    x1, x2 : ht.DNDarray
            The tensors containing the elements to be compared. They must have the same shape, or shapes that can be broadcast to a single shape.
            For broadcasting semantics, see: https://pytorch.org/docs/stable/notes/broadcasting.html

    out : ht.DNDarray or None, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated tensor is returned.

    Returns:
    --------

    maximum: ht.DNDarray
            Element-wise maximum of the two input tensors.

    Examples:
    ---------
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    <torch._C.Generator object at 0x105c50b50>

    >>> a = ht.random.randn(3, 4)
    >>> a
    tensor([[-0.1955, -0.9656,  0.4224,  0.2673],
            [-0.4212, -0.5107, -1.5727, -0.1232],
            [ 3.5870, -1.8313,  1.5987, -1.2770]])

    >>> b = ht.random.randn(3, 4)
    >>> b
    tensor([[ 0.8310, -0.2477, -0.8029,  0.2366],
            [ 0.2857,  0.6898, -0.6331,  0.8795],
            [-0.6842,  0.4533,  0.2912, -0.8317]])

    >>> ht.maximum(a, b)
    tensor([[ 0.8310, -0.2477,  0.4224,  0.2673],
            [ 0.2857,  0.6898, -0.6331,  0.8795],
            [ 3.5870,  0.4533,  1.5987, -0.8317]])

    >>> c = ht.random.randn(1, 4)
    >>> c
    tensor([[-1.6428,  0.9803, -0.0421, -0.8206]])

    >>> ht.maximum(a, c)
    tensor([[-0.1955,  0.9803,  0.4224,  0.2673],
            [-0.4212,  0.9803, -0.0421, -0.1232],
            [ 3.5870,  0.9803,  1.5987, -0.8206]])

    >>> b.__setitem__((0, 1), ht.nan)
    >>> b
    tensor([[ 0.8310,     nan, -0.8029,  0.2366],
            [ 0.2857,  0.6898, -0.6331,  0.8795],
            [-0.6842,  0.4533,  0.2912, -0.8317]])
    >>> ht.maximum(a, b)
    tensor([[ 0.8310,     nan,  0.4224,  0.2673],
            [ 0.2857,  0.6898, -0.6331,  0.8795],
            [ 3.5870,  0.4533,  1.5987, -0.8317]])

    >>> d = ht.random.randn(3, 4, 5)
    >>> ht.maximum(a, d)
    ValueError: operands could not be broadcast, input shapes (3, 4) (3, 4, 5)
    """
    # perform sanitation
    if not isinstance(x1, dndarray.DNDarray) or not isinstance(x2, dndarray.DNDarray):
        raise TypeError(
            "expected x1 and x2 to be a ht.DNDarray, but were {}, {} ".format(type(x1), type(x2))
        )
    if out is not None and not isinstance(out, dndarray.DNDarray):
        raise TypeError("expected out to be None or an ht.DNDarray, but was {}".format(type(out)))

    # apply split semantics
    if x1.split is not None or x2.split is not None:
        if x1.split is None:
            x1.resplit_(x2.split)
        if x2.split is None:
            x2.resplit_(x1.split)
        if x1.split != x2.split:
            if np.prod(x1.gshape) < np.prod(x2.gshape):
                x1.resplit_(x2.split)
            if np.prod(x2.gshape) < np.prod(x1.gshape):
                x2.resplit_(x1.split)
            else:
                if x1.split < x2.split:
                    x2.resplit_(x1.split)
                else:
                    x1.resplit_(x2.split)
        split = x1.split
    else:
        split = None

    # locally: apply torch.max(x1, x2)
    output_lshape = stride_tricks.broadcast_shape(x1.lshape, x2.lshape)
    lresult = factories.empty(output_lshape, dtype=x1.dtype)
    lresult._DNDarray__array = torch.max(x1._DNDarray__array, x2._DNDarray__array)
    lresult._DNDarray__dtype = types.promote_types(x1.dtype, x2.dtype)
    lresult._DNDarray__split = split
    if x1.split is not None or x2.split is not None:
        if x1.comm.is_distributed():  # assuming x1.comm = x2.comm
            output_gshape = stride_tricks.broadcast_shape(x1.gshape, x2.gshape)
            result = factories.empty(output_gshape, dtype=x1.dtype)
            x1.comm.Allgather(lresult, result)
            # TODO: adopt Allgatherv() as soon as it is fixed, Issue #233
            result._DNDarray__dtype = lresult._DNDarray__dtype
            result._DNDarray__split = split

            if out is not None:
                if out.shape != output_gshape:
                    raise ValueError(
                        "Expecting output buffer of shape {}, got {}".format(
                            output_gshape, out.shape
                        )
                    )
                out._DNDarray__array = result._DNDarray__array
                out._DNDarray__dtype = result._DNDarray__dtype
                out._DNDarray__split = split
                out._DNDarray__device = x1.device
                out._DNDarray__comm = x1.comm

                return out
            return result

    if out is not None:
        if out.shape != output_lshape:
            raise ValueError(
                "Expecting output buffer of shape {}, got {}".format(output_lshape, out.shape)
            )
        out._DNDarray__array = lresult._DNDarray__array
        out._DNDarray__dtype = lresult._DNDarray__dtype
        out._DNDarray__split = split
        out._DNDarray__device = x1.device
        out._DNDarray__comm = x1.comm

    return lresult


def mean(x, axis=None):
    """
    Calculates and returns the mean of a tensor.
    If a axis is given, the mean will be taken in that direction.

    Parameters
    ----------
    x : ht.DNDarray
        Values for which the mean is calculated for.
        The dtype of x must be a float
    axis : None, Int, iterable, defaults to None
        Axis which the mean is taken in. Default None calculates mean of all data items.

    Returns
    -------
    means : ht.DNDarray
        The mean/s, if split, then split in the same direction as x, if possible. Fpr more
        information on the split semantics see Notes.

    Notes
    -----
    Split semantics when axis is an integer:
        if axis = x.split, then means.split = None
        if axis > split, then means.split = x.split
        if axis < split, then means.split = x.split - 1

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
        This does an element-wise update of the calculated means to merge them together using the
        merge_means function. This function operates using x from the mean function parameters.

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
            mu = torch.mean(x._DNDarray__array, dim=axis)
        else:
            mu = factories.zeros(output_shape_i, device=x.device)

        mu_shape = list(mu.shape) if list(mu.shape) else [1]

        mu_tot = factories.zeros(([x.comm.size] + mu_shape), device=x.device)
        n_tot = factories.zeros(x.comm.size, device=x.device)
        n_tot[x.comm.rank] = float(x.lshape[x.split])
        mu_tot[x.comm.rank, :] = mu
        x.comm.Allreduce(MPI.IN_PLACE, n_tot, MPI.SUM)
        x.comm.Allreduce(MPI.IN_PLACE, mu_tot, MPI.SUM)

        for i in range(1, x.comm.size):
            mu_tot[0, :], n_tot[0] = __merge_moments(
                (mu_tot[0, :], n_tot[0]), (mu_tot[i, :], n_tot[i])
            )
        return mu_tot[0][0] if mu_tot[0].size == 1 else mu_tot[0]

    # ----------------------------------------------------------------------------------------------

    if axis is None:
        # full matrix calculation
        if not x.is_distributed():
            # if x is not distributed do a torch.mean on x
            ret = torch.mean(x._DNDarray__array.float())
            return factories.array(ret, is_split=None, device=x.device)
        else:
            # if x is distributed and no axis is given: return mean of the whole set
            mu_in = torch.mean(x._DNDarray__array)
            if torch.isnan(mu_in):
                mu_in = 0.0
            n = x.lnumel
            mu_tot = factories.zeros((x.comm.size, 2), device=x.device)
            mu_proc = factories.zeros((x.comm.size, 2), device=x.device)
            mu_proc[x.comm.rank] = mu_in, float(n)
            x.comm.Allreduce(mu_proc, mu_tot, MPI.SUM)

            for i in range(1, x.comm.size):
                mu_tot[0, 0], mu_tot[0, 1] = __merge_moments(
                    (mu_tot[0, 0], mu_tot[0, 1]), (mu_tot[i, 0], mu_tot[i, 1])
                )
            return mu_tot[0][0]

    output_shape = list(x.shape)
    if isinstance(axis, (list, tuple, dndarray.DNDarray, torch.Tensor)):
        if isinstance(axis, (list, tuple)):
            if len(set(axis)) != len(axis):
                raise ValueError("duplicate value in axis")
        if isinstance(axis, (dndarray.DNDarray, torch.Tensor)):
            if axis.unique().numel() != axis.numel():
                raise ValueError("duplicate value in axis")
        if any([not isinstance(j, int) for j in axis]):
            raise ValueError(
                "items in axis iterable must be integers, axes: {}".format([type(q) for q in axis])
            )
        if any(d < 0 for d in axis):
            axis = [stride_tricks.sanitize_axis(x.shape, j) for j in axis]
        if any(d > len(x.shape) for d in axis):
            raise ValueError(
                "axes (axis) must be < {}, currently are {}".format(len(x.shape), axis)
            )

        output_shape = [output_shape[it] for it in range(len(output_shape)) if it not in axis]
        # multiple dimensions
        if x.split is None:
            return factories.array(
                torch.mean(x._DNDarray__array, dim=axis), is_split=x.split, device=x.device
            )

        if x.split in axis:
            # merge in the direction of the split
            return reduce_means_elementwise(output_shape)
        else:
            # multiple dimensions which does *not* include the split axis
            # combine along the split axis
            return factories.array(
                torch.mean(x._DNDarray__array, dim=axis),
                is_split=x.split if x.split < len(output_shape) else len(output_shape) - 1,
                device=x.device,
            )
    elif isinstance(axis, int):
        if axis >= len(x.shape):
            raise ValueError("axis (axis) must be < {}, currently is {}".format(len(x.shape), axis))
        axis = stride_tricks.sanitize_axis(x.shape, axis=axis)

        # only one axis given
        output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]
        output_shape = output_shape if output_shape else (1,)

        if x.split is None:

            return factories.array(
                torch.mean(x._DNDarray__array, dim=axis), is_split=None, device=x.device
            )
        elif axis == x.split:
            return reduce_means_elementwise(output_shape)
        else:
            # singular axis given (axis) not equal to split direction (x.split)
            return factories.array(
                torch.mean(x._DNDarray__array, dim=axis),
                is_split=x.split if axis > x.split else x.split - 1,
                device=x.device,
            )
    raise TypeError(
        "axis (axis) must be an int or a list, ht.DNDarray, "
        "torch.Tensor, or tuple, but was {}".format(type(axis))
    )


def __merge_moments(m1, m2, bessel=True):
    """
    Merge two statistical moments. If the length of m1/m2 (must be equal) is == 3 then the second moment (variance)
    is merged. This function can be expanded to merge other moments according to Reference 1 as well.
    Note: all tensors/arrays must be either the same size or individual values

    Parameters
    ----------
    m1 : tuple
        Tuple of the moments to merge together, the 0th element is the moment to be merged. The tuple must be
        sorted in descending order of moments
    m2 : tuple
        Tuple of the moments to merge together, the 0th element is the moment to be merged. The tuple must be
        sorted in descending order of moments
    bessel : bool
        Flag for the use of the bessel correction for the calculation of the variance

    Returns
    -------
    merged_moments : tuple
        a tuple of the merged moments

    References
    ----------
    [1] J. Bennett, R. Grout, P. Pebay, D. Roe, D. Thompson, Numerically stable, single-pass, parallel statistics
        algorithms, IEEE International Conference on Cluster Computing and Workshops, 2009, Oct 2009, New Orleans, LA,
        USA.
    """
    if len(m1) != len(m2):
        raise ValueError(
            "m1 and m2 must be same length, currently {} and {}".format(len(m1), len(m2))
        )
    n1, n2 = m1[-1], m2[-1]
    mu1, mu2 = m1[-2], m2[-2]
    n = n1 + n2
    delta = mu2 - mu1
    mu = mu1 + n2 * (delta / n)
    if len(m1) == 2:  # merge means
        return mu, n

    var1, var2 = m1[-3], m2[-3]
    if bessel:
        var_m = (var1 * (n1 - 1) + var2 * (n2 - 1) + (delta ** 2) * n1 * n2 / n) / (n - 1)
    else:
        var_m = (var1 * n1 + var2 * n2 + (delta ** 2) * n1 * n2 / n) / n

    if len(m1) == 3:  # merge vars
        return var_m, mu, n


def min(x, axis=None, out=None, keepdim=None):
    # TODO: initial : scalar, optional Issue #101
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    x : ht.DNDarray
        Input data.
    axis : None or int or tuple of ints
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ht.DNDarray, optional
        Tuple of two output tensors (min, min_indices). Must be of the same shape and buffer length as the expected
        output. The maximum value of an output element. Must be present to allow computation on empty slice.
    keepdim : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original arr.

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
            result = result[0]
        return result

    largest_value = constants.sanitize_infinity(x._DNDarray__array.dtype)
    return operations.__reduce_op(
        x, local_min, MPI.MIN, axis=axis, out=out, neutral=largest_value, keepdim=keepdim
    )


def minimum(x1, x2, out=None):
    """
    Compares two tensors and returns a new tensor containing the element-wise minima.
    If one of the elements being compared is a NaN, then that element is returned. TODO: Check this: If both elements are NaNs then the first is returned.
    The latter distinction is important for complex NaNs, which are defined as at least one of the real or imaginary parts being a NaN. The net effect is that NaNs are propagated.

    Parameters:
    -----------

    x1, x2 : ht.DNDarray
            The tensors containing the elements to be compared. They must have the same shape, or shapes that can be broadcast to a single shape.
            For broadcasting semantics, see: https://pytorch.org/docs/stable/notes/broadcasting.html

    out : ht.DNDarray or None, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or None, a freshly-allocated tensor is returned.

    Returns:
    --------

    minimum: ht.DNDarray
            Element-wise minimum of the two input tensors.

    Examples:
    ---------
    >>> import heat as ht
    >>> import torch
    >>> torch.manual_seed(1)
    <torch._C.Generator object at 0x105c50b50>

    >>> a = ht.random.randn(3,4)
    >>> a
    tensor([[-0.1955, -0.9656,  0.4224,  0.2673],
            [-0.4212, -0.5107, -1.5727, -0.1232],
            [ 3.5870, -1.8313,  1.5987, -1.2770]])

    >>> b = ht.random.randn(3,4)
    >>> b
    tensor([[ 0.8310, -0.2477, -0.8029,  0.2366],
            [ 0.2857,  0.6898, -0.6331,  0.8795],
            [-0.6842,  0.4533,  0.2912, -0.8317]])

    >>> ht.minimum(a,b)
    tensor([[-0.1955, -0.9656, -0.8029,  0.2366],
            [-0.4212, -0.5107, -1.5727, -0.1232],
            [-0.6842, -1.8313,  0.2912, -1.2770]])

    >>> c = ht.random.randn(1,4)
    >>> c
    tensor([[-1.6428,  0.9803, -0.0421, -0.8206]])

    >>> ht.minimum(a,c)
    tensor([[-1.6428, -0.9656, -0.0421, -0.8206],
            [-1.6428, -0.5107, -1.5727, -0.8206],
            [-1.6428, -1.8313, -0.0421, -1.2770]])

    >>> b.__setitem__((0,1), ht.nan)
    >>> b
    tensor([[ 0.8310,     nan, -0.8029,  0.2366],
            [ 0.2857,  0.6898, -0.6331,  0.8795],
            [-0.6842,  0.4533,  0.2912, -0.8317]])
    >>> ht.minimum(a,b)
    tensor([[-0.1955,     nan, -0.8029,  0.2366],
            [-0.4212, -0.5107, -1.5727, -0.1232],
            [-0.6842, -1.8313,  0.2912, -1.2770]])

    >>> d = ht.random.randn(3,4,5)
    >>> ht.minimum(a,d)
    ValueError: operands could not be broadcast, input shapes (3, 4) (3, 4, 5)
    """
    # perform sanitation
    if not isinstance(x1, dndarray.DNDarray) or not isinstance(x2, dndarray.DNDarray):
        raise TypeError(
            "expected x1 and x2 to be a ht.DNDarray, but were {}, {} ".format(type(x1), type(x2))
        )
    if out is not None and not isinstance(out, dndarray.DNDarray):
        raise TypeError("expected out to be None or an ht.DNDarray, but was {}".format(type(out)))

    # apply split semantics
    if x1.split is not None or x2.split is not None:
        if x1.split is None:
            x1.resplit_(x2.split)
        if x2.split is None:
            x2.resplit_(x1.split)
        if x1.split != x2.split:
            if np.prod(x1.gshape) < np.prod(x2.gshape):
                x1.resplit_(x2.split)
            if np.prod(x2.gshape) < np.prod(x1.gshape):
                x2.resplit_(x1.split)
            else:
                if x1.split < x2.split:
                    x2.resplit_(x1.split)
                else:
                    x1.resplit_(x2.split)
        split = x1.split
    else:
        split = None

    # locally: apply torch.min(x1, x2)
    output_lshape = stride_tricks.broadcast_shape(x1.lshape, x2.lshape)
    lresult = factories.empty(output_lshape, dtype=x1.dtype)
    lresult._DNDarray__array = torch.min(x1._DNDarray__array, x2._DNDarray__array)
    lresult._DNDarray__dtype = types.promote_types(x1.dtype, x2.dtype)
    lresult._DNDarray__split = split
    if x1.split is not None or x2.split is not None:
        if x1.comm.is_distributed():  # assuming x1.comm = x2.comm
            output_gshape = stride_tricks.broadcast_shape(x1.gshape, x2.gshape)
            result = factories.empty(output_gshape, dtype=x1.dtype)
            x1.comm.Allgather(lresult, result)
            # TODO: adopt Allgatherv() as soon as it is fixed, Issue #233
            result._DNDarray__dtype = lresult._DNDarray__dtype
            result._DNDarray__split = split

            if out is not None:
                if out.shape != output_gshape:
                    raise ValueError(
                        "Expecting output buffer of shape {}, got {}".format(
                            output_gshape, out.shape
                        )
                    )
                out._DNDarray__array = result._DNDarray__array
                out._DNDarray__dtype = result._DNDarray__dtype
                out._DNDarray__split = split
                out._DNDarray__device = x1.device
                out._DNDarray__comm = x1.comm

                return out
            return result

    if out is not None:
        if out.shape != output_lshape:
            raise ValueError(
                "Expecting output buffer of shape {}, got {}".format(output_lshape, out.shape)
            )
        out._DNDarray__array = lresult._DNDarray__array
        out._DNDarray__dtype = lresult._DNDarray__dtype
        out._DNDarray__split = split
        out._DNDarray__device = x1.device
        out._DNDarray__comm = x1.comm

    return lresult


def mpi_argmax(a, b, _):
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0]), dim=1)
    indices = torch.stack((lhs.chunk(2)[1], rhs.chunk(2)[1]), dim=1)

    # determine the minimum value and select the indices accordingly
    max, max_indices = torch.max(values, dim=1)
    result = torch.cat((max, indices[torch.arange(values.shape[0]), max_indices]))

    rhs.copy_(result)


MPI_ARGMAX = MPI.Op.Create(mpi_argmax, commute=True)


def mpi_argmin(a, b, _):
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))
    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0]), dim=1)
    indices = torch.stack((lhs.chunk(2)[1], rhs.chunk(2)[1]), dim=1)

    # determine the minimum value and select the indices accordingly
    min, min_indices = torch.min(values, dim=1)
    result = torch.cat((min, indices[torch.arange(values.shape[0]), min_indices]))

    rhs.copy_(result)


MPI_ARGMIN = MPI.Op.Create(mpi_argmin, commute=True)


def std(x, axis=None, ddof=0, **kwargs):
    """
    Calculates and returns the standard deviation of a tensor with the bessel correction.
    If a axis is given, the variance will be taken in that direction.

    Parameters
    ----------
    x : ht.DNDarray
        Values for which the std is calculated for.
        The dtype of x must be a float
    axis : None, Int, iterable, defaults to None
        Axis which the std is taken in. Default None calculates std of all data items.
    ddof : int, optional
        Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
        represents the number of elements. Default: ddof=0. If ddof=1, the Bessel correction will be applied.
        Setting ddof > 1 raises a NotImplementedError.

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
    tensor(1.2742)
    >>> a = ht.random.randn(4,4)
    >>> a
    tensor([[-1.0206,  0.3229,  1.1800,  1.5471],
            [ 0.2732, -0.0965, -0.1087, -1.3805],
            [ 0.2647,  0.5998, -0.1635, -0.0848],
            [ 0.0343,  0.1618, -0.8064, -0.1031]])
    >>> ht.std(a, 0, ddof=1)
    tensor([0.6157, 0.2918, 0.8324, 1.1996])
    >>> ht.std(a, 1, ddof=1)
    tensor([1.1405, 0.7236, 0.3506, 0.4324])
    >>> ht.std(a, 1)
    tensor([0.9877, 0.6267, 0.3037, 0.3745])
    """
    if not axis:
        return np.sqrt(var(x, axis, ddof, **kwargs))
    else:
        return exponential.sqrt(var(x, axis, ddof, **kwargs), out=None)


def var(x, axis=None, ddof=0, **kwargs):
    """
    Calculates and returns the variance of a tensor. If an axis is given, the variance will be
    taken in that direction.

    Parameters
    ----------
    x : ht.DNDarray
        Values for which the variance is calculated for.
        The dtype of x must be a float
    axis : None, Int, iterable, defaults to None
        Axis which the variance is taken in. Default None calculates variance of all data items.
    ddof : int, optional (see Notes)
        Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
        represents the number of elements. Default: ddof=0. If ddof=1, the Bessel correction will be applied.
        Setting ddof > 1 raises a NotImplementedError.

    Returns
    -------
    variances : ht.DNDarray
        The var/s, if split, then split in the same direction as x, if possible. Fpr more
        information on the split semantics see Notes.

    Notes
    -----
    Split semantics when axis is an integer:
        if axis = x.split, then variances.split = None
        if axis > split, then variances.split = x.split
        if axis < split, then variances.split = x.split - 1

    Notes on ddof (from numpy)
    --------------------------
    The variance is the average of the squared deviations from the mean, i.e., var = mean(abs(x - x.mean())**2).
    The mean is normally calculated as x.sum() / N, where N = len(x). If, however, ddof is specified, the divisor
    N - ddof is used instead. In standard statistical practice, ddof=1 provides an unbiased estimator of the
    variance of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate of the variance
    for normally distributed variables.

    Examples
    --------
    >>> a = ht.random.randn(1,3)
    >>> a
    tensor([[-1.9755,  0.3522,  0.4751]])
    >>> ht.var(a)
    tensor(1.2710)
    >>> ht.var(a, ddof=1)
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
    >>> ht.var(a, 0, ddof=1)
    tensor([1.3624, 3.2563, 0.1447, 1.2042])
    >>> ht.var(a, 0, ddof=0)
    tensor([1.0218, 2.4422, 0.1085, 0.9032])
    """

    if not isinstance(ddof, int):
        raise TypeError("ddof must be integer, is {}".format(type(ddof)))
    elif ddof > 1:
        raise NotImplementedError("Not implemented for ddof > 1.")
    elif ddof < 0:
        raise ValueError("Expected ddof=0 or ddof=1, got {}".format(ddof))
    else:
        if kwargs.get("bessel"):
            bessel = kwargs.get("bessel")
        else:
            bessel = bool(ddof)

    def reduce_vars_elementwise(output_shape_i):
        """
        Function to combine the calculated vars together. This does an element-wise update of the
        calculated vars to merge them together using the merge_vars function. This function operates
         using x from the var function parameters.

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
            mu = torch.mean(x._DNDarray__array, dim=axis)
            var = torch.var(x._DNDarray__array, dim=axis, unbiased=bessel)
        else:
            mu = factories.zeros(output_shape_i, dtype=x.dtype, device=x.device)
            var = factories.zeros(output_shape_i, dtype=x.dtype, device=x.device)

        var_shape = list(var.shape) if list(var.shape) else [1]

        var_tot = factories.zeros(([x.comm.size, 2] + var_shape), dtype=x.dtype, device=x.device)
        n_tot = factories.zeros(x.comm.size, device=x.device)
        var_tot[x.comm.rank, 0, :] = var
        var_tot[x.comm.rank, 1, :] = mu
        n_tot[x.comm.rank] = float(x.lshape[x.split])
        x.comm.Allreduce(MPI.IN_PLACE, var_tot, MPI.SUM)
        x.comm.Allreduce(MPI.IN_PLACE, n_tot, MPI.SUM)

        for i in range(1, x.comm.size):
            var_tot[0, 0, :], var_tot[0, 1, :], n_tot[0] = __merge_moments(
                (var_tot[0, 0, :], var_tot[0, 1, :], n_tot[0]),
                (var_tot[i, 0, :], var_tot[i, 1, :], n_tot[i]),
                bessel=bessel,
            )
        return var_tot[0, 0, :][0] if var_tot[0, 0, :].size == 1 else var_tot[0, 0, :]

    # ----------------------------------------------------------------------------------------------
    if axis is None:  # no axis given
        if not x.is_distributed():  # not distributed (full tensor on one node)
            ret = torch.var(x._DNDarray__array.float(), unbiased=bessel)
            return factories.array(ret)

        else:  # case for full matrix calculation (axis is None)
            mu_in = torch.mean(x._DNDarray__array)
            var_in = torch.var(x._DNDarray__array, unbiased=bessel)
            # Nan is returned when local tensor is empty
            if torch.isnan(var_in):
                var_in = 0.0
            if torch.isnan(mu_in):
                mu_in = 0.0

            n = x.lnumel
            var_tot = factories.zeros((x.comm.size, 3), dtype=x.dtype, device=x.device)
            var_proc = factories.zeros((x.comm.size, 3), dtype=x.dtype, device=x.device)
            var_proc[x.comm.rank] = var_in, mu_in, float(n)
            x.comm.Allreduce(var_proc, var_tot, MPI.SUM)

            for i in range(1, x.comm.size):
                var_tot[0, 0], var_tot[0, 1], var_tot[0, 2] = __merge_moments(
                    (var_tot[0, 0], var_tot[0, 1], var_tot[0, 2]),
                    (var_tot[i, 0], var_tot[i, 1], var_tot[i, 2]),
                    bessel=bessel,
                )
            return var_tot[0][0]

    else:  # axis is given
        # case for var in one dimension
        output_shape = list(x.shape)
        if isinstance(axis, (list, tuple, dndarray.DNDarray, torch.Tensor)):
            if isinstance(axis, (list, tuple)):
                if len(set(axis)) != len(axis):
                    raise ValueError("duplicate value in axis")
            if isinstance(axis, (dndarray.DNDarray, torch.Tensor)):
                if axis.unique().numel() != axis.numel():
                    raise ValueError("duplicate value in axis")
            if any([not isinstance(j, int) for j in axis]):
                raise ValueError(
                    "items in axis iterable must be integers, axes: {}".format(
                        [type(q) for q in axis]
                    )
                )
            if any(d < 0 for d in axis):
                axis = [stride_tricks.sanitize_axis(x.shape, j) for j in axis]
            if any(d > len(x.shape) for d in axis):
                raise ValueError(
                    "axes (axis) must be < {}, currently are {}".format(len(x.shape), axis)
                )

            output_shape = [output_shape[it] for it in range(len(output_shape)) if it not in axis]
            # multiple dimensions
            if x.split is None:
                return factories.array(
                    torch.var(x._DNDarray__array, dim=axis), is_split=x.split, device=x.device
                )
            if x.split in axis:
                # merge in the direction of the split
                return reduce_vars_elementwise(output_shape)
            else:
                # multiple dimensions which does *not* include the split axis
                # combine along the split axis
                return factories.array(
                    torch.var(x._DNDarray__array, dim=axis),
                    is_split=x.split if x.split < len(output_shape) else len(output_shape) - 1,
                    device=x.device,
                )
        elif isinstance(axis, int):
            if axis >= len(x.shape):
                raise ValueError("axis must be < {}, currently is {}".format(len(x.shape), axis))
            axis = stride_tricks.sanitize_axis(x.shape, axis)
            # only one axis given
            output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]
            output_shape = output_shape if output_shape else (1,)

            if x.split is None:  # x is *not* distributed -> no need to distributed
                return factories.array(
                    torch.var(x._DNDarray__array, dim=axis, unbiased=bessel),
                    dtype=x.dtype,
                    device=x.device,
                )
            elif axis == x.split:  # x is distributed and axis chosen is == to split
                return reduce_vars_elementwise(output_shape)
            else:
                # singular axis given (axis) not equal to split direction (x.split)
                lcl = torch.var(x._DNDarray__array, dim=axis, keepdim=False)
                return factories.array(
                    lcl,
                    is_split=x.split if axis > x.split else x.split - 1,
                    dtype=x.dtype,
                    device=x.device,
                )
        else:
            raise TypeError("axis (axis) must be an int, tuple, list, etc.; currently it is {}. ")
