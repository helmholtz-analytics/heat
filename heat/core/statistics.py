"""
Distributed statistical operations.
"""

import numpy as np
import torch
from typing import Any, Callable, Union, Tuple, List, Optional

from .communication import MPI
from . import arithmetics
from . import exponential
from . import factories
from . import linalg
from . import manipulations
from . import _operations
from .dndarray import DNDarray
from . import types
from . import sanitation
from . import stride_tricks
from . import logical
from . import constants
from .random import randint
from warnings import warn

__all__ = [
    "argmax",
    "argmin",
    "average",
    "bincount",
    "bucketize",
    "cov",
    "digitize",
    "histc",
    "histogram",
    "kurtosis",
    "max",
    "maximum",
    "mean",
    "median",
    "min",
    "minimum",
    "percentile",
    "skew",
    "std",
    "var",
]


def argmax(
    x: DNDarray, axis: Optional[int] = None, out: Optional[DNDarray] = None, **kwargs: object
) -> DNDarray:
    """
    Returns an array of the indices of the maximum values along an axis. It has the same shape as ``x.shape`` with the
    dimension along axis removed.

    Parameters
    ----------
    x : DNDarray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.
    out : DNDarray, optional.
        If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
    **kwargs
        Extra keyword arguments

    Examples
    --------
    >>> a = ht.random.randn(3, 3)
    >>> a
    DNDarray([[ 1.0661,  0.7036, -2.0908],
              [-0.7534, -0.4986, -0.7751],
              [-0.4815,  1.9436,  0.6400]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.argmax(a)
    DNDarray([7], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.argmax(a, axis=0)
    DNDarray([0, 2, 2], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.argmax(a, axis=1)
    DNDarray([0, 1, 1], dtype=ht.int64, device=cpu:0, split=None)
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

        if maxima.is_mps:
            # MPS framework doesn't support float64
            out = torch.cat([maxima.float(), indices.float()])
        else:
            out = torch.cat([maxima.double(), indices.double()])

        return out

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError(f"axis must be None or int, was {type(axis)}")

    # perform the global reduction
    smallest_value = -sanitation.sanitize_infinity(x)
    return _operations.__reduce_op(
        x, local_argmax, MPI_ARGMAX, axis=axis, out=out, neutral=smallest_value, **kwargs
    )


DNDarray.argmax: Callable[[DNDarray, int, DNDarray, object], DNDarray] = (
    lambda self, axis=None, out=None, **kwargs: argmax(self, axis, out, **kwargs)
)
DNDarray.argmax.__doc__ = argmax.__doc__


def argmin(
    x: DNDarray, axis: Optional[int] = None, out: Optional[DNDarray] = None, **kwargs: object
) -> DNDarray:
    """
    Returns an array of the indices of the minimum values along an axis. It has the same shape as ``x.shape`` with the
    dimension along axis removed.

    Parameters
    ----------
    x : DNDarray
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise along the specified axis.
    out : DNDarray, optional
        Issue #100 If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
    **kwargs
        Extra keyword arguments

    Examples
    --------
    >>> a = ht.random.randn(3, 3)
    >>> a
    DNDarray([[ 1.0661,  0.7036, -2.0908],
              [-0.7534, -0.4986, -0.7751],
              [-0.4815,  1.9436,  0.6400]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.argmin(a)
    DNDarray([2], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.argmin(a, axis=0)
    DNDarray([1, 1, 0], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.argmin(a, axis=1)
    DNDarray([2, 2, 0], dtype=ht.int64, device=cpu:0, split=None)
    """

    def local_argmin(*args, **kwargs):
        axis = kwargs.get("dim", -1)
        shape = x.shape

        # case where the argmin axis is set to None
        # argmin will be the flattened index, computed standalone and the actual minimum value obtain separately
        if len(args) <= 1 and axis < 0:
            indices = torch.argmin(*args, **kwargs).reshape(1)
            minima = args[0].flatten()[indices]

            # artificially flatten the input tensor shape to correct the offset computation
            axis = 0
            shape = [np.prod(shape)]
        # usual case where indices and minimum values are both returned. Axis is not equal to None
        else:
            minima, indices = torch.min(*args, **kwargs)

        # add offset of data chunks if reduction is computed across split axis
        if axis == x.split:
            offset, _, _ = x.comm.chunk(shape, x.split)
            indices += torch.tensor(offset, dtype=indices.dtype)

        if minima.is_mps:
            # MPS framework doesn't support float64
            out = torch.cat([minima.float(), indices.float()])
        else:
            out = torch.cat([minima.double(), indices.double()])

        return out

    # axis sanitation
    if axis is not None and not isinstance(axis, int):
        raise TypeError(f"axis must be None or int, was {type(axis)}")

    # perform the global reduction
    largest_value = sanitation.sanitize_infinity(x)
    return _operations.__reduce_op(
        x, local_argmin, MPI_ARGMIN, axis=axis, out=out, neutral=largest_value, **kwargs
    )


DNDarray.argmin: Callable[[DNDarray, int, DNDarray, object], DNDarray] = (
    lambda self, axis=None, out=None, **kwargs: argmin(self, axis, out, **kwargs)
)
DNDarray.argmin.__doc__ = argmin.__doc__


def average(
    x: DNDarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    weights: Optional[DNDarray] = None,
    returned: bool = False,
) -> Union[DNDarray, Tuple[DNDarray, ...]]:
    """
    Compute the weighted average along the specified axis.

    If ``returned=True``, return a tuple with the average as the first element and the sum
    of the weights as the second element. ``sum_of_weights`` is of the same type as ``average``.

    Parameters
    ----------
    x : DNDarray
        Array containing data to be averaged.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which to average ``x``.  The default,
        ``axis=None``, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
        #TODO Issue #351: If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : DNDarray, optional
        An array of weights associated with the values in ``x``. Each value in
        ``x`` contributes to the average according to its associated weight.
        The weights array can either be 1D (in which case its length must be
        the size of ``x`` along the given axis) or of the same shape as ``x``.
        If ``weights=None``, then all data in ``x`` are assumed to have a
        weight equal to one, the result is equivalent to :func:`mean`.
    returned : bool, optional
        If ``True``, the tuple ``(average, sum_of_weights)``
        is returned, otherwise only the average is returned.
        If ``weights=None``, ``sum_of_weights`` is equivalent to the number of
        elements over which the average is taken.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero.
    TypeError
        When the length of 1D weights is not the same as the shape of ``x``
        along axis.

    Examples
    --------
    >>> data = ht.arange(1, 5, dtype=float)
    >>> data
    DNDarray([1., 2., 3., 4.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.average(data)
    DNDarray(2.5000, dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.average(ht.arange(1, 11, dtype=float), weights=ht.arange(10, 0, -1))
    DNDarray([4.], dtype=ht.float64, device=cpu:0, split=None)
    >>> data = ht.array([[0, 1],
                         [2, 3],
                        [4, 5]], dtype=float, split=1)
    >>> weights = ht.array([1.0 / 4, 3.0 / 4])
    >>> ht.average(data, axis=1, weights=weights)
    DNDarray([0.7500, 2.7500, 4.7500], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.average(data, weights=weights)
    Traceback (most recent call last):
        ...
    TypeError: Axis must be specified when shapes of x and weights differ.
    """
    # perform sanitation
    if not isinstance(x, DNDarray):
        raise TypeError(f"expected x to be a ht.DNDarray, but was {type(x)}")
    if weights is not None and not isinstance(weights, DNDarray):
        raise TypeError(f"expected weights to be a ht.DNDarray, but was {type(x)}")
    axis = stride_tricks.sanitize_axis(x.shape, axis)

    if weights is None:
        result = mean(x, axis)
        num_elements = x.gnumel / result.gnumel
        cumwgt = factories.empty(1, dtype=result.dtype)
        cumwgt.larray = torch.tensor(num_elements)
    else:
        # Weights sanitation:
        # weights (global) is either same size as x (global), or it is 1D and same size as x along chosen axis
        if x.gshape != weights.gshape:
            if axis is None:
                raise TypeError("Axis must be specified when shapes of x and weights differ.")
            elif isinstance(axis, tuple):
                raise NotImplementedError("Weighted average over tuple axis not implemented yet.")
            if weights.ndim != 1:
                raise TypeError("1D weights expected when shapes of x and weights differ.")
            if weights.gshape[0] != x.gshape[axis]:
                raise ValueError("Length of weights not compatible with specified axis.")

            wgt_lshape = tuple(
                weights.lshape[0] if dim == axis else 1 for dim in list(range(x.ndim))
            )
            wgt_slice = [slice(None) if dim == axis else 0 for dim in list(range(x.ndim))]
            wgt_split = None if weights.split is None else axis
            wgt = torch.empty(
                wgt_lshape, dtype=weights.dtype.torch_type(), device=x.device.torch_device
            )
            wgt[wgt_slice] = weights.larray
            wgt = factories.array(wgt, is_split=wgt_split, copy=False)
        else:
            if x.comm.is_distributed():
                if x.split is not None and weights.split != x.split and weights.ndim != 1:
                    # fix after Issue #425 is solved
                    raise NotImplementedError(
                        "weights.split does not match data.split: not implemented yet."
                    )
            wgt = factories.empty_like(weights, device=x.device)
            wgt.larray = weights.larray
        cumwgt = wgt.sum(axis=axis)
        if logical.any(cumwgt == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        result = (x * wgt).sum(axis=axis) / cumwgt

    if returned:
        if cumwgt.gshape != result.gshape:
            cumwgt = factories.array(
                torch.broadcast_tensors(cumwgt.larray, result.larray)[0],
                is_split=result.split,
                device=result.device,
                comm=result.comm,
                copy=False,
            )
        return (result, cumwgt)

    return result


DNDarray.average: Callable[
    [DNDarray, Union[int, Tuple[int, ...]], DNDarray, bool], Union[DNDarray, Tuple[DNDarray, ...]]
] = lambda self, axis=None, weights=None, returned=False: average(self, axis, weights, returned)
DNDarray.average.__doc__ = average.__doc__


def bincount(x: DNDarray, weights: Optional[DNDarray] = None, minlength: int = 0) -> DNDarray:
    """
    Count number of occurrences of each value in array of non-negative ints. Return a
    non-distributed ``DNDarray`` of length `max(x) + 1` if input is non-empty, else 0.

    The number of bins (size 1) is one larger than the largest value in `x`
    unless `x` is empty, in which case the result is a tensor of size 0.
    If `minlength` is specified, the number of bins is at least `minlength` and
    if `x` is empty, then the result is tensor of size `minlength` filled with zeros.
    If `n` is the value at position `i`, `out[n] += weights[i]` if weights is specified else `out[n] += 1`.

    Parameters
    ----------
    x : DNDarray
        1-dimensional, non-negative ints
    weights : DNDarray, optional
        Weight for each value in the input tensor. Array of the same shape as x. Same split as `x`.
    minlength : int, non-negative, optional
        Minimum number of bins

    Raises
    ------
    ValueError
        If `x` and `weights` don't have the same distribution.

    Examples
    --------
    >>> ht.bincount(ht.arange(5))
    DNDarray([1, 1, 1, 1, 1], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bincount(ht.array([0, 1, 3, 2, 1]), weights=ht.array([0, 0.5, 1, 1.5, 2]))
    DNDarray([0.0000, 2.5000, 1.5000, 1.0000], dtype=ht.float32, device=cpu:0, split=None)
    """
    if isinstance(weights, DNDarray):
        if weights.split != x.split:
            raise ValueError("weights must have the same split value as x")
        weights = weights.larray

    counts = torch.bincount(x.larray, weights, minlength)

    size = counts.numel()
    maxlength = x.comm.allreduce(size, op=MPI.MAX)

    # resize tensors
    if size == 0:
        dtype = torch.int64
        if weights is not None:
            dtype = torch.float64
        counts = torch.zeros(maxlength, dtype=dtype, device=counts.device)
    elif size < maxlength:
        counts = torch.cat(
            (counts, torch.zeros(maxlength - size, dtype=counts.dtype, device=counts.device))
        )

    # collect results
    if x.split == 0:
        data = torch.empty_like(counts)
        x.comm.Allreduce(counts, data, op=MPI.SUM)
    else:
        data = counts

    return DNDarray(
        data,
        gshape=tuple(data.shape),
        dtype=types.heat_type_of(data),
        split=None,
        device=x.device,
        comm=x.comm,
        balanced=True,
    )


def bucketize(
    input: DNDarray,
    boundaries: Union[DNDarray, torch.Tensor],
    out_int32: bool = False,
    right: bool = False,
    out: DNDarray = None,
) -> DNDarray:
    """
    Returns the indices of the buckets to which each value in the input belongs, where the boundaries of the buckets are set by boundaries.

    Parameters
    ----------
    input : DNDarray
        The input array.
    boundaries : DNDarray or torch.Tensor
        monotonically increasing sequence defining the bucket boundaries, 1-dimensional, not distributed
    out_int32 : bool, optional
        set the dtype of the output to ``ht.int64`` (`False`) or ``ht.int32`` (True)
    right : bool, optional
        indicate whether the buckets include the right (`False`) or left (`True`) boundaries, see Notes.
    out : DNDarray, optional
        The output array, must be the shame shape and split as the input array.

    Notes
    -----
    This function uses the PyTorch's setting for ``right``:

    ===== ====================================
    right returned index `i` satisfies
    ===== ====================================
    False boundaries[i-1] < x <= boundaries[i]
    True  boundaries[i-1] <= x < boundaries[i]
    ===== ====================================

    Raises
    ------
    RuntimeError
        If `boundaries` is distributed.

    See Also
    --------
    digitize
        NumPy-like version of this function.

    Examples
    --------
    >>> boundaries = ht.array([1, 3, 5, 7, 9])
    >>> v = ht.array([[3, 6, 9], [3, 6, 9]])
    >>> ht.bucketize(v, boundaries)
    DNDarray([[1, 3, 4],
              [1, 3, 4]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.bucketize(v, boundaries, right=True)
    DNDarray([[2, 3, 5],
              [2, 3, 5]], dtype=ht.int64, device=cpu:0, split=None)
    """
    if isinstance(boundaries, DNDarray):
        if boundaries.is_distributed():
            raise RuntimeError("'boundaries' must not be distributed.")
        boundaries = boundaries.larray
    else:
        boundaries = torch.as_tensor(boundaries)

    return _operations.__local_op(
        torch.bucketize,
        input,
        out,
        no_cast=True,
        boundaries=boundaries,
        out_int32=out_int32,
        right=right,
    )


def cov(
    m: DNDarray,
    y: Optional[DNDarray] = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
) -> DNDarray:
    """
    Estimate the covariance matrix of some data, m. For more imformation on the algorithm please see the numpy function of the same name

    Parameters
    ----------
    m : DNDarray
        A 1-D or 2-D array containing multiple variables and observations. Each row of ``m`` represents a variable, and each column a single
        observation of all those variables.
    y : DNDarray, optional
        An additional set of variables and observations. ``y`` has the same form as that of ``m``.
    rowvar : bool, optional
        If ``True`` (default), then each row represents a variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows contain observations.
    bias : bool, optional
        Default normalization (``False``) is by (N - 1), where N is the number of observations given (unbiased estimate).
        If ``True``, then normalization is by N. These values can be overridden by using the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by ``bias`` is overridden. Note that ``ddof=1`` will return the unbiased estimate and
        ``ddof=0`` will return the simple average.
    """
    if ddof is not None and not isinstance(ddof, int):
        raise TypeError("ddof must be integer")
    if not isinstance(m, DNDarray):
        raise TypeError("m must be a DNDarray")
    if not m.is_balanced():
        raise RuntimeError("balance is required for cov(). use balance_() to balance m")
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if m.ndim == 1:
        m = m.expand_dims(1)
    x = m.copy()
    if not rowvar and x.shape[0] != 1:
        x = x.T

    if ddof is None:
        if bias == 0:
            ddof = 1.0
        else:
            ddof = 0.0
    elif isinstance(ddof, int):
        ddof = float(ddof)

    if y is not None:
        if not isinstance(y, DNDarray):
            raise TypeError("y must be a DNDarray")
        if y.ndim > 2:
            raise ValueError("y has too many dimensions, max=2")
        if y.ndim == 1:
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
        raise ValueError(f"ddof >= number of elements in m, {ddof} {m.gnumel}")
    x -= avg.expand_dims(1)
    c = linalg.dot(x, x.T)
    c /= norm
    return c


def digitize(x: DNDarray, bins: Union[DNDarray, torch.Tensor], right: bool = False) -> DNDarray:
    """
    Return the indices of the bins to which each value in the input array `x` belongs.
    If values in `x` are beyond the bounds of bins, 0 or len(bins) is returned as appropriate.

    Parameters
    ----------
    x : DNDarray
        The input array
    bins : DNDarray or torch.Tensor
        A 1-dimensional array containing a monotonic sequence describing the bin boundaries, not distributed.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin edge, see Notes.

    Notes
    -----
    This function uses NumPy's setting for ``right``:

    ===== ============= ============================
    right order of bins returned index `i` satisfies
    ===== ============= ============================
    False increasing    bins[i-1] <= x < bins[i]
    True  increasing    bins[i-1] < x <= bins[i]
    False decreasing    bins[i-1] > x >= bins[i]
    True  decreasing    bins[i-1] >= x > bins[i]
    ===== ============= ============================

    Raises
    ------
    RuntimeError
        If `bins` is distributed.

    See Also
    --------
    bucketize
        PyTorch-like version of this function.

    Examples
    --------
    >>> x = ht.array([1.2, 10.0, 12.4, 15.5, 20.0])
    >>> bins = ht.array([0, 5, 10, 15, 20])
    >>> ht.digitize(x, bins, right=True)
    DNDarray([1, 2, 3, 4, 4], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.digitize(x, bins, right=False)
    DNDarray([1, 3, 3, 4, 5], dtype=ht.int64, device=cpu:0, split=None)
    """
    if isinstance(bins, DNDarray):
        if bins.is_distributed():
            raise RuntimeError("'bins' must not be distributed.")
        bins = bins.larray
    else:
        bins = torch.as_tensor(bins)

    reverse = False

    if bins[0] > bins[-1]:
        bins = torch.flipud(bins)
        reverse = True

    result = _operations.__local_op(
        torch.bucketize,
        x,
        out=None,
        no_cast=True,
        boundaries=bins,
        out_int32=False,
        right=not right,
    )

    if reverse:
        result = bins.numel() - result

    return result


def histc(
    input: DNDarray, bins: int = 100, min: int = 0, max: int = 0, out: Optional[DNDarray] = None
) -> DNDarray:
    """
    Return the histogram of a DNDarray.

    The elements are sorted into equal width bins between min and max.
    If min and max are both equal, the minimum and maximum values of the data are used.
    Elements lower than min and higher than max are ignored.

    Parameters
    ----------
    input : DNDarray
            the input array, must be of float type
    bins  : int, optional
            number of histogram bins
    min   : int, optional
            lower end of the range (inclusive)
    max   : int, optional
            upper end of the range (inclusive)
    out   : DNDarray, optional
            the output tensor, same dtype as input

    Examples
    --------
    >>> ht.histc(ht.array([1.0, 2, 1]), bins=4, min=0, max=3)
    DNDarray([0., 2., 1., 0.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.histc(ht.arange(10, dtype=ht.float64, split=0), bins=10)
    DNDarray([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=ht.float64, device=cpu:0, split=None)
    """
    if min == max:
        min = float(input.min())
        max = float(input.max())

    hist = torch.histc(
        input._DNDarray__array,
        bins,
        min,
        max,
        out=out._DNDarray__array if out is not None and input.split is None else None,
    )

    if not input.is_distributed():
        if out is None:
            out = DNDarray(
                hist,
                gshape=tuple(hist.shape),
                dtype=types.canonical_heat_type(hist.dtype),
                split=None,
                device=input.device,
                comm=input.comm,
                balanced=True,
            )
    else:
        if out is None:
            out = factories.empty(
                hist.size(), dtype=types.canonical_heat_type(hist.dtype), device=input.device
            )
        input.comm.Allreduce(hist, out, op=MPI.SUM)

    return out


def histogram(
    a: DNDarray,
    bins: int = 10,
    range: Tuple[int, int] = (0, 0),
    normed: Optional[bool] = None,
    weights: Optional[DNDarray] = None,
    density: Optional[bool] = None,
) -> DNDarray:
    """
    Compute the histogram of a DNDarray.

    Parameters
    ----------
    a       : DNDarray
              the input array, must be of float type
    bins    : int, optional
              number of histogram bins
    range   : Tuple[int,int], optional
              lower and upper end of the bins. If not provided, range is simply (a.min(), a.max()).
    normed  : bool, optional
              Deprecated since NumPy version 1.6. TODO: remove.
    weights : DNDarray, optional
              array of weights. Not implemented yet.
    density : bool, optional
              Not implemented yet.

    Notes
    -----
    This is a wrapper function of :func:`histc` for some basic compatibility with the NumPy API.

    See Also
    --------
    :func:`histc`
    """
    # TODO: Rewrite to make it a proper implementation of the NumPy function

    if normed is not None:
        raise NotImplementedError("'normed' is not supported")
    if weights is not None:
        raise NotImplementedError("'weights' is not supported")
    if density is not None:
        raise NotImplementedError("'density' is not supported")
    if not isinstance(bins, int):
        raise NotImplementedError("'bins' only supports integer values")

    return histc(a, bins, range[0], range[1])


def kurtosis(
    x: DNDarray, axis: Optional[int] = None, unbiased: bool = True, Fischer: bool = True
) -> DNDarray:
    """
    Compute the kurtosis (Fisher or Pearson) of a dataset.
    Kurtosis is the fourth central moment divided by the square of the variance.
    If Fisher’s definition is used, then 3.0 is subtracted from the result to give 0.0 for a normal distribution.

    If unbiased is True (defualt) then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Parameters
    ----------
    x : ht.DNDarray
        Input array
    axis : NoneType or Int
        Axis along which skewness is calculated, Default is to compute over the whole array `x`
    unbiased : Bool
        if True (default) the calculations are corrected for bias
    Fischer : bool
        Whether use Fischer's definition or not. If true 3. is subtracted from the result.

    Warnings
    --------
    UserWarning: Dependent on the axis given and the split configuration, a UserWarning may be thrown during this function as data is transferred between processes.
    """
    if axis is None or (isinstance(axis, int) and x.split == axis):  # no axis given
        # TODO: determine if this is a valid (and fast implementation)
        mu = mean(x, axis=axis)
        if axis is not None and axis > 0:
            mu = mu.expand_dims(axis)
        diff = x - mu
        n = float(x.shape[axis]) if axis is not None else x.gnumel

        m4 = arithmetics.sum(arithmetics.pow(diff, 4.0), axis) / n
        m2 = arithmetics.sum(arithmetics.pow(diff, 2.0), axis) / n
        res = m4 / arithmetics.pow(m2, 2.0)
        if unbiased:
            res = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * res - 3 * (n - 1.0)) + 3.0
        if Fischer:
            res -= 3.0
        return res.item() if res.gnumel == 1 else res
    elif isinstance(axis, (list, tuple)):
        raise TypeError(f"axis cannot be a list or a tuple, currently {type(axis)}")
    else:
        return __moment_w_axis(__torch_kurtosis, x, axis, None, unbiased, Fischer)


DNDarray.kurtosis: Callable[[DNDarray, int, bool, bool], DNDarray] = (
    lambda x, axis=None, unbiased=True, Fischer=True: kurtosis(x, axis, unbiased, Fischer)
)
DNDarray.kurtosis.__doc__ = average.__doc__


def max(
    x: DNDarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    out: Optional[DNDarray] = None,
    keepdims: Optional[bool] = None,
) -> DNDarray:
    # TODO: initial : scalar, optional Issue #101
    """
    Return the maximum along a given axis.

    Parameters
    ----------
    x : DNDarray
        Input array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : DNDarray, optional
        Tuple of two output arrays ``(max, max_indices)``. Must be of the same shape and buffer length as the expected
        output. The minimum value of an output element. Must be present to allow computation on empty slice.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
    --------
    >>> a = ht.float32([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    >>> ht.max(a)
    DNDarray([12.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.max(a, axis=0)
    DNDarray([10., 11., 12.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.max(a, axis=1)
    DNDarray([ 3.,  6.,  9., 12.], dtype=ht.float32, device=cpu:0, split=None)
    """

    def local_max(*args, **kwargs):
        result = torch.max(*args, **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        return result

    if isinstance(x, (Tuple, List)):
        return torch.tensor(x).max().item()

    smallest_value = -sanitation.sanitize_infinity(x)
    return _operations.__reduce_op(
        x, local_max, MPI.MAX, axis=axis, out=out, neutral=smallest_value, keepdims=keepdims
    )


DNDarray.max: Callable[[DNDarray, Union[int, Tuple[int, ...]], DNDarray, bool], DNDarray] = (
    lambda x, axis=None, out=None, keepdims=None: max(x, axis, out, keepdims)
)
DNDarray.max.__doc__ = max.__doc__


def maximum(x1: DNDarray, x2: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compares two ``DNDarrays`` and returns a new :class:`~heat.core.dndarray.DNDarray` containing the element-wise maxima.
    The ``DNDarrays`` must have the same shape, or shapes that can be broadcast to a single shape.
    For broadcasting semantics, see: https://pytorch.org/docs/stable/notes/broadcasting.html
    If one of the elements being compared is ``NaN``, then that element is returned.
    TODO: Check this: If both elements are NaNs then the first is returned.
    The latter distinction is important for complex NaNs, which are defined as at least one of the real or
    imaginary parts being ``NaN``. The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1 : DNDarray
            The first array containing the elements to be compared.
    x2 : DNDarray
            The second array containing the elements to be compared.
    out : DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or ``None``, a freshly-allocated array is returned.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.random.randn(3, 4)
    >>> a
    DNDarray([[ 0.2701, -0.6993,  1.2197,  0.0579],
              [ 0.6815,  0.4722, -0.3947, -0.3030],
              [ 1.0101, -1.2460, -1.3953, -0.6879]], dtype=ht.float32, device=cpu:0, split=None)
    >>> b = ht.random.randn(3, 4)
    >>> b
    DNDarray([[ 0.9664,  0.6159, -0.8555,  0.8204],
              [-1.2200, -0.0759,  0.0437,  0.4700],
              [ 1.2271,  1.0530,  0.1095,  0.8386]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.maximum(a, b)
    DNDarray([[0.9664, 0.6159, 1.2197, 0.8204],
              [0.6815, 0.4722, 0.0437, 0.4700],
              [1.2271, 1.0530, 0.1095, 0.8386]], dtype=ht.float32, device=cpu:0, split=None)
    >>> c = ht.random.randn(1, 4)
    >>> c
    DNDarray([[-0.5363, -0.9765,  0.4099,  0.3520]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.maximum(a, c)
    DNDarray([[ 0.2701, -0.6993,  1.2197,  0.3520],
              [ 0.6815,  0.4722,  0.4099,  0.3520],
              [ 1.0101, -0.9765,  0.4099,  0.3520]], dtype=ht.float32, device=cpu:0, split=None)
    >>> d = ht.random.randn(3, 4, 5)
    >>> ht.maximum(a, d)
    ValueError: operands could not be broadcast, input shapes (3, 4) (3, 4, 5)
    """
    return _operations.__binary_op(torch.max, x1, x2, out)


def mean(x: DNDarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> DNDarray:
    """
    Calculates and returns the mean of a ``DNDarray``.
    If an axis is given, the mean will be taken in that direction.

    Parameters
    ----------
    x : DNDarray
        Values for which the mean is calculated for.
        The dtype of ``x`` must be a float
    axis : None or int or iterable
        Axis which the mean is taken in. Default ``None`` calculates mean of all data items.

    Notes
    -----
    Split semantics when axis is an integer:

    - if ``axis==x.split``, then ``mean(x).split=None``

    - if ``axis>split``, then ``mean(x).split=x.split``

    - if ``axis<split``, then ``mean(x).split=x.split-1``

    Examples
    --------
    >>> a = ht.random.randn(1, 3)
    >>> a
    DNDarray([[-0.1164,  1.0446, -0.4093]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.mean(a)
    DNDarray(0.1730, dtype=ht.float32, device=cpu:0, split=None)
    >>> a = ht.random.randn(4, 4)
    >>> a
    DNDarray([[-1.0585,  0.7541, -1.1011,  0.5009],
              [-1.3575,  0.3344,  0.4506,  0.7379],
              [-0.4337, -0.6516, -1.3690, -0.8772],
              [ 0.6929, -1.0989, -0.9961,  0.3547]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.mean(a, 1)
    DNDarray([-0.2262,  0.0413, -0.8328, -0.2619], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.mean(a, 0)
    DNDarray([-0.5392, -0.1655, -0.7539,  0.1791], dtype=ht.float32, device=cpu:0, split=None)
    >>> a = ht.random.randn(4, 4)
    >>> a
    DNDarray([[-0.1441,  0.5016,  0.8907,  0.6318],
              [-1.1690, -1.2657,  1.4840, -0.1014],
              [ 0.4133,  1.4168,  1.3499,  1.0340],
              [-0.9236, -0.7535, -0.2466, -0.9703]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.mean(a, (0, 1))
    DNDarray(0.1342, dtype=ht.float32, device=cpu:0, split=None)
    """

    def reduce_means_elementwise(output_shape_i: torch.Tensor) -> DNDarray:
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
            mu = torch.mean(x.larray, dim=axis)
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
    # sanitize dtype
    if types.heat_type_is_exact(x.dtype):
        if x.dtype is types.int64 and not x.larray.is_mps:
            x = x.astype(types.float64)
        else:
            x = x.astype(types.float32)

    if axis is None:
        # full matrix calculation
        if not x.is_distributed():
            # if x is not distributed do a torch.mean on x
            ret = torch.mean(x.larray)
            return DNDarray(
                ret,
                gshape=tuple(ret.shape),
                dtype=types.heat_type_of(ret),
                split=None,
                device=x.device,
                comm=x.comm,
                balanced=True,
            )
        else:
            # if x is distributed and no axis is given: return mean of the whole set
            mu_in = torch.mean(x.larray)
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
    return __moment_w_axis(torch.mean, x, axis, reduce_means_elementwise)


DNDarray.mean: Callable[[DNDarray, Union[int, List, Tuple]], DNDarray] = lambda x, axis=None: mean(
    x, axis
)
DNDarray.mean.__doc__ = mean.__doc__


def median(
    x: DNDarray,
    axis: Optional[int] = None,
    keepdims: bool = False,
    sketched: bool = False,
    sketch_size: Optional[float] = 1.0 / MPI.COMM_WORLD.size,
) -> DNDarray:
    """
    Compute the median of the data along the specified axis.
    Returns the median of the ``DNDarray`` elements.
    Per default, the "true" median of the entire data set is computed; however, the argument
    `sketched` allows to switch to a faster but less accurate version that computes
    the median only on behalf of a random subset of the data set ("sketch").

    Parameters
    ----------
    x : DNDarray
        Input tensor
    axis : int, or None, optional
        Axis along which the median is computed. Default is ``None``, i.e.,
        the median is computed along a flattened version of the ``DNDarray``.

    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result can broadcast correctly against the original array ``a``.

    sketched : bool, optional
        If True, the median is computed on a random subset of the data set ("sketch").
        This is faster but less accurate.  Default is False. The size of the sketch is controlled by the argument `sketch_size`.
    sketch_size : float, optional
        The size of the sketch as a fraction of the data set size. Default is `1./n_proc`  where `n_proc` is the number of MPI processes, e.g. `n_proc =  MPI.COMM_WORLD.size`. Must be in the range (0, 1).
        Ignored for sketched = False.
    """
    return percentile(
        x, q=50, axis=axis, keepdims=keepdims, sketched=sketched, sketch_size=sketch_size
    )


DNDarray.median: Callable[[DNDarray, int, bool, bool, float], DNDarray] = (
    lambda x,
    axis=None,
    keepdims=False,
    sketched=False,
    sketch_size=1.0 / MPI.COMM_WORLD.size: median(
        x, axis, keepdims, sketched=sketched, sketch_size=sketch_size
    )
)
DNDarray.median.__doc__ = median.__doc__


def __merge_moments(
    m1: torch.Tensor, m2: torch.Tensor, unbiased: bool = True
) -> Tuple[torch.Tensor, ...]:
    """
    Merge two statistical moments.
    If the length of ``m1`` and ``m2`` (must be equal) is ``==3`` then the second moment (variance)
    is merged. This function can be expanded to merge other moments according to Reference [1] as well.
    Note: all arrays must be either the same size or individual values

    Parameters
    ----------
    m1 : Tuple
        Tuple of the moments to merge together, the 0th element is the moment to be merged. The tuple must be
        sorted in descending order of moments
    m2 : Tuple
        Tuple of the moments to merge together, the 0th element is the moment to be merged. The tuple must be
        sorted in descending order of moments
    unbiased : bool
        Flag for the use of unbiased estimators (when available)

    References
    ----------
    [1] J. Bennett, R. Grout, P. Pebay, D. Roe, D. Thompson, Numerically stable, single-pass, parallel statistics
        algorithms, IEEE International Conference on Cluster Computing and Workshops, 2009, Oct 2009, New Orleans, LA,
        USA.
    """
    if len(m1) != len(m2):
        raise ValueError(f"m1 and m2 must be same length, currently {len(m1)} and {len(m2)}")
    n1, n2 = m1[-1], m2[-1]
    mu1, mu2 = m1[-2], m2[-2]
    n = n1 + n2
    delta = mu2 - mu1
    mu = mu1 + n2 * (delta / n)
    if len(m1) == 2:  # merge means
        return mu, n

    var1, var2 = m1[-3], m2[-3]
    if unbiased:
        var_m = (var1 * (n1 - 1) + var2 * (n2 - 1) + (delta**2) * n1 * n2 / n) / (n - 1)
    else:
        var_m = (var1 * n1 + var2 * n2 + (delta**2) * n1 * n2 / n) / n

    if len(m1) == 3:  # merge vars
        return var_m, mu, n

    # TODO: This code block can be added if skew or kurtosis support multiple axes:
    # sk1, sk2 = m1[-4], m2[-4]
    # dn = delta / n
    # if all(var_m != 0):  # Skewness does not exist if var is 0
    #     s1 = sk1 + sk2
    #     s2 = dn * (n1 * var2 - n2 * var1) / 6.0
    #     s3 = (dn ** 3) * n1 * n2 * (n1 ** 2 - n2 ** 2)
    #     skew_m = s1 + s2 + s3
    # else:
    #     skew_m = None
    # if len(m1) == 4:
    #     return skew_m, var_m, mu, n
    #
    # k1, k2 = m1[-5], m2[-5]
    # if skew_m is None:
    #     return None, skew_m, var_m, mu, n
    # s1 = (1 / 24.0) * dn * (n1 * sk2 - n2 * sk1)
    # s2 = (1 / 12.0) * (dn ** 2) * ((n1 ** 2) * var2 + (n2 ** 2) * var1)
    # s3 = (dn ** 4) * n1 * n2 * ((n1 ** 3) + (n2 ** 3))
    # k = k1 + k2 + s1 + s2 + s3
    # if len(m1) == 5:
    #     return k, skew_m, var_m, mu, n


def min(
    x: DNDarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    out: Optional[DNDarray] = None,
    keepdims: Optional[bool] = None,
) -> DNDarray:
    # TODO: initial : scalar, optional Issue #101
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    x : DNDarray
        Input array.
    axis : None or int or Tuple[int,...]
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : Tuple[DNDarray,DNDarray], optional
        Tuple of two output arrays ``(min, min_indices)``. Must be of the same shape and buffer length as the expected
        output. The maximum value of an output element. Must be present to allow computation on empty slice.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.


    Examples
    --------
    >>> a = ht.float32([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
    >>> ht.min(a)
    DNDarray([1.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.min(a, axis=0)
    DNDarray([1., 2., 3.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.min(a, axis=1)
    DNDarray([ 1.,  4.,  7., 10.], dtype=ht.float32, device=cpu:0, split=None)
    """

    def local_min(*args, **kwargs):
        result = torch.min(*args, **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        return result

    if isinstance(x, (Tuple, List)):
        return torch.tensor(x).min().item()

    largest_value = sanitation.sanitize_infinity(x)
    return _operations.__reduce_op(
        x, local_min, MPI.MIN, axis=axis, out=out, neutral=largest_value, keepdims=keepdims
    )


DNDarray.min: Callable[[DNDarray, Union[int, Tuple[int, ...]], DNDarray, bool], DNDarray] = (
    lambda self, axis=None, out=None, keepdims=None: min(self, axis, out, keepdims)
)
DNDarray.min.__doc__ = min.__doc__


def minimum(x1: DNDarray, x2: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Compares two ``DNDarrays`` and returns a new :class:`~heat.core.dndarray.DNDarray`  containing the element-wise minima.
    If one of the elements being compared is ``NaN``, then that element is returned. They must have the same shape,
    or shapes that can be broadcast to a single shape. For broadcasting semantics,
    see: https://pytorch.org/docs/stable/notes/broadcasting.html
    TODO: Check this: If both elements are NaNs then the first is returned.
    The latter distinction is important for complex NaNs, which are defined as at least one of the real or
    imaginary parts being ``NaN``. The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1 : DNDarray
        The first array containing the elements to be compared.
    x2 : DNDarray
        The second array containing the elements to be compared.
    out : DNDarray, optional
        A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
        If not provided or ``None``, a freshly-allocated array is returned.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.random.randn(3, 4)
    >>> a
    DNDarray([[-0.5462,  0.0079,  1.2828,  1.4980],
              [ 0.6503, -1.1069,  1.2131,  1.4003],
              [-0.3203, -0.2318,  1.0388,  0.4439]], dtype=ht.float32, device=cpu:0, split=None)
    >>> b = ht.random.randn(3, 4)
    >>> b
    DNDarray([[ 1.8505,  2.3055, -0.2825, -1.4718],
              [-0.3684,  1.6866, -0.8570, -0.4779],
              [ 1.0532,  0.3775, -0.8669, -1.7275]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.minimum(a, b)
    DNDarray([[-0.5462,  0.0079, -0.2825, -1.4718],
              [-0.3684, -1.1069, -0.8570, -0.4779],
              [-0.3203, -0.2318, -0.8669, -1.7275]], dtype=ht.float32, device=cpu:0, split=None)
    >>> c = ht.random.randn(1, 4)
    >>> c
    DNDarray([[-1.4358,  1.2914, -0.6042, -1.4009]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.minimum(a, c)
    DNDarray([[-1.4358,  0.0079, -0.6042, -1.4009],
              [-1.4358, -1.1069, -0.6042, -1.4009],
              [-1.4358, -0.2318, -0.6042, -1.4009]], dtype=ht.float32, device=cpu:0, split=None)
    >>> d = ht.random.randn(3, 4, 5)
    >>> ht.minimum(a, d)
    ValueError: operands could not be broadcast, input shapes (3, 4) (3, 4, 5)
    """
    return _operations.__binary_op(torch.min, x1, x2, out)


def __moment_w_axis(
    function: Callable,
    x: DNDarray,
    axis: Optional[Union[int, Tuple[int, ...]]],
    elementwise_function: Callable,
    unbiased: Optional[bool] = None,
    Fischer: Optional[bool] = None,
) -> DNDarray:
    """
    Helper function for calculating a statistical moment along a given axis.

    Parameters
    ----------
    function : Callable
        local torch moment function
    x : DNDarray
        target dataset
    axis : Union[None, int, list, tuple]
        axis/axes to calculate the moment
    elementwise_function : Callable
        function to merge the moment across processes
    unbiased : bool
        if the moment should be unbiased
    Fischer : bool
        if the Fischer correction is to be applied (only used in skew and Kurtosis)
    """
    # helper for calculating a statistical moment with a given axis
    kwargs = {"dim": axis}
    if unbiased:
        kwargs["unbiased"] = unbiased
    if Fischer:
        kwargs["Fischer"] = Fischer

    output_shape = list(x.shape)
    if isinstance(axis, int):
        if axis >= len(x.shape):
            raise ValueError(f"axis must be < {len(x.shape)}, currently is {axis}")
        axis = stride_tricks.sanitize_axis(x.shape, axis)
        # only one axis given
        output_shape = [output_shape[it] for it in range(len(output_shape)) if it != axis]
        output_shape = output_shape if output_shape else (1,)

        if x.split is None:  # x is *not* distributed -> no need to distribute
            ret = function(x.larray, **kwargs)
            return DNDarray(
                ret,
                gshape=tuple(ret.shape),
                dtype=x.dtype,
                split=None,
                device=x.device,
                comm=x.comm,
                balanced=x.balanced,
            )
        elif axis == x.split:  # x is distributed and axis chosen is == to split
            return elementwise_function(output_shape)
        # singular axis given (axis) not equal to split direction (x.split)
        lcl = function(x.larray, **kwargs)
        return factories.array(
            lcl,
            is_split=x.split if axis > x.split else x.split - 1,
            dtype=x.dtype,
            device=x.device,
            comm=x.comm,
            copy=False,
        )
    elif not isinstance(axis, (list, tuple, torch.Tensor)):
        raise TypeError(
            f"axis must be an int, tuple, list, or torch.Tensor; currently it is {type(axis)}."
        )
    # else:
    if isinstance(axis, torch.Tensor):
        axis = axis.tolist()

    if isinstance(axis, (list, tuple)) and len(set(axis)) != len(axis):  # most common case
        raise ValueError("duplicate value in axis")
    if any(not isinstance(j, int) for j in axis):
        raise ValueError(
            f"items in axis iterable must be integers, axes: {[type(q) for q in axis]}"
        )

    if any(d < 0 for d in axis):
        axis = [stride_tricks.sanitize_axis(x.shape, j) for j in axis]
    if any(d > len(x.shape) for d in axis):
        raise ValueError(f"axes (axis) must be < {len(x.shape)}, currently are {axis}")

    output_shape = [output_shape[it] for it in range(len(output_shape)) if it not in axis]
    # multiple dimensions
    if x.split is None:
        ret = function(x.larray, **kwargs)
        return DNDarray(
            ret,
            gshape=tuple(ret.shape),
            dtype=types.heat_type_of(ret),
            split=None,
            device=x.device,
            comm=x.comm,
            balanced=True,
        )
    if x.split in axis:
        # merge in the direction of the split
        return elementwise_function(output_shape)
    # multiple dimensions which does *not* include the split axis
    # combine along the split axis
    return factories.array(
        function(x.larray, **kwargs),
        is_split=x.split if x.split < len(output_shape) else len(output_shape) - 1,
        device=x.device,
        comm=x.comm,
        copy=False,
    )


def mpi_argmax(a: str, b: str, _: Any) -> torch.Tensor:
    """
    Create the MPI function for doing argmax, for more info see :func:`argmax <argmax>`

    Parameters
    ----------
    a : str
        left hand side buffer
    b : str
        right hand side buffer
    _ : Any
        placeholder
    """
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))

    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    idx_l, idx_r = lhs.chunk(2)[1], rhs.chunk(2)[1]

    if idx_l[0] < idx_r[0]:
        values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0]), dim=1)
        indices = torch.stack((idx_l, idx_r), dim=1)
    else:
        values = torch.stack((rhs.chunk(2)[0], lhs.chunk(2)[0]), dim=1)
        indices = torch.stack((idx_r, idx_l), dim=1)

    # determine the minimum value and select the indices accordingly
    max, max_indices = torch.max(values, dim=1)
    result = torch.cat((max, indices[torch.arange(values.shape[0]), max_indices]))

    rhs.copy_(result)


MPI_ARGMAX = MPI.Op.Create(mpi_argmax, commute=True)


def mpi_argmin(a: str, b: str, _: Any) -> torch.Tensor:
    """
    Create the MPI function for doing argmin, for more info see :func:`argmin <argmin>`

    Parameters
    ----------
    a : str
        left hand side
    b : str
        right hand side
    _ : Any
        placeholder
    """
    lhs = torch.from_numpy(np.frombuffer(a, dtype=np.float64))
    rhs = torch.from_numpy(np.frombuffer(b, dtype=np.float64))
    # extract the values and minimal indices from the buffers (first half are values, second are indices)
    idx_l, idx_r = lhs.chunk(2)[1], rhs.chunk(2)[1]

    if idx_l[0] < idx_r[0]:
        values = torch.stack((lhs.chunk(2)[0], rhs.chunk(2)[0]), dim=1)
        indices = torch.stack((idx_l, idx_r), dim=1)
    else:
        values = torch.stack((rhs.chunk(2)[0], lhs.chunk(2)[0]), dim=1)
        indices = torch.stack((idx_r, idx_l), dim=1)

    # determine the minimum value and select the indices accordingly
    min, min_indices = torch.min(values, dim=1)
    result = torch.cat((min, indices[torch.arange(values.shape[0]), min_indices]))

    rhs.copy_(result)


MPI_ARGMIN = MPI.Op.Create(mpi_argmin, commute=True)


def percentile(
    x: DNDarray,
    q: Union[DNDarray, int, float, Tuple, List],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    out: Optional[DNDarray] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    sketched: bool = False,
    sketch_size: Optional[float] = 1.0 / MPI.COMM_WORLD.size,
) -> DNDarray:
    r"""
    Compute the q-th percentile of the data along the specified axis/axes.
    Returns the q-th percentile(s) of the ``DNDarray`` elements.
    Per default, the "true" percentile(s) of the entire data set are computed; however, the argument
    `sketched` allows to switch to a faster but inaccurate version that computes
    the percentile only on behalf of a random subset of the data set ("sketch").

    Parameters
    ----------
    x : DNDarray
        Input tensor
    q : DNDarray, scalar, or list of scalars
        Percentile or sequence of percentiles to compute. Must belong to the interval [0, 100].
    axis : int, tuple of ints, or None, optional
        Axis (if int) or axes (if tuple) along which the percentiles are computed. Default is None, corresponds to calculating the percentile over the flattened array.
    out : DNDarray, optional.
        Output buffer.
    interpolation : str, optional
        Interpolation method to use when the desired percentile lies between two data points :math:`i < j`.
        Can be one of:

        - ‘linear’: :math:`i + (j - i) \cdot fraction`, where `fraction` is the fractional part of the index surrounded by `i` and `j`.

        - ‘lower’: `i`.

        - ‘higher’: `j`.

        - ‘nearest’: `i` or `j`, whichever is nearest.

        - ‘midpoint’: :math:`(i + j) / 2`.

    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result can broadcast correctly against the original array x.
    sketched : bool, optional
        If False (default), the entire data is used and no sketching is performed.
        If True, a fraction of the data to use for estimating the percentile. The fraction is determined by `sketch_size`.
    sketch_size : float, optional
        The fraction of the data to use for estimating the percentile; needs to be strictly between 0 and 1.
        The default is `1/nprocs`, where `nprocs` is the number of MPI processes involved in the calculation, i.e., roughly the portion of the data that is anyway processed on a single process.
        Ignored for sketched = False.
    """

    def _create_sketch(
        a: DNDarray,
        axis: Union[int, None],
        sketch_size_relative: Optional[float] = None,
        sketch_size_absolute: Optional[int] = None,
    ) -> DNDarray:
        """
        Create a sketch of a DNDarray along a specified axis. The sketch is created by sampling the DNDarray along the specified axis.

        Parameters
        ----------
        a : DNDarray
            The DNDarray for which to create a sketch.
        axis : int
            The axis along which to create the sketch.
        sketch_size_relative : optional, float
            The size of the sketch. Fraction of samples to take, hence between 0 and 1.
        sketch_size_absolute : optional, int
            The size of the sketch. Number of samples to take, hence must not exceed the size of the axis along which the sketch is taken.
        """
        if (sketch_size_relative is None and sketch_size_absolute is None) or (
            sketch_size_relative is not None and sketch_size_absolute is not None
        ):
            raise ValueError(
                "Exactly one of sketch_size_relative and sketch_size_absolute must be specified."
            )
        if sketch_size_absolute is None:
            sketch_size = int(sketch_size_relative * a.shape[axis])
        else:
            sketch_size = sketch_size_absolute

        # create a random sample of indices
        indices = manipulations.sort(
            randint(0, a.shape[axis], sketch_size, device=a.device, dtype=types.int64)
        )[0]
        sketch = a.swapaxes(0, axis)
        sketch = a[indices, ...].resplit_(None)
        return sketch.swapaxes(0, axis)

    # sanitize input data
    sanitation.sanitize_in(x)
    if x.dtype in types._complexfloating:
        raise TypeError("Percentile is not supported for complex data types.")

    # sanitize q, keep track of size of percentile dim
    if np.isscalar(q):
        q = torch.tensor(q, device=x.device.torch_device)
    else:
        try:
            q = torch.tensor(q, device=x.device.torch_device)
        except (TypeError, IndexError):
            if isinstance(q, DNDarray):
                # q must be local for now. TODO: support distributed q after indexing update
                q.resplit_(axis=None)
                q = q.larray
            else:
                raise TypeError(f"q can be scalar, list, tuple, or DNDarray, was {type(q)}.")
        except ValueError:
            raise TypeError(f"q must be a scalar, list, tuple, or DNDarray, was {q} .")
    perc_size = tuple(q.shape)

    # sanitize axis
    axis = stride_tricks.sanitize_axis(x.shape, axis)
    original_axis = axis

    # sanitize output buffer: set output_shape, output_split, and output_dtype
    if axis is not None:
        # calculate output_shape
        if isinstance(axis, int):
            axis = (axis,)
        # axis is tuple: multiple axes
        if keepdims:
            output_shape = tuple(x.shape[ax] if ax not in axis else 1 for ax in range(x.ndim))
        else:
            # loop over non-reduced axes
            output_shape = tuple(x.shape[ax] for ax in range(x.ndim) if ax not in axis)
        # calculate output_split
        if x.split is not None and not sketched:
            split_bookkeeping = [None] * x.ndim
            split_bookkeeping[x.split] = "split"
            if not keepdims:
                # remove reduced axes
                split_bookkeeping = [
                    split_bookkeeping[ax] for ax in range(x.ndim) if ax not in tuple(axis)
                ]
            # insert percentile dimension at axis 0 if needed
            split_bookkeeping = [None] * len(perc_size) + split_bookkeeping
            # identify output split
            output_split = (
                split_bookkeeping.index("split") if "split" in split_bookkeeping else None
            )
        else:
            output_split = None
    else:
        # axis is None
        if keepdims:
            output_shape = (1,) * x.ndim
        else:
            output_shape = ()
        output_split = None
    if len(perc_size) > 0:
        output_shape = perc_size + output_shape

    # output data type must be float
    if x.larray.element_size() == 4 or x.larray.is_mps:
        output_dtype = types.float32
    else:
        output_dtype = types.float64
    if out is not None:
        sanitation.sanitize_out(out, output_shape, output_split, x.device, x.comm)
        if output_dtype != out.dtype:
            raise TypeError(f"Expected output buffer of dtype {output_dtype}, got {out.dtype}.")

    # prepare data to index/calculate percentiles along first dimension
    axis = original_axis
    original_dims = x.ndim
    if axis is None:
        # percentile along flattened data
        x = x.flatten()
        axis = 0
        if keepdims:
            x = manipulations.expand_dims(x, axis=tuple(range(1, original_dims + 1)))
    elif isinstance(axis, tuple):
        # percentile along multiple axes
        # transpose x so that the axes along which the percentiles are calculated are at the beginning
        non_op_dims = list(range(x.ndim))
        for ax in axis:
            non_op_dims.remove(ax)
        transpose_axes = (*axis, *non_op_dims)
        x = x.transpose(transpose_axes)
        # flatten the data along the axes along which the percentiles are calculated
        non_op_shape = tuple(x.shape[dim] for dim in non_op_dims)
        # calculate new split axis
        if x.is_distributed():
            if x.split in axis:
                reshaped_split = min(axis)
            # x.split < min(axis) does not occur bc of earlier transpose
            elif x.split > max(axis):
                reshaped_split = x.split - (len(axis) - 1)
        else:
            reshaped_split = None
        x = x.reshape(-1, *non_op_shape, new_split=reshaped_split)
        axis = 0
        if keepdims:
            x = manipulations.expand_dims(x, axis=tuple(og_ax + 1 for og_ax in original_axis))
    else:
        # percentile along a single axis
        # transpose x so that the axis along which the percentiles are calculated is the first axis
        transpose_axes = (axis,) + tuple(range(axis)) + tuple(range(axis + 1, x.ndim))
        x = x.transpose(transpose_axes)
        axis = 0
        if keepdims:
            x = manipulations.expand_dims(x, axis=original_axis + 1)

    if sketched:
        if (
            not isinstance(sketch_size, float)
            or sketch_size <= 0
            or (MPI.COMM_WORLD.size > 1 and sketch_size == 1)
            or sketch_size > 1
        ):
            raise ValueError(
                f"If sketched=True, sketch_size must be float strictly between 0 and 1, but is {sketch_size}."
            )
        else:
            x = _create_sketch(x, axis, sketch_size_relative=sketch_size)

    # compute indices
    length = x.shape[axis]
    perc_indices = q / 100 * (length - 1)
    if interpolation == "linear":
        # leave fractional indices, interpolate linearly
        pass
    elif interpolation == "lower":
        perc_indices = perc_indices.floor().type(torch.int)
    elif interpolation == "higher":
        perc_indices = perc_indices.ceil().type(torch.int)
    elif interpolation == "midpoint":
        perc_indices = 0.5 * (perc_indices.floor() + perc_indices.ceil())
    elif interpolation == "nearest":
        perc_indices = perc_indices.round().type(torch.int)
    else:
        raise ValueError(
            "Invalid interpolation method. Interpolation can be 'lower', 'higher', 'midpoint', 'nearest', or 'linear'."
        )

    # sort data
    sorted_x, _ = manipulations.sort(x, axis=axis)
    del _
    sorted_x = sorted_x.astype(output_dtype)

    # calculate percentiles
    if perc_indices.dtype.is_floating_point:
        # interpolate linearly
        floors = sorted_x[perc_indices.floor().type(torch.int)]
        ceils = sorted_x[perc_indices.ceil().type(torch.int)]
        del sorted_x
        if output_split is None:
            # gather results
            floors.resplit_(None)
            ceils.resplit_(None)
        else:
            ceils.redistribute_(target_map=floors.lshape_map)
        fractional_indices = perc_indices - perc_indices.floor()
        while fractional_indices.ndim < floors.ndim:
            # expand fractional indices for later binary op
            # fractional_indices is still a torch tensor here
            fractional_indices.unsqueeze_(-1)
        if out is not None:
            out.larray = floors.larray + (ceils.larray - floors.larray) * (fractional_indices)
            del floors, ceils, fractional_indices
            return out
        fractional_indices = factories.array(fractional_indices, device=x.device, comm=x.comm)
        percentile = floors + (ceils - floors) * fractional_indices
        del floors, ceils, fractional_indices
    else:
        if out is not None:
            out.larray = sorted_x[perc_indices].larray
            del sorted_x
            return out
        percentile = sorted_x[perc_indices]
        del sorted_x

    return percentile


def skew(x: DNDarray, axis: int = None, unbiased: bool = True) -> DNDarray:
    """
    Compute the sample skewness of a data set.

    Parameters
    ----------
    x : ht.DNDarray
        Input array
    axis : NoneType or Int
        Axis along which skewness is calculated, Default is to compute over the whole array `x`
    unbiased : Bool
        if True (default) the calculations are corrected for bias

    Warnings
    --------
    UserWarning: Dependent on the axis given and the split configuration, a UserWarning may be thrown during this function as data is transferred between processes.
    """
    if axis is None or (isinstance(axis, int) and x.split == axis):  # no axis given
        # TODO: determine if this is a valid (and fast implementation)
        mu = mean(x, axis=axis)
        if axis is not None and axis > 0:
            mu = mu.expand_dims(axis)
        diff = x - mu

        n = float(x.shape[axis]) if axis is not None else x.gnumel

        m3 = arithmetics.sum(arithmetics.pow(diff, 3.0), axis) / n
        m2 = arithmetics.sum(arithmetics.pow(diff, 2.0), axis) / n
        res = m3 / arithmetics.pow(m2, 1.5)
        if unbiased:
            res *= ((n * (n - 1.0)) ** 0.5) / (n - 2.0)
        return res.item() if res.gnumel == 1 else res
    elif isinstance(axis, (list, tuple)):
        raise TypeError(f"axis cannot be a list or a tuple, currently {type(axis)}")
    else:
        # if multiple axes are required, need to add a reduce_skews_elementwise function
        return __moment_w_axis(__torch_skew, x, axis, None, unbiased)


DNDarray.skew: Callable[[DNDarray, int, bool], DNDarray] = (
    lambda self, axis=None, unbiased=True: skew(self, axis, unbiased)
)
DNDarray.skew.__doc__ = skew.__doc__


def std(
    x: DNDarray, axis: Union[int, Tuple[int], List[int]] = None, ddof: int = 0, **kwargs: object
) -> DNDarray:
    """
    Calculates the standard deviation of a ``DNDarray`` with the bessel correction.
    If an axis is given, the variance will be taken in that direction.

    Parameters
    ----------
    x : DNDarray
        array for which the std is calculated for.
        The datatype of ``x`` must be a float
    axis : None or int or iterable
        Axis which the std is taken in. Default ``None`` calculates std of all data items.
    ddof : int, optional
        Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
        represents the number of elements. If ``ddof=1``, the Bessel correction will be applied.
        Setting ``ddof>1`` raises a ``NotImplementedError``.
    **kwargs
        Extra keyword arguments

    Examples
    --------
    >>> a = ht.random.randn(1, 3)
    >>> a
    DNDarray([[ 0.5714,  0.0048, -0.2942]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.std(a)
    DNDarray(0.3590, dtype=ht.float32, device=cpu:0, split=None)
    >>> a = ht.random.randn(4, 4)
    >>> a
    DNDarray([[ 0.8488,  1.2225,  1.2498, -1.4592],
              [-0.5820, -0.3928,  0.1509, -0.0174],
              [ 0.6426, -1.8149,  0.1369,  0.0042],
              [-0.6043, -0.0523, -1.6653,  0.6631]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.std(a, 1, ddof=1)
    DNDarray([1.2961, 0.3362, 1.0739, 0.9820], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.std(a, 1)
    DNDarray([1.2961, 0.3362, 1.0739, 0.9820], dtype=ht.float32, device=cpu:0, split=None)
    """
    # sanitize dtype
    if types.heat_type_is_exact(x.dtype):
        if x.dtype is types.int64 and not x.larray.is_mps:
            x = x.astype(types.float64)
        else:
            x = x.astype(types.float32)

    if not isinstance(ddof, int):
        raise TypeError(f"ddof must be integer, is {type(ddof)}")
    # elif ddof > 1:
    #     raise NotImplementedError("Not implemented for ddof > 1.")
    elif ddof < 0:
        raise ValueError(f"Expected ddof >= 0, got {ddof}")
    else:
        if kwargs.get("bessel"):
            unbiased = kwargs.get("bessel")
        else:
            unbiased = bool(ddof)
        ddof = 1 if unbiased else ddof
    if not x.is_distributed() and str(x.device).startswith("cpu"):
        loc = np.std(x.larray.numpy(), axis=axis, ddof=ddof)
        if loc.size == 1:
            return loc.item()
        return factories.array(loc, copy=False)
    return exponential.sqrt(var(x, axis, ddof, **kwargs), out=None)


DNDarray.std: Callable[[DNDarray, Union[int, Tuple[int], List[int]], int, object], DNDarray] = (
    lambda self, axis=None, ddof=0, **kwargs: std(self, axis, ddof, **kwargs)
)
DNDarray.std.__doc__ = std.__doc__


def __torch_skew(
    torch_tensor: torch.Tensor, dim: int = None, unbiased: bool = False
) -> torch.Tensor:
    """
    Calculate the sample skewness of a torch tensor
    return the bias corrected Fischer-Pearson standardized moment coefficient by default

    Parameters
    ----------
    torch_tensor : torch.Tensor
        target data
    dim : int
        dimension along which to calculate the skew
        If None, then the skew of the full dataset is calculated
    unbiased : bool
        return the unbiased estimator
    """
    if dim is not None:
        n = torch_tensor.shape[dim]
        diff = torch_tensor - torch.mean(torch_tensor, dim=dim, keepdim=True)
        m3 = torch.true_divide(torch.sum(torch.pow(diff, 3), dim=dim), n)
        m2 = torch.true_divide(torch.sum(torch.pow(diff, 2), dim=dim), n)
    else:
        n = torch_tensor.gnumel()
        diff = torch_tensor - torch.mean(torch_tensor)
        m3 = torch.true_divide(torch.sum(torch.pow(diff, 3)), n)
        m2 = torch.true_divide(torch.sum(torch.pow(diff, 2)), n)
    if not unbiased:
        return torch.true_divide(m3, torch.pow(m2, 1.5))
    coeff = ((n * (n - 1)) ** 0.5) / (n - 2.0)
    return coeff * torch.true_divide(m3, torch.pow(m2, 1.5))


def __torch_kurtosis(
    torch_tensor: torch.Tensor, dim: int = None, Fischer: bool = True, unbiased: bool = False
) -> torch.Tensor:
    """
    Calculate the sample kurtosis of a dataset
    default is unbiased and with the Fischer correction

    Parameters
    ----------
    torch_tensor : torch.Tensor
        target dataset
    dim : int, optional
        dimension along which to do calculate
        If None, then the kurtosis of the full dataset is calculated
    Fischer : bool
        if to apply the Fischer correction
    unbiased : bool
        if the returned value/s should be biased or unbiased
    """
    if dim is not None:
        n = torch_tensor.shape[dim]
        diff = torch_tensor - torch.mean(torch_tensor, dim=dim, keepdim=True)
        m4 = torch.true_divide(torch.sum(torch.pow(diff, 4.0), dim=dim), n)
        m2 = torch.true_divide(torch.sum(torch.pow(diff, 2.0), dim=dim), n)
    else:
        n = torch_tensor.gnumel()
        diff = torch_tensor - torch.mean(torch_tensor)
        m4 = torch.true_divide(torch.pow(diff, 4.0), n)
        m2 = torch.true_divide(torch.pow(diff, 2.0), n)
    res = torch.true_divide(m4, torch.pow(m2, 2.0))
    if unbiased:
        res = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * res - 3.0 * (n - 1.0)) + 3.0
    if Fischer:
        res -= 3.0
    return res


def var(
    x: DNDarray, axis: Union[int, Tuple[int], List[int]] = None, ddof: int = 0, **kwargs: object
) -> DNDarray:
    """
    Calculates and returns the variance of a ``DNDarray``. If an axis is given, the variance will be
    taken in that direction.

    Parameters
    ----------
    x : DNDarray
        Array for which the variance is calculated for.
        The datatype of ``x`` must be a float
    axis : None or int or iterable
        Axis which the std is taken in. Default ``None`` calculates std of all data items.
    ddof : int, optional
        Delta Degrees of Freedom: the denominator implicitely used in the calculation is N - ddof, where N
        represents the number of elements. If ``ddof=1``, the Bessel correction will be applied.
        Setting ``ddof>1`` raises a ``NotImplementedError``.
    **kwargs
        Extra keyword arguments


    Notes
    -----
    Split semantics when axis is an integer:

    - if ``axis=x.split``, then ``var(x).split=None``

    - if ``axis>split``, then ``var(x).split = x.split``

    - if ``axis<split``, then ``var(x).split=x.split - 1``

    The variance is the average of the squared deviations from the mean, i.e., ``var=mean(abs(x - x.mean())**2)``.
    The mean is normally calculated as ``x.sum()/N``, where ``N = len(x)``. If, however, ``ddof`` is specified, the divisor
    ``N - ddof`` is used instead. In standard statistical practice, ``ddof=1`` provides an unbiased estimator of the
    variance of a hypothetical infinite population. ``ddof=0`` provides a maximum likelihood estimate of the variance
    for normally distributed variables.

    Examples
    --------
    >>> a = ht.random.randn(1, 3)
    >>> a
    DNDarray([[-2.3589, -0.2073,  0.8806]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.var(a)
    DNDarray(1.8119, dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.var(a, ddof=1)
    DNDarray(2.7179, dtype=ht.float32, device=cpu:0, split=None)
    >>> a = ht.random.randn(4, 4)
    >>> a
    DNDarray([[-0.8523, -1.4982, -0.5848, -0.2554],
              [ 0.8458, -0.3125, -0.2430,  1.9016],
              [-0.6778, -0.3584, -1.5112,  0.6545],
              [-0.9161,  0.0168,  0.0462,  0.5964]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.var(a, 1)
    DNDarray([0.2777, 1.0957, 0.8015, 0.3936], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.var(a, 0)
    DNDarray([0.7001, 0.4376, 0.4576, 0.7890], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.var(a, 0, ddof=1)
    DNDarray([0.7001, 0.4376, 0.4576, 0.7890], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.var(a, 0, ddof=0)
    DNDarray([0.7001, 0.4376, 0.4576, 0.7890], dtype=ht.float32, device=cpu:0, split=None)
    """
    if not isinstance(ddof, int):
        raise TypeError(f"ddof must be integer, is {type(ddof)}")
    elif ddof > 1:
        raise NotImplementedError("Not implemented for ddof > 1.")
    elif ddof < 0:
        raise ValueError(f"Expected ddof=0 or ddof=1, got {ddof}")
    else:
        if kwargs.get("bessel"):
            unbiased = kwargs.get("bessel")
        else:
            unbiased = bool(ddof)

    def reduce_vars_elementwise(output_shape_i: torch.Tensor) -> DNDarray:
        """
        Function to combine the calculated vars together. This does an element-wise update of the
        calculated vars to merge them together using the merge_vars function. This function operates
        using x from the var function parameters.

        Parameters
        ----------
        output_shape_i : iterable
            Iterable with the dimensions of the output of the var function.
        """
        if x.lshape[x.split] != 0:
            mu = torch.mean(x.larray, dim=axis)
            var = torch.var(x.larray, dim=axis, unbiased=unbiased)
        else:
            mu = factories.zeros(output_shape_i, dtype=x.dtype, device=x.device)
            var = factories.zeros(output_shape_i, dtype=x.dtype, device=x.device)

        var_shape = list(var.shape) if list(var.shape) else [1]

        var_tot = factories.zeros(([x.comm.size, 3] + var_shape), dtype=x.dtype, device=x.device)
        var_tot[x.comm.rank, 0, :] = var
        var_tot[x.comm.rank, 1, :] = mu
        var_tot[x.comm.rank, 2, :] = float(x.lshape[x.split])
        x.comm.Allreduce(MPI.IN_PLACE, var_tot, MPI.SUM)

        for i in range(1, x.comm.size):
            var_tot[0, 0, :], var_tot[0, 1, :], var_tot[0, 2, :] = __merge_moments(
                (var_tot[0, 0, :], var_tot[0, 1, :], var_tot[0, 2, :]),
                (var_tot[i, 0, :], var_tot[i, 1, :], var_tot[i, 2, :]),
                unbiased=unbiased,
            )
        return var_tot[0, 0, :][0] if var_tot[0, 0, :].size == 1 else var_tot[0, 0, :]

    # ----------------------------------------------------------------------------------------------
    if axis is None:  # no axis given
        if not x.is_distributed():  # not distributed (full tensor on one node)
            ret = torch.var(x.larray.float(), unbiased=unbiased)
            return DNDarray(
                ret, tuple(ret.shape), types.heat_type_of(ret), None, x.device, x.comm, True
            )

        else:  # case for full matrix calculation (axis is None)
            mu_in = torch.mean(x.larray)
            var_in = torch.var(x.larray, unbiased=unbiased)
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
                    unbiased=unbiased,
                )
            return var_tot[0][0]

    else:  # axis is given
        return __moment_w_axis(torch.var, x, axis, reduce_vars_elementwise, unbiased)


DNDarray.var: Callable[[DNDarray, Union[int, Tuple[int], List[int]], int, object], DNDarray] = (
    lambda self, axis=None, ddof=0, **kwargs: var(self, axis, ddof, **kwargs)
)
DNDarray.var.__doc__ = var.__doc__
