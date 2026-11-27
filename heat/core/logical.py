"""
Logical functions for the DNDarrays
"""

import numpy as np
import torch

from typing import Callable, Optional, Sequence, Tuple, Union

from . import factories
from . import manipulations
from . import sanitation

from . import _operations
from . import stride_tricks
from . import types
from ._indexing_utils import _prepare_alltoallv_routing

from .communication import MPI
from .dndarray import DNDarray

__all__ = [
    "all",
    "allclose",
    "any",
    "in1d",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "signbit",
]


def all(
    x: DNDarray,
    axis: Union[int, Tuple[int], None] = None,
    out: Optional[DNDarray] = None,
    keepdims: bool = False,
) -> Union[DNDarray, bool]:
    """
    Test whether all array elements along a given axis evaluate to ``True``.
    A new boolean or :class:`~heat.core.dndarray.DNDarray` is returned unless out is specified, in which case a
    reference to ``out`` is returned.

    Parameters
    ----------
    x : DNDarray
        Input array or object that can be converted to an array.
    axis : None or int or Tuple[int,...], optional
        Axis or axes along which a logical AND reduction is performed. The default (``axis=None``) is to perform a
        logical AND over all the dimensions of the input array. ``axis`` may be negative, in which case it counts
        from the last to the first axis.
    out : DNDarray, optional
        Alternate output array in which to place the result. It must have the same shape as the expected output
        and its type is preserved.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
    --------
    >>> x = ht.random.randn(4, 5)
    >>> x
    DNDarray([[ 0.7199,  1.3718,  1.5008,  0.3435,  1.2884],
              [ 0.1532, -0.0968,  0.3739,  1.7843,  0.5614],
              [ 1.1522,  1.9076,  1.7638,  0.4110, -0.2803],
              [-0.5475, -0.0271,  0.8564, -1.5870,  1.3108]], dtype=ht.float32, device=cpu:0, split=None)
    >>> y = x < 0.5
    >>> y
    DNDarray([[False, False, False,  True, False],
              [ True,  True,  True, False, False],
              [False, False, False,  True,  True],
              [ True,  True, False,  True, False]], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(y)
    DNDarray([False], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(y, axis=0)
    DNDarray([False, False, False, False, False], dtype=ht.bool, device=cpu:0, split=None)
    >>> ht.all(x, axis=1)
    DNDarray([True, True, True, True], dtype=ht.bool, device=cpu:0, split=None)
    >>> out = ht.zeros(5)
    >>> ht.all(y, axis=0, out=out)
    DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)
    >>> out
    DNDarray([False, False, False, False, False], dtype=ht.float32, device=cpu:0, split=None)
    """

    def local_all(t, *args, **kwargs):
        return torch.all(t != 0, *args, **kwargs)

    if keepdims and axis is None:
        axis = tuple(range(x.ndim))

    return _operations.__reduce_op(
        x, local_all, MPI.LAND, axis=axis, out=out, neutral=1, keepdims=keepdims
    )


DNDarray.all: Callable[
    [Union[int, Tuple[int], None], Optional[DNDarray], bool], Union[DNDarray, bool]
] = lambda self, axis=None, out=None, keepdims=False: all(self, axis, out, keepdims)
DNDarray.all.__doc__ = all.__doc__


def allclose(
    x: DNDarray, y: DNDarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    """
    Test whether two tensors are element-wise equal within a tolerance. Returns ``True`` if ``|x-y|<=atol+rtol*|y|``
    for all elements of ``x`` and ``y``, ``False`` otherwise

    Parameters
    ----------
    x : DNDarray
        First array to compare
    y : DNDarray
        Second array to compare
    atol: float, optional
        Absolute tolerance.
    rtol: float, optional
        Relative tolerance (with respect to ``y``).
    equal_nan: bool, optional
        Whether to compare NaN’s as equal. If ``True``, NaN’s in ``x`` will be considered equal to NaN’s in ``y`` in
        the output array.

    Examples
    --------
    >>> x = ht.float32([[2, 2], [2, 2]])
    >>> ht.allclose(x, x)
    True
    >>> y = ht.float32([[2.00005, 2.00005], [2.00005, 2.00005]])
    >>> ht.allclose(x, y)
    False
    >>> ht.allclose(x, y, atol=1e-04)
    True
    """
    t1, t2 = __sanitize_close_input(x, y)

    # no sanitation for shapes of x and y needed, torch.allclose raises relevant errors
    try:
        _local_allclose = torch.tensor(torch.allclose(t1.larray, t2.larray, rtol, atol, equal_nan))
    except RuntimeError:
        promoted_dtype = torch.promote_types(t1.larray.dtype, t2.larray.dtype)
        _local_allclose = torch.tensor(
            torch.allclose(
                t1.larray.type(promoted_dtype),
                t2.larray.type(promoted_dtype),
                rtol,
                atol,
                equal_nan,
            )
        )

    # If x is distributed, then y is also distributed along the same axis
    if t1.comm.is_distributed():
        t1.comm.Allreduce(MPI.IN_PLACE, _local_allclose, MPI.LAND)

    return bool(_local_allclose.item())


DNDarray.allclose: Callable[[DNDarray, DNDarray, float, float, bool], bool] = (
    lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False: allclose(
        self, other, rtol, atol, equal_nan
    )
)
DNDarray.allclose.__doc__ = all.__doc__


def any(
    x, axis: Optional[int] = None, out: Optional[DNDarray] = None, keepdims: bool = False
) -> DNDarray:
    """
    Returns a :class:`~heat.core.dndarray.DNDarray` containing the result of the test whether any array elements along a
    given axis evaluate to ``True``.
    The returning array is one dimensional unless axis is not ``None``.

    Parameters
    ----------
    x : DNDarray
        Input tensor
    axis : int, optional
        Axis along which a logic OR reduction is performed. With ``axis=None``, the logical OR is performed over all
        dimensions of the array.
    out : DNDarray, optional
        Alternative output tensor in which to place the result. It must have the same shape as the expected output.
        The output is a array with ``datatype=bool``.
    keepdims : bool, optional
        If this is set to ``True``, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array.

    Examples
    --------
    >>> x = ht.float32([[0.3, 0, 0.5]])
    >>> x.any()
    DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
    >>> x.any(axis=0)
    DNDarray([ True, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    >>> x.any(axis=1)
    DNDarray([True], dtype=ht.bool, device=cpu:0, split=None)
    >>> y = ht.int32([[0, 0, 1], [0, 0, 0]])
    >>> res = ht.zeros(3, dtype=ht.bool)
    >>> y.any(axis=0, out=res)
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    >>> res
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """

    def local_any(t, *args, **kwargs):
        return torch.any(t != 0, *args, **kwargs)

    if keepdims and axis is None:
        axis = tuple(range(x.ndim))

    return _operations.__reduce_op(
        x, local_any, MPI.LOR, axis=axis, out=out, neutral=0, keepdims=keepdims
    )


DNDarray.any: Callable[[DNDarray, Optional[int], Optional[DNDarray], bool], DNDarray] = (
    lambda self, axis=None, out=None, keepdims=False: any(self, axis, out, keepdims)
)
DNDarray.any.__doc__ = any.__doc__


def in1d(
    ar1: DNDarray,
    ar2: Union[DNDarray, bool, int, float, Sequence],
    assume_unique: bool = False,
    invert: bool = False,
) -> DNDarray:
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as ``ar1`` that is ``True`` where an element of ``ar1``
    is in ``ar2`` and ``False`` otherwise. Returns a flattened 1-D result, see
    :func:`heat.isin <heat.core.logical.isin>` for the version that preserves the shape of the input.

    Parameters
    ----------
    ar1 : DNDarray
        Input array, likely with 1 dimensions.
    ar2 : DNDarray or array_like
        The values against which to test each value of ``ar1``.
    assume_unique : bool, optional
        If ``True``, the input arrays are both assumed to be unique, which can speed up the calculation.
    invert : bool, optional
        If ``True``, the values in the returned array are inverted, as if calculating ``ar1 not in ar2``.

    See Also
    --------
    :func:`heat.isin()`
        ``isin`` works elementwise instead, i.e. it preserves the shape of the input.

    Examples
    --------
    >>> a = ht.array([1, 2, 3, 2, 5], split=0)
    >>> ht.in1d(a, [2, 5])
    DNDarray([False,  True, False,  True,  True], dtype=ht.bool, device=cpu:0, split=0)
    """
    sanitation.sanitize_in(ar1)
    return manipulations.flatten(isin(ar1, ar2, assume_unique=assume_unique, invert=invert))


def isclose(
    x: DNDarray, y: DNDarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> DNDarray:
    """
    Returns a boolean :class:`~heat.core.dndarray.DNDarray`, with elements ``True`` where ``a`` and ``b`` are equal
    within the given tolerance. If both ``x`` and ``y`` are scalars, returns a single boolean value.

    Parameters
    ----------
    x : DNDarray
        Input array to compare.
    y : DNDarray
        Input array to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan : bool
        Whether to compare NaN’s as equal. If ``True``, NaN’s in x will be considered equal to NaN’s in y in the output
        array.
    """
    t1, t2 = __sanitize_close_input(x, y)

    # no sanitation for shapes of x and y needed, torch.isclose raises relevant errors
    _local_isclose = torch.isclose(t1.larray, t2.larray, rtol, atol, equal_nan)

    # If x is distributed, then y is also distributed along the same axis
    if t1.comm.is_distributed() and t1.split is not None:
        output_gshape = stride_tricks.broadcast_shape(t1.gshape, t2.gshape)
        res = torch.empty(output_gshape, device=t1.device.torch_device).bool()
        t1.comm.Allgather(_local_isclose, res)
        result = DNDarray(
            res,
            gshape=output_gshape,
            dtype=types.bool,
            split=t1.split,
            device=t1.device,
            comm=t1.comm,
            balanced=t1.is_balanced,
        )
    else:
        if _local_isclose.dim() == 0:
            # both x and y are scalars, return a single boolean value
            result = bool(_local_isclose.item())
        else:
            result = DNDarray(
                _local_isclose,
                gshape=tuple(_local_isclose.shape),
                dtype=types.bool,
                split=None,
                device=t1.device,
                comm=t1.comm,
                balanced=t1.is_balanced,
            )

    return result


def isfinite(x: DNDarray) -> DNDarray:
    """
    Test element-wise for finiteness (not infinity or not Not a Number) and return result as a boolean
    :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input tensor

    Examples
    --------
    >>> ht.isfinite(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([ True, False, False, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isfinite, x, None, no_cast=True)


def __interval_owner(pivots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Assigns each value to the rank owning its value interval: the number of pivots that are
    not greater than the value. Duplicated values always get the same owner.
    """
    if pivots.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.int64, device=values.device)
    return torch.searchsorted(pivots, values, side="right").to(torch.int64)


def __complex64_to_key(value: torch.Tensor) -> torch.Tensor:
    """Map complex values to integer keys with equality-compatible signed-zero handling."""
    components = torch.view_as_real(value)
    components = torch.where(components == 0, 0.0, components)
    return torch.view_as_complex(components).view(torch.int64)


def __distributed_isin(
    l_elem: torch.Tensor,
    l_test: torch.Tensor,
    comm,
    device: str,
    assume_unique: bool,
    invert: bool,
) -> torch.Tensor:
    """
    Membership test of flat local element values against a distributed test set, where the
    test values are scattered across the communicator.

    The sorted, locally deduplicated test set is partitioned into per-rank value intervals: each
    process publishes a quantile sketch of its local shard, the pooled sketch yields the interval
    pivots, and test values are exchanged once so that each process owns exactly one interval.
    Query values are then routed to the owner of their value interval, tested locally with
    ``torch.isin``, and the results are routed back. With even test-value and query distributions,
    per-process memory and communication scale with the local shards of both inputs. Skewed
    distributions can concentrate data and work on individual interval owners.
    """
    p = comm.Get_size()
    l_elem = l_elem.reshape(-1)
    l_test = l_test.reshape(-1)

    # empty global test set: membership trivially reduces to ``invert``
    total = torch.tensor([l_test.numel()], dtype=torch.int64)
    total_buf = torch.empty_like(total)
    comm.Allreduce(total, total_buf, op=MPI.SUM)
    if total_buf.item() == 0:
        return torch.full(l_elem.shape, fill_value=invert, dtype=torch.bool, device=l_elem.device)

    # map to a key space torch can sort on; the mapping is bijective after normalizing zeros
    nan_elements = None
    if l_test.dtype == torch.bool:
        l_test, l_elem = l_test.to(torch.int8), l_elem.to(torch.int8)
    elif l_test.dtype == torch.complex64:
        nan_elements = torch.isnan(l_elem)
        l_test, l_elem = __complex64_to_key(l_test), __complex64_to_key(l_elem)
    elif l_test.dtype == torch.complex128:
        raise NotImplementedError(
            "isin does not support complex128 with distributed test_elements; "
            "pass the test values with split=None instead."
        )

    # Discard duplicates locally before constructing the sketch and exchanging values. Besides
    # shrinking communication for repeated values, this avoids concentrating redundant copies
    # on the rank owning their interval.
    l_test = torch.unique(l_test, sorted=True)

    # 1. quantile sketch of the sorted local shards (small, independent of shard size)
    s = max(2, p)
    n = l_test.numel()
    if n == 0:
        sample = torch.empty((0,), dtype=l_test.dtype, device=device)
    elif n > s:
        # float64 on purpose: float32 linspace cannot represent n-1 exactly
        # for n > 2**24 and may round the last sample index out of bounds
        sample = l_test.sort().values[torch.linspace(0, n - 1, s, dtype=torch.float64).long()]
    else:
        sample = l_test
    sample_count = torch.tensor([sample.numel()], dtype=torch.int64)
    sample_counts = torch.empty((p,), dtype=torch.int64)
    comm.Allgather(sample_count, sample_counts)
    sample_counts = sample_counts.tolist()
    pooled = torch.empty((sum(sample_counts),), dtype=l_test.dtype, device=device)
    comm.Allgatherv(
        sample, (pooled, sample_counts, [0] + np.cumsum(sample_counts)[:-1].tolist()), recv_axis=0
    )

    # 2. interval pivots from the pooled sketch
    if p > 1:
        pooled = pooled.sort().values
        cuts = torch.round(
            torch.arange(1, p, dtype=torch.float, device=device) * pooled.numel() / p
        ).long()
        pivots = pooled[torch.clamp(cuts, 0, pooled.numel() - 1)]
    else:
        pivots = torch.empty((0,), dtype=l_test.dtype, device=device)

    # 3. exchange test values so each process owns one interval (and only that interval)
    sorted_test = l_test.sort().values
    send_counts = torch.bincount(__interval_owner(pivots, sorted_test), minlength=p)
    recv_counts = torch.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    total_recv = recv_counts.sum().item()
    owned = torch.empty((total_recv,), dtype=l_test.dtype, device=device)
    comm.Alltoallv(
        (
            sorted_test,
            send_counts.tolist(),
            [0] + list(np.cumsum(send_counts.tolist())[:-1]),
        ),
        (owned, recv_counts.tolist(), [0] + list(np.cumsum(recv_counts.tolist())[:-1])),
    )
    # unique test values for the local kernel; duplicates are redundant for membership
    owned = torch.unique(owned)

    # 4. route each query value to the owner of its value interval
    # (no early return for empty local elements: all ranks must join every collective)
    result = torch.zeros(l_elem.shape, dtype=torch.bool, device=device)

    owners = __interval_owner(pivots, l_elem)
    order, send_counts_t, send_displs_t, recv_counts_t, recv_displs_t = _prepare_alltoallv_routing(
        owners, comm
    )
    send_counts = send_counts_t.tolist()
    send_disps = send_displs_t.tolist()
    recv_counts = recv_counts_t.tolist()
    recv_disps = recv_displs_t.tolist()

    send_vals = l_elem[order]
    total_recv = sum(recv_counts)
    recv_vals = torch.empty((total_recv,), dtype=l_elem.dtype, device=device)
    comm.Alltoallv(
        (send_vals, send_counts, send_disps),
        (recv_vals, recv_counts, recv_disps),
    )

    # 5. answer the received batches with the local torch kernel
    if assume_unique:
        # Keep origins separate: concatenating unique shards from replicated inputs can create
        # duplicates, which would violate the precondition of torch.isin's unique algorithm.
        hits = torch.empty(total_recv, dtype=torch.bool, device=device)
        for origin in range(p):
            lo, hi = recv_disps[origin], recv_disps[origin] + recv_counts[origin]
            if hi > lo:
                hits[lo:hi] = torch.isin(recv_vals[lo:hi], owned, assume_unique=True)
    else:
        hits = torch.isin(recv_vals, owned)

    # 6. route the results back in the destination-blocked order of ``order``
    answers = torch.empty(l_elem.numel(), dtype=torch.bool, device=device)
    comm.Alltoallv(
        (hits, recv_counts, recv_disps),
        (answers, send_counts, send_disps),
    )
    result[order] = answers
    if nan_elements is not None:
        result[nan_elements] = False

    return torch.logical_not(result) if invert else result


def __isin_local(
    local: torch.Tensor, test: torch.Tensor, assume_unique: bool, invert: bool
) -> torch.Tensor:
    """
    Element-wise membership test on local tensors, mirroring the behavior of ``numpy.isin``.
    ``torch.isin`` does not support boolean and complex dtypes. Booleans are mapped to ``int8``;
    ``complex64`` values use integer keys after normalizing signed zeros, with NaNs excluded from
    matches, while ``complex128`` uses a component-wise comparison (there is no 16-byte integer
    type to bitcast to).
    """
    if local.dtype == torch.bool or test.dtype == torch.bool:
        local = local.to(torch.int8)
        test = test.to(torch.int8)
    elif local.dtype == torch.complex64:
        shape = local.shape
        local = local.reshape(-1)
        nan_elements = torch.isnan(local)
        local_key = __complex64_to_key(local)
        test_key = __complex64_to_key(test.reshape(-1))
        result = torch.isin(local_key, test_key, assume_unique=assume_unique)
        result[nan_elements] = False
        result = result.view(shape)
        return torch.logical_not(result) if invert else result
    elif local.dtype == torch.complex128:
        local_real = torch.view_as_real(local.reshape(-1))
        test_real = torch.view_as_real(test)
        # broadcasting comparison, memory usage is O(local.numel() * test.numel())
        result = (local_real.unsqueeze(-2) == test_real).all(dim=-1).any(dim=-1)
        result = result.view(local.shape)
        return torch.logical_not(result) if invert else result

    result = torch.isin(local, test, assume_unique=assume_unique, invert=invert)
    return result


def isin(
    element: Union[DNDarray, bool, int, float, Sequence],
    test_elements: Union[DNDarray, bool, int, float, Sequence],
    assume_unique: bool = False,
    invert: bool = False,
) -> DNDarray:
    """
    Calculate ``element in test_elements``, broadcasting over the values of ``element``.

    Returns a boolean ``DNDarray`` of the same shape as ``element`` that is ``True`` where
    ``element`` is in ``test_elements`` and ``False`` otherwise. Values in ``test_elements`` are
    considered globally, i.e. also values that reside on other processes count as members.

    Parameters
    ----------
    element : DNDarray or array_like
        Input array. Scalars and array-likes are accepted, mirroring numpy and the Python array
        API standard, as long as they can be converted to a :class:`~heat.core.dndarray.DNDarray`.
    test_elements : DNDarray or array_like
        The values against which to test each value of ``element``. May itself be distributed.
    assume_unique : bool, optional
        If ``True``, the input arrays are both assumed to be unique, which can speed up the calculation.
        Note that, in contrast to numpy, violating this assumption is not tolerated for
        ``test_elements``: the underlying torch algorithm silently returns wrong results if
        ``test_elements`` contains duplicates while ``assume_unique=True``.
    invert : bool, optional
        If ``True``, the values in the returned array are inverted, as if calculating
        ``element not in test_elements``.

    Notes
    -----
    If ``test_elements`` is distributed, the test values are not gathered or replicated.
    Instead, each process sketches its locally deduplicated shard. The pooled sketch contains
    ``O(comm.size)`` samples per rank, hence ``O(comm.size**2)`` samples in total, independent
    of the input sizes. Its quantiles define per-rank value intervals, the test values are
    exchanged once so that each process owns a single interval, and every query value is routed
    to the process owning its value interval and tested there with ``torch.isin``. For balanced
    value distributions, per-process memory and communication scale with the local shards of
    both inputs; skewed test values or queries can concentrate data and work on individual
    interval owners. The result keeps the ``split`` of ``element``.

    ``complex128`` elements are not supported when ``test_elements`` is distributed; pass
    the test values with ``split=None`` in that case. In the distributed path, each owning
    process deduplicates its interval before the local test, so duplicated test values do
    not affect the ``assume_unique`` contract there.

    See Also
    --------
    :func:`heat.in1d()`
        ``in1d`` flattens the input instead of preserving the shape of ``element``.

    Examples
    --------
    >>> element = ht.array([1, 2, 3, 4, 3, 2, 2, 1], split=0)
    >>> test_elements = ht.array([1, 3, 5, 7])
    >>> ht.isin(element, test_elements)
    DNDarray([ True, False,  True, False,  True, False, False,  True], dtype=ht.bool, device=cpu:0, split=0)
    >>> ht.isin(element, test_elements, invert=True)
    DNDarray([False,  True, False,  True, False,  True,  True, False], dtype=ht.bool, device=cpu:0, split=0)
    """
    # numpy and the array API standard accept scalar/array-like elements
    if not isinstance(element, DNDarray):
        if isinstance(test_elements, DNDarray):
            element = factories.array(element, device=test_elements.device, comm=test_elements.comm)
        else:
            element = factories.array(element)
    sanitation.sanitize_in(element)

    if isinstance(test_elements, DNDarray):
        sanitation.sanitize_in(test_elements)
    else:
        test_elements = factories.array(
            test_elements, split=None, device=element.device, comm=element.comm
        )

    # match the numpy behavior of promoting the dtypes of both inputs
    common_dtype = types.promote_types(element.dtype, test_elements.dtype)
    torch_dtype = common_dtype.torch_type()
    l_elem = element.larray.to(torch_dtype)
    l_test = test_elements.larray.to(torch_dtype)

    if not test_elements.is_distributed() or test_elements.comm.Get_size() == 1:
        # test values are replicated (or only one process exists): direct local kernel
        result = __isin_local(
            l_elem, l_test.reshape(-1), assume_unique=assume_unique, invert=invert
        )
    else:
        # test values are scattered across the processes: interval-based distributed path
        result = __distributed_isin(
            l_elem,
            l_test,
            test_elements.comm,
            element.device.torch_device,
            assume_unique=assume_unique,
            invert=invert,
        )

    return DNDarray(
        result.reshape(element.lshape),
        gshape=element.gshape,
        dtype=types.bool,
        split=element.split,
        device=element.device,
        comm=element.comm,
        balanced=element.is_balanced(),
    )


def isinf(x: DNDarray) -> DNDarray:
    """
    Test element-wise for positive or negative infinity and return result as a boolean
    :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input tensor

    Examples
    --------
    >>> ht.isinf(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False,  True,  True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isinf, x, None, no_cast=True)


def isnan(x: DNDarray) -> DNDarray:
    """
    Test element-wise for NaN and return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isnan, x, None, no_cast=True)


def isneginf(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Test if each element of `x` is negative infinite, return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor
    out : DNDarray, optional
          Alternate output array in which to place the result. It must have the same shape as the expected output
          and its type is preserved.

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, False, True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isneginf, x, out, no_cast=True)


def isposinf(x: DNDarray, out: Optional[DNDarray] = None):
    """
    Test if each element of `x` is positive infinite, return result as a boolean :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x   : DNDarray
          Input tensor
    out : DNDarray, optional
          Alternate output array in which to place the result. It must have the same shape as the expected output
          and its type is preserved.

    Examples
    --------
    >>> ht.isnan(ht.array([1, ht.inf, -ht.inf, ht.nan]))
    DNDarray([False, True, False, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.isposinf, x, out, no_cast=True)


DNDarray.isclose: Callable[[DNDarray, DNDarray, float, float, bool], DNDarray] = (
    lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False: isclose(
        self, other, rtol, atol, equal_nan
    )
)
DNDarray.isclose.__doc__ = isclose.__doc__


def logical_and(x: DNDarray, y: DNDarray) -> DNDarray:
    """
    Compute the truth value of ``x`` AND ``y`` element-wise. Returns a boolean :class:`~heat.core.dndarray.DNDarray` containing the truth value of ``x`` AND ``y`` element-wise.

    Parameters
    ----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    --------
    >>> ht.logical_and(ht.array([True, False]), ht.array([False, False]))
    DNDarray([False, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__binary_op(
        torch.logical_and, types.bool(x, device=x.device), types.bool(y, device=y.device)
    )


def logical_not(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Computes the element-wise logical NOT of the given input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input array
    out : DNDarray, optional
        Alternative output array in which to place the result. It must have the same shape as the expected output.
        The output is a :class:`~heat.core.dndarray.DNDarray` with ``datatype=bool``.

    Examples
    --------
    >>> ht.logical_not(ht.array([True, False]))
    DNDarray([False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.logical_not, x, out)


def logical_or(x: DNDarray, y: DNDarray) -> DNDarray:
    """
    Returns boolean :class:`~heat.core.dndarray.DNDarray` containing the element-wise logical NOT of the given
    input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    --------
    >>> ht.logical_or(ht.array([True, False]), ht.array([False, False]))
    DNDarray([ True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__binary_op(
        torch.logical_or, types.bool(x, device=x.device), types.bool(y, device=y.device)
    )


def logical_xor(x: DNDarray, y: DNDarray) -> DNDarray:
    """
    Computes the element-wise logical XOR of the given input :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    x : DNDarray
        Input array of same shape
    y : DNDarray
        Input array of same shape

    Examples
    --------
    >>> ht.logical_xor(ht.array([True, False, True]), ht.array([True, False, False]))
    DNDarray([False, False,  True], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__binary_op(torch.logical_xor, x, y)


def __sanitize_close_input(x: DNDarray, y: DNDarray) -> Tuple[DNDarray, DNDarray]:
    """
    Makes sure that both ``x`` and ``y`` are :class:`~heat.core.dndarray.DNDarray`.
    Provides copies of ``x`` and ``y`` distributed along the same split axis (if original split axes do not match).

    Parameters
    ----------
    x : DNDarray
        The left-hand side operand.
    y : DNDarray
        The right-hand side operand.

    Raises
    ------
    TypeError
        If ``x`` is neither :class:`~heat.core.dndarray.DNDarray` or numeric scalar
    """

    def sanitize_input_type(
        x: Union[int, float, DNDarray], y: Union[int, float, DNDarray]
    ) -> DNDarray:
        """
        Verifies that ``x`` and ``y`` are either scalar, or a :class:`~heat.core.dndarray.DNDarray`.
        In the former case, the scalar is wrapped in a :class:`~heat.core.dndarray.DNDarray`.

        Parameters
        ----------
        x : Union[int, float, DNDarray]
            The left-hand side operand.
        y : Union[int, float, DNDarray]
            The right-hand side operand.

        Raises
        ------
        TypeError
            If ``x`` or ``y`` are not
        """
        if not isinstance(x, DNDarray):
            if np.ndim(x) != 0:
                raise TypeError(f"Expected DNDarray or numeric scalar, input was {type(x)}")

            dtype = getattr(x, "dtype", float)
            device = getattr(y, "device", None)
            x = factories.array(x, dtype=dtype, device=device)

        return x

    x = sanitize_input_type(x, y)
    y = sanitize_input_type(y, x)

    # if one of the DNDarrays is distributed and the other is not
    if x.is_distributed() and not y.is_distributed() and y.ndim > 0:
        t2 = factories.array(y.larray, device=x.device, split=x.split)
        x, t2 = sanitation.sanitize_distribution(x, t2, target=x)
        return x, t2

    # if y is distributed, x is not distributed, and x is not a scalar
    elif y.is_distributed() and not x.is_distributed() and x.ndim > 0:
        t1 = factories.array(x.larray, device=y.device, split=y.split)
        t1, y = sanitation.sanitize_distribution(t1, y, target=y)
        return t1, y

    elif x.split != y.split:
        t2 = manipulations.resplit(y, axis=x.split)
        return x, t2

    else:
        return x, y


def signbit(x: DNDarray, out: Optional[DNDarray] = None) -> DNDarray:
    """
    Checks if signbit is set element-wise (less than zero).

    Parameters
    ----------
    x : DNDarray
        The input array.
    out : DNDarray, optional
        The output array.

    Examples
    --------
    >>> a = ht.array([2, -1.3, 0])
    >>> ht.signbit(a)
    DNDarray([False,  True, False], dtype=ht.bool, device=cpu:0, split=None)
    """
    return _operations.__local_op(torch.signbit, x, out, no_cast=True)
