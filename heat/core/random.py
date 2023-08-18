"""Provides parallel random number generators (pRNG)"""
from __future__ import annotations

import time
import torch

from typing import List, Optional, Tuple, Type, Union

from . import communication
from . import devices
from . import factories
from . import logical
from . import stride_tricks
from . import types

from .communication import Communication
from .devices import Device
from .dndarray import DNDarray
from .types import datatype

__all__ = [
    "get_state",
    "normal",
    "permutation",
    "rand",
    "ranf",
    "randint",
    "random_integer",
    "randn",
    "random",
    "random_sample",
    "randperm",
    "sample",
    "seed",
    "set_state",
    "standard_normal",
]

# introduce the global random state variables, will be correctly initialized at the end of file
__seed: int = None
"""The current global random seed for the pRNG"""
__counter: Optional[int] = None
"""Stateful counter tracking the already pulled random numbers from the current seed"""


# float conversion constants
__INT32_TO_FLOAT32: float = 1.0 / 8388608.0
"""Bit-mask for float-32 that retains the mantissa bits only via multiplication in order to convert to int32"""
__INT64_TO_FLOAT64: float = 1.0 / 9007199254740992.0
"""Bit-mask for float-64 that retains the mantissa bits only via multiplication in order to convert to int64"""
__KUNDU_INVERSE: float = 1.0 / 0.3807
"""magical number for generalized exponential random numbers by Kundu, see __kundu_inverse below for more information"""


def __counter_sequence(
    shape: Tuple[int, ...],
    dtype: Type[torch.dtype],
    split: Optional[int],
    device: Device,
    comm: Communication,
) -> Tuple[torch.tensor, torch.tensor, Tuple[int, ...], slice]:
    """
    Generates a sequence of numbers to be used as the "clear text" for the threefry encryption, i.e. the pseudo random
    number generator. Due to the fact that threefry always requires pairs of inputs, the input sequence may not just be
    a simple range including the global offset, but rather needs to be to independent vectors, one containing the range
    and the other having the interleaved high-bits counter in it.
    Returns the high-bits and low-bits vectors for the threefry encryption (``torch.tensor``), the shape ``x_0`` and
    ``x_1`` and the slice that needs to be applied to the resulting random number tensor.

    Parameters
    ----------
    shape : tuple[int, ...]
        The global shape of the random tensor to be generated.
    dtype : torch.dtype
        The data type of the elements to be generated. Needs to be either ``torch.int32`` or ``torch.int64``.
    split : int or None
        The split axis along which the random number tensor is split
    device : Device
        Specifies the device the tensor shall be allocated on.
    comm: Communication
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    x_0 : torch.Tensor
        The high-bits vector for the threefry encryption.
    x_1 : torch.Tensor
        The low-bits vector for the threefry encryption.
    lshape : tuple of ints
        The shape x_0 and x_1 need to be reshaped to after encryption. May be slightly larger than the actual local
        portion of the random number tensor due to sequence overlaps of the counter sequence.
    slice : python slice
        The slice that needs to be applied to the resulting random number tensor
    """
    # get the global random state into the function, might want to factor this out into a class later
    global __counter

    # Share this initial local state to update it correctly later
    tmp_counter = __counter
    rank = comm.Get_rank()
    size = comm.Get_size()
    max_count = 0xFFFFFFFF if dtype == torch.int32 else 0xFFFFFFFFFFFFFFFF

    # extract the counter state of the random number generator
    if dtype is torch.int32:
        c_0 = (__counter & (max_count << 32)) >> 32
    else:  # torch.int64
        c_0 = (__counter & (max_count << 64)) >> 64
    c_1 = __counter & max_count
    total_elements = torch.prod(torch.tensor(shape))
    if total_elements.item() > 2 * max_count:
        raise ValueError(f"Shape is to big with {total_elements} elements")

    if split is None:
        values = total_elements.item() // 2 + total_elements.item() % 2
        even_end = total_elements.item() % 2 == 0
        lslice = slice(None) if even_end else slice(None, -1)
        start = c_1
        end = start + int(values)
        lshape = shape
    else:
        offset, lshape, _ = comm.chunk(shape, split)
        counts, displs, _ = comm.counts_displs_shape(shape, split)

        # Calculate number of local elements per process
        local_elements = [total_elements.item() / shape[split] * counts[i] for i in range(size)]
        cum_elements = torch.cumsum(torch.tensor(local_elements, device=device.torch_device), dim=0)

        # Calculate the correct borders and slices
        even_start = True if rank == 0 else cum_elements[rank - 1] % 2 == 0
        start = c_1 if rank == 0 else int(cum_elements[rank - 1] / 2) + c_1
        elements = local_elements[rank] / 2
        lslice = slice(None)
        if even_start:
            # no overlap with previous processes
            if elements == int(elements):
                # even number of elements
                end = int(elements)
            else:
                # odd number of elements
                end = int(elements) + 1
                lslice = slice(None, -1)
        else:
            # overlap with previous processes
            if elements == int(elements):
                # even number of elements
                end = int(elements) + 1
                lslice = slice(1, -1)
            else:
                # Odd number of elements
                end = int(elements) + 1
                lslice = slice(1, None)
        start = int(start)
        end += start

    # check x_1 for overflow
    lrange = [start, end]
    signed_mask = 0x7FFFFFFF if dtype == torch.int32 else 0x7FFFFFFFFFFFFFFF
    diff = 0 if lrange[1] <= signed_mask else lrange[1] - signed_mask
    lrange[0], lrange[1] = lrange[0] - diff, lrange[1] - diff

    # create x_1 counter sequence
    x_1 = torch.arange(*lrange, dtype=dtype, device=device.torch_device)
    while diff > signed_mask:
        # signed_mask is maximum that can be added at a time because torch does not support unit64 or unit32
        x_1 += signed_mask
        diff -= signed_mask
    x_1 += diff

    # generate the x_0 counter sequence
    x_0 = torch.empty_like(x_1)
    diff = c_0 - signed_mask
    if diff > 0:
        # same problem as for x_1 with the overflow
        x_0.fill_(signed_mask)
        while diff > signed_mask:
            x_0 += signed_mask
            diff -= signed_mask
        x_0 += diff
    else:
        x_0.fill_(c_0)

    # detect if x_0 needs to be increased for current values
    if end > max_count:
        if start > max_count:
            # x_0 changed in previous process, increase all values
            x_0 += 1
        else:
            # x_0 changes after reaching the overflow in this process
            x_0[-(end - max_count - 1) :] += 1

    # correctly increase the counter variable
    used_values = int(torch.ceil(total_elements / 2))
    # increase counter but not over 128 bit
    tmp_counter += used_values
    __counter = tmp_counter & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # 128-bit mask

    return x_0.contiguous(), x_1.contiguous(), lshape, lslice


def get_state() -> Tuple[str, int, int, int, float]:
    """
    Return a tuple representing the internal state of the generator. The returned tuple has the following items:

    1. The string ‘Threefry’,

    2. The Threefry key value, aka seed,

    3. The internal counter value,

    4. An integer has_gauss, always set to 0 (present for compatibility with numpy) and

    5. A float cached_gaussian, always set to 0.0 (present for compatibility with numpy).
    """
    return "Threefry", __seed, __counter, 0, 0.0


def __int32_to_float32(values: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of 32-bit (random) numbers to matching single-precision floating point numbers (equally 32-bit) in
    the bounded interval [0.0, 1.0). Extracts the 23 least-significant bits of the integers (0x7fffff) and sets them to
    be the mantissa of the floating point number. Interval is bound by dividing by 2^23 = 8388608.0.

    Parameters
    ----------
    values : torch.Tensor (int32)
        Values to be converted to floating points numbers in interval [0.0, 1.0).
    """
    return (values & 0x7FFFFF).type(torch.float32) * __INT32_TO_FLOAT32


def __int64_to_float64(values: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of 64-bit (random) numbers to matching double-precision floating point numbers (equally 64-bit) in
    the bounded interval [0.0, 1.0). Extracts the 53 least-significant bits of the integers (0x1fffffffffffff) and sets
    them to be the mantissa of the floating point number. Interval is bound by dividing by 2^53 = 9007199254740992.0.

    Parameters
    ----------
    values : torch.Tensor (int64)
        Values to be converted to floating points numbers in interval [0.0, 1.0).
    """
    return (values & 0x1FFFFFFFFFFFFF).type(torch.float64) * __INT64_TO_FLOAT64


def __kundu_transform(values: torch.Tensor) -> torch.Tensor:
    """
    Transforms uniformly distributed floating point random values in the interval [0.0, 1.0) into normal distributed
    floating point random values with mean 0.0 and standard deviation 1.0. The algorithm makes use of the generalized
    exponential distribution transformation [1].

    Parameters
    ----------
    values : torch.Tensor
        A tensor containing uniformly distributed floating point values in the interval [0.0, 1.0).

    References
    ----------
    [1] Boiroju, N. K. and Reddy, K. M., "Generation of Standard Normal Random Numbers", Interstat, vol 5., 2012.
    """
    inner = 1 - values**0.0775
    tiny = torch.finfo(inner.dtype).tiny
    return (torch.log(-torch.log(inner + tiny) + tiny) - 1.0821) * __KUNDU_INVERSE


def normal(
    mean: Union[float, DNDarray] = 0.0,
    std: Union[float, DNDarray] = 1.0,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Returns an array filled with random numbers from a normal distribution whose mean and standard deviation are given.
    If `std` and `mean` are DNDarrays, they have to match `shape`.

    Parameters
    ----------
    mean : float or DNDarray
        The mean of the distribution.
    std : float or DNDarray
        The standard deviation of the distribution. Must be non-negative.
    shape : tuple[int]
        The shape of the returned array, should all be positive. If no argument is given a single random sample is
        generated.
    dtype : Type[datatype], optional
        The datatype of the returned values. Has to be one of :class:`~heat.core.types.float32` or
        :class:`~heat.core.types.float64`.
    split : int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device`  the array shall be allocated on, defaults to globally
        set default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    See Also
    --------
    randn
        Uses the standard normal distribution
    standard_noramal
        Uses the standard normal distribution

    Examples
    --------
    >>> ht.random.normal(ht.array([-1,2]), ht.array([0.5, 2]), (2,))
    DNDarray([-1.4669,  1.6596], dtype=ht.float64, device=cpu:0, split=None)
    """
    if not (isinstance(mean, (float, int))) and not isinstance(mean, DNDarray):
        raise TypeError("'mean' must be float or DNDarray")
    if not (isinstance(std, (float, int))) and not isinstance(std, DNDarray):
        raise TypeError("'mean' must be float or DNDarray")

    if ((isinstance(std, float) or isinstance(std, int)) and std < 0) or (
        isinstance(std, DNDarray) and logical.any(std < 0)
    ):
        raise ValueError("'std' must be non-negative")

    return mean + std * standard_normal(shape, dtype, split, device, comm)


def permutation(x: Union[int, DNDarray]) -> DNDarray:
    """
    Randomly permute a sequence, or return a permuted range. If ``x`` is a multi-dimensional array, it is only shuffled
    along its first index.

    Parameters
    -----------
    x : int or DNDarray
        If ``x`` is an integer, call :func:`heat.random.randperm <heat.core.random.randperm>`. If ``x`` is an array,
        make a copy and shuffle the elements randomly.

    See Also
    -----------
    :func:`heat.random.randperm <heat.core.random.randperm>` for randomly permuted ranges.

    Examples
    ----------
    >>> ht.random.permutation(10)
    DNDarray([9, 1, 5, 4, 8, 2, 7, 6, 3, 0], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.random.permutation(ht.array([1, 4, 9, 12, 15]))
    DNDarray([ 9,  1, 12,  4, 15], dtype=ht.int64, device=cpu:0, split=None)
    >>> arr = ht.arange(9).reshape((3, 3))
    >>> ht.random.permutation(arr)
    DNDarray([[3, 4, 5],
              [6, 7, 8],
              [0, 1, 2]], dtype=ht.int32, device=cpu:0, split=None)
    """
    if isinstance(x, int):
        return randperm(x)
    if not isinstance(x, DNDarray):
        raise TypeError("x must be int or DNDarray")

    # random permutation
    recv = torch.randperm(x.shape[0], device=x.device.torch_device)

    # rearrange locally
    if (x.split is None) or (x.split != 0):
        return x[recv]

    # split == 0 -> need for communication
    if x.lshape[0] > 0:
        cumsum = [x.comm.chunk(x.gshape, 0, i)[0] for i in range(0, x.comm.size)]
        cumsum.append(x.shape[0])

        send = torch.argsort(recv)
        size = cumsum[x.comm.rank + 1] - cumsum[x.comm.rank]
        torch_cumsum = torch.tensor(cumsum, device=x.device.torch_device)

        buf = []
        requests = []

        for i in range(size):
            proc_recv = torch.where(recv[torch_cumsum[x.comm.rank] + i] < torch_cumsum)[0][0] - 1
            buf.append(torch.empty_like(x.lloc[i]))
            requests.append(x.comm.Irecv(buf[-1], proc_recv, tag=i))

            proc_send = torch.where(send[torch_cumsum[x.comm.rank] + i] < torch_cumsum)[0][0] - 1
            tag = send[torch_cumsum[x.comm.rank] + i] - torch_cumsum[proc_send]
            requests.append(x.comm.Isend(x.lloc[i].clone(), proc_send, tag=tag))

        for req in requests:
            req.Wait()

        data = torch.stack(buf)
    else:
        data = torch.empty_like(x.larray)

    return DNDarray(
        data,
        gshape=x.gshape,
        dtype=x.dtype,
        split=x.split,
        device=x.device,
        comm=x.comm,
        balanced=x.is_balanced,
    )


def rand(
    *args: List[int],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Random values in a given shape. Create a :class:`~heat.core.dndarray.DNDarray` of the given shape and populate it
    with random samples from a uniform distribution over :math:`[0, 1)`.

    Parameters
    ----------
    d1,d2,…,dn : List[int,...]
        The dimensions of the returned array, should all be positive. If no argument is given a single random samples is
        generated.
    dtype : Type[datatype], optional
        The datatype of the returned values. Has to be one of :class:`~heat.core.types.float32` or
        :class:`~heat.core.types.float64`.
    split : int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    Raises
    ------
    ValueError
        If there are negative or not-integer convertible dimensions given or if the passed ``dtype`` was non-floating
        point.

    Examples
    --------
    >>> ht.rand(3)
    DNDarray([0.1921, 0.9635, 0.5047], dtype=ht.float32, device=cpu:0, split=None)
    """
    # if args are not set, generate a single sample
    if not args:
        args = (1,)

    # ensure that the passed dimensions are positive integer-likes
    shape = tuple(int(ele) for ele in args)
    if any(ele <= 0 for ele in shape):
        raise ValueError("negative dimensions are not allowed")

    # make sure the remaining parameters are of proper type
    split = stride_tricks.sanitize_axis(shape, split)
    device = devices.sanitize_device(device)
    comm = communication.sanitize_comm(comm)
    balanced = True

    # generate the random sequence
    if dtype == types.float32:
        x_0, x_1, lshape, lslice = __counter_sequence(shape, torch.int32, split, device, comm)
        x_0, x_1 = __threefry32(x_0, x_1, seed=__seed)

        # combine the values into one tensor and convert them to floats
        values = __int32_to_float32(torch.stack([x_0, x_1], dim=1).flatten()[lslice]).reshape(
            lshape
        )
    elif dtype == types.float64:
        x_0, x_1, lshape, lslice = __counter_sequence(shape, torch.int64, split, device, comm)
        x_0, x_1 = __threefry64(x_0, x_1, seed=__seed)

        # combine the values into one tensor and convert them to floats
        values = __int64_to_float64(torch.stack([x_0, x_1], dim=1).flatten()[lslice]).reshape(
            lshape
        )
    else:
        # Unsupported type
        raise ValueError(f"dtype is none of ht.float32 or ht.float64 but was {dtype}")

    return DNDarray(values, shape, dtype, split, device, comm, balanced)


def randint(
    low: int,
    high: Optional[int] = None,
    size: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[Type[datatype]] = types.int32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    r"""
    Random values in a given shape. Create a tensor of the given shape and populate it with random integer samples from
    a uniform distribution over :math:`[low, high)` or :math:`[0, low)` if ``high`` is not provided.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless `high=None`, in which case this parameter
        is one above the highest such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior
        if `high=None`).
    size : int or Tuple[int,...], optional
        Output shape. If the given shape is, e.g., :math:`(m, n, k)`, then :math:`m \times n \times k` samples are drawn.
        Default is None, in which case a single value is returned.
    dtype : datatype, optional
        Desired datatype of the result. Must be an integer type, defaults to int32.
    split : int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    Raises
    -------
    TypeError
        If one of low or high is not an int.
    ValueError
        If low >= high, dimensions are negative or the passed datatype is not an integer.

    Examples
    --------
    >>> ht.randint(3)
    DNDarray([4, 101, 16], dtype=ht.int32, device=cpu:0, split=None)
    """
    # determine range bounds
    if high is None:
        low, high = 0, int(low)
    else:
        low, high = int(low), int(high)
    if low >= high:
        raise ValueError("low >= high")
    span = high - low

    # sanitize shape
    if size is None:
        size = ()
    try:
        shape = tuple(int(ele) for ele in size)
    except TypeError:
        shape = (int(size),)
    else:
        if any(ele < 0 for ele in shape):
            raise ValueError("negative dimensions are not allowed")

    # sanitize the data type
    if dtype is None:
        dtype = types.int32
    dtype = types.canonical_heat_type(dtype)
    if dtype not in [types.int64, types.int32]:
        raise ValueError("Unsupported dtype for randint")
    torch_dtype = dtype.torch_type()

    # make sure the remaining parameters are of proper type
    split = stride_tricks.sanitize_axis(shape, split)
    device = devices.sanitize_device(device)
    comm = communication.sanitize_comm(comm)
    balanced = True

    # generate the random sequence
    x_0, x_1, lshape, lslice = __counter_sequence(shape, dtype.torch_type(), split, device, comm)
    if torch_dtype is torch.int32:
        x_0, x_1 = __threefry32(x_0, x_1, seed=__seed)
    else:  # torch.int64
        x_0, x_1 = __threefry64(x_0, x_1, seed=__seed)

    # stack the resulting sequence and normalize to given range
    values = torch.stack([x_0, x_1], dim=1).flatten()[lslice].reshape(lshape)
    # ATTENTION: this is biased and known, bias-free rejection sampling is difficult to do in parallel
    values = (values.abs_() % span) + low

    return DNDarray(values, shape, dtype, split, device, comm, balanced)


# alias
def random_integer(
    low: int,
    high: Optional[int] = None,
    size: Optional[Union[int, Tuple[int]]] = None,
    dtype: Optional[Type[datatype]] = types.int32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Alias for :func:`heat.random.randint <heat.core.random.randint>`.
    """
    return randint(low, high, size, dtype, split, device, comm)


def randn(
    *args: List[int],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Returns a tensor filled with random numbers from a standard normal distribution with zero mean and variance of one.

    Parameters
    ----------
    d1,d2,…,dn : List[int,...]
        The dimensions of the returned array, should be all positive.
    dtype : Type[datatype], optional
        The datatype of the returned values. Has to be one of :class:`~heat.core.types.float32` or
        :class:`~heat.core.types.float64`.
    split : int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    See Also
    --------
    normal
        Similar, but takes a tuple as its argumant.
    standard_normal
        Accepts arguments for mean and standard deviation.

    Raises
    -------
    TypeError
        If one of ``d1`` to ``dn`` is not an integer.
    ValueError
        If one of ``d1`` to ``dn`` is less or equal to 0.

    Examples
    --------
    >>> ht.randn(3)
    DNDarray([ 0.1921, -0.9635,  0.5047], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.randn(4, 4)
    DNDarray([[-1.1261,  0.5971,  0.2851,  0.9998],
              [-1.8548, -1.2574,  0.2391, -0.3302],
              [ 1.3365, -1.5212,  1.4159, -0.1671],
              [ 0.1260,  1.2126, -0.0804,  0.0907]], dtype=ht.float32, device=cpu:0, split=None)
    """
    # generate uniformly distributed random numbers first
    normal_tensor = rand(*args, dtype=dtype, split=split, device=device, comm=comm)
    # convert the the values to a normal distribution using the Kundu transform
    normal_tensor.larray = __kundu_transform(normal_tensor.larray)

    return normal_tensor


def randperm(
    n: int,
    dtype: Type[datatype] = types.int64,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    r"""
    Returns a random permutation of integers from :math:`0` to :math:`n - 1`.

    Parameters
    ----------
    n : int
        Upper, exclusive bound for the integer range.
    dtype : datatype, optional
        The datatype of the returned values.
    split : int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device`  the array shall be allocated on, defaults to globally
        set default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    Raises
    -------
    TypeError
        If ``n`` is not an integer.

    Examples
    --------
    >>> ht.random.randperm(4)
    DNDarray([2, 3, 1, 0], dtype=ht.int64, device=cpu:0, split=None)
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer.")

    device = devices.sanitize_device(device)
    comm = communication.sanitize_comm(comm)
    perm = torch.randperm(n, dtype=dtype.torch_type(), device=device.torch_device)

    return factories.array(perm, dtype=dtype, device=device, split=split, comm=comm)


def random(
    shape: Optional[Tuple[int]] = None,
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
):
    """
    Populates a :class:`~heat.core.dndarray.DNDarray` of the given shape with random samples from a continuous uniform
    distribution over :math:`[0.0, 1.0)`.

    Parameters
    ----------
    shape : tuple[int]
        The shape of the returned array, should all be positive. If no argument is given a single random sample is
        generated.
    dtype: Type[datatype], optional
        The datatype of the returned values. Has to be one of :class:`~heat.core.types.float32` or
        :class:`~heat.core.types.float64`.
    split : int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device`  the array shall be allocated on, defaults to globally
        set default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    Examples
    --------
    >>> ht.random.random_sample()
    0.47108547995356098
    >>> ht.random.random_sample((3,))
    DNDarray([0.30220482, 0.86820401, 0.1654503], dtype=ht.float32, device=cpu:0, split=None)
    """
    if not shape:
        shape = (1,)
    shape = stride_tricks.sanitize_shape(shape)
    return rand(*shape, dtype=dtype, split=split, device=device, comm=comm)


def ranf(
    shape: Optional[Tuple[int]] = None,
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
):
    """
    Alias for :func:`heat.random.random <heat.core.random.random>`.
    """
    return random(shape, dtype, split, device, comm)


def random_sample(
    shape: Optional[Tuple[int]] = None,
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
):
    """
    Alias for :func:`heat.random.random <heat.core.random.random>`.
    """
    return random(shape, dtype, split, device, comm)


def sample(
    shape: Optional[Tuple[int]] = None,
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
):
    """
    Alias for :func:`heat.random.random <heat.core.random.random>`.
    """
    return random(shape, dtype, split, device, comm)


def seed(seed: Optional[int] = None):
    """
    Seed the generator.

    Parameters
    ----------
    seed : int, optional
        Value to seed the algorithm with, if not set a time-based seed is generated.
    """
    if seed is None:
        seed = communication.MPI_WORLD.bcast(int(time.time() * 256))

    global __seed, __counter
    __seed = seed
    __counter = 0
    torch.manual_seed(seed)


def set_state(state: Tuple[str, int, int, int, float]):
    """
    Set the internal state of the generator from a tuple. The tuple has the following items:

    1. The string ‘Threefry’,

    2. The Threefry key value, aka seed,

    3. The internal counter value,

    4. An integer ``has_gauss``, ignored (present for compatibility with numpy), optional and

    5. A float ``cached_gaussian``, ignored (present for compatibility with numpy), optional.

    Parameters
    ----------
    state : Tuple[str, int, int, int, float]
        Sets the state of the random generator to the passed values. Allows to select seed and counter values manually.

    Raises
    ------
    TypeError
        If and improper state is passed.
    ValueError
        If one of the items in the state tuple is of wrong type or value.
    """
    if not isinstance(state, tuple) or len(state) not in [3, 5]:
        raise TypeError("state needs to be a three- or five-tuple")

    if state[0] != "Threefry":
        raise ValueError("algorithm must be 'Threefry'")

    global __seed, __counter
    __seed = int(state[1])
    __counter = int(state[2])


def standard_normal(
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Returns an array filled with random numbers from a standard normal distribution with zero mean and variance of one.

    Parameters
    ----------
    shape : tuple[int]
        The shape of the returned array, should all be positive. If no argument is given a single random sample is
        generated.
    dtype : Type[datatype], optional
        The datatype of the returned values. Has to be one of :class:`~heat.core.types.float32` or
        :class:`~heat.core.types.float64`.
    split : int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device`  the array shall be allocated on, defaults to globally
        set default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    See Also
    --------
    randn
        Similar, but accepts separate arguments for the shape dimensions.
    normal
        Equivalent function with arguments for the mean and standard deviation.

    Examples
    --------
    >>> ht.random.standard_normal((3,))
    DNDarray([ 0.1921, -0.9635,  0.5047], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.random.standard_normal((4, 4))
    DNDarray([[-1.1261,  0.5971,  0.2851,  0.9998],
              [-1.8548, -1.2574,  0.2391, -0.3302],
              [ 1.3365, -1.5212,  1.4159, -0.1671],
              [ 0.1260,  1.2126, -0.0804,  0.0907]], dtype=ht.float32, device=cpu:0, split=None)
    """
    if not shape:
        shape = (1,)
    shape = stride_tricks.sanitize_shape(shape)
    return randn(*shape, dtype=dtype, split=split, device=device, comm=comm)


def __threefry32(
    x0: torch.Tensor, x1: torch.Tensor, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1]. Returns
    Two vectors with num_samples / 2 (rounded-up) pseudo random numbers. This is the 32-bit version.

    Parameters
    ----------
    x0 : torch.Tensor
        Upper bits of the to be encoded random sequence
    x1 : torch.Tensor
        Lower bits of the to be encoded random sequence
    seed : int
        The seed, i.e. key, for the threefry32 encryption

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2,
    3", Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
    p. 16, 2011
    """
    samples = len(x0)

    # Seed is > 32 bit
    seed_32 = seed & 0x7FFFFFFF

    # set up key buffer
    ks_0 = torch.full((samples,), seed_32, dtype=torch.int32, device=x0.device)
    ks_1 = torch.full((samples,), seed_32, dtype=torch.int32, device=x1.device)
    ks_2 = torch.full((samples,), 466688986, dtype=torch.int32, device=x0.device)
    # ks_2 ^= ks_0
    # ks_2 ^= ks_1
    ks_2 = torch.bitwise_xor(torch.bitwise_xor(ks_2, ks_0), ks_1)

    # initialize output using the key
    x0 += ks_0
    x1 += ks_1

    # perform rounds
    # round 1
    x0 += x1
    x1 = (x1 << 13) | ((x1 >> 19) & 0x1FFF)
    x1 = torch.bitwise_xor(x1, x0)
    # x1 ^= x0
    # round 2
    x0 += x1
    x1 = (x1 << 15) | ((x1 >> 17) & 0x7FFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 3
    x0 += x1
    x1 = (x1 << 26) | ((x1 >> 6) & 0x3FFFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 4
    x0 += x1
    x1 = (x1 << 6) | ((x1 >> 26) & 0x3F)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)

    # inject key
    x0 += ks_1
    x1 += ks_2 + 1

    # round 5
    x0 += x1
    x1 = (x1 << 17) | ((x1 >> 15) & 0x1FFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 6
    x0 += x1
    x1 = (x1 << 29) | ((x1 >> 3) & 0x1FFFFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 7
    x0 += x1
    x1 = (x1 << 16) | ((x1 >> 16) & 0xFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 8
    x0 += x1
    x1 = (x1 << 24) | ((x1 >> 8) & 0xFFFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)

    # inject key
    # x0 += ks_2; x1 += (ks_0 + 2)
    #
    # x0 += x1; x1 = (x1 << 13) | (x1 >> 19); x1 ^= x0  # round 9
    # x0 += x1; x1 = (x1 << 15) | (x1 >> 17); x1 ^= x0  # round 10
    # x0 += x1; x1 = (x1 << 26) | (x1 >>  6); x1 ^= x0  # round 11
    # x0 += x1; x1 = (x1 <<  6) | (x1 >> 26); x1 ^= x0  # round 12

    # inject key
    x0 += ks_0
    x1 += ks_1 + 3

    return x0, x1


# @torch.jit.script
def __threefry64(
    x0: torch.Tensor, x1: torch.Tensor, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1]. This is the
    64-bit version.

    Parameters
    ----------
    x0 : torch.Tensor
        Upper bits of the to be encoded random sequence
    x1 : torch.Tensor
        Lower bits of the to be encoded random sequence
    seed : int
        The seed, i.e. key, for the threefry64 encryption

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2,
    3", Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
    p. 16, 2011
    """
    samples = len(x0)

    # set up key buffer
    ks_0 = torch.full((samples,), seed, dtype=torch.int64, device=x0.device)
    ks_1 = torch.full((samples,), seed, dtype=torch.int64, device=x1.device)
    ks_2 = torch.full((samples,), 2004413935125273122, dtype=torch.int64, device=x0.device)
    # ks_2 ^= ks_0
    # ks_2 ^= ks_1
    ks_2 = torch.bitwise_xor(torch.bitwise_xor(ks_2, ks_0), ks_1)

    # initialize output using the key
    x0 += ks_0
    x1 += ks_1

    # perform rounds
    # round 1
    x0 += x1
    x1 = (x1 << 16) | ((x1 >> 48) & 0xFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 2
    x0 += x1
    x1 = (x1 << 42) | ((x1 >> 22) & 0x3FFFFFFFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 3
    x0 += x1
    x1 = (x1 << 12) | ((x1 >> 52) & 0xFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 4
    x0 += x1
    x1 = (x1 << 31) | ((x1 >> 33) & 0x7FFFFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)

    # inject key
    x0 += ks_1
    x1 += ks_2 + 1

    # round 5
    x0 += x1
    x1 = (x1 << 16) | ((x1 >> 48) & 0xFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 6
    x0 += x1
    x1 = (x1 << 32) | ((x1 >> 32) & 0xFFFFFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 7
    x0 += x1
    x1 = (x1 << 24) | ((x1 >> 40) & 0xFFFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)
    # round 8
    x0 += x1
    x1 = (x1 << 21) | ((x1 >> 43) & 0x1FFFFF)
    # x1 ^= x0
    x1 = torch.bitwise_xor(x1, x0)

    # inject key
    # x0 += ks_2; x1 += (ks_0 + 2)
    #
    # x0 += x1; x1 = (x1 << 16) | (x1 >> 48); x1 ^= x0  # round 9
    # x0 += x1; x1 = (x1 << 42) | (x1 >> 22); x1 ^= x0  # round 10
    # x0 += x1; x1 = (x1 << 12) | (x1 >> 52); x1 ^= x0  # round 11
    # x0 += x1; x1 = (x1 << 31) | (x1 >> 33); x1 ^= x0  # round 12

    # inject key
    x0 += ks_0
    x1 += ks_1 + 3

    return x0, x1


# roll a global time-based seed
seed()
