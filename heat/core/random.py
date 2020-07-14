from __future__ import annotations

import numpy as np
import time
import torch
from typing import List, Optional, Tuple

from . import communication
from .communication import Communication
from . import devices
from .dndarray import DNDarray
from . import stride_tricks
from . import types

from .types import datatype


# introduce the global random state variables, will be correctly initialized at the end of file
__seed = None
__counter = None


# float conversion constants
__INT32_TO_FLOAT32 = 1.0 / 8388608.0
__INT64_TO_FLOAT64 = 1.0 / 9007199254740992.0
__KUNDU_INVERSE = 1.0 / 0.3807


def __counter_sequence(
    shape: Tuple[int, ...], dtype: datatype, split: int, comm: Communication
) -> Tuple[torch.tensor, torch.tensor, Tuple[int, ...], slice]:
    """
    Generates a sequence of numbers to be used as the "clear text" for the threefry encryption, i.e. the pseudo random
    number generator. Due to the fact that threefry always requires pairs of inputs, the input sequence may not just be
    a simple range including the global offset, but rather needs to be to independent vectors, one containing the range
    and the other having the interleaved high-bits counter in it.
    Returns the high-bits and low-bits vectors for the threefry encryption (``torch.tensor``), the shape ``x_0`` and ``x_1`` and
    the slice that needs to be applied to the resulting random number tensor.

    Parameters
    ----------
    shape : Tuple[int,...]
        The global shape of the random tensor to be generated.
    dtype : torch.dtype
        The data type of the elements to be generated. Needs to be either ``torch.int32`` or ``torch.int64``.
    split : int or None
        The split axis along which the random number tensor is split
    comm: Communication
        Handle to the nodes holding distributed parts or copies of this tensor.
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
        c_1 = __counter & max_count
    else:  # torch.int64
        c_0 = (__counter & (max_count << 64)) >> 64
        c_1 = __counter & max_count

    total_elements = np.prod(shape)
    if total_elements > 2 * max_count:
        raise ValueError("Shape is to big with {} elements".format(total_elements))

    if split is None:
        values = np.ceil(total_elements / 2)
        even_end = total_elements % 2 == 0
        lslice = slice(None) if even_end else slice(None, -1)
        start = c_1
        end = start + int(values)
        lshape = shape
    else:
        offset, lshape, _ = comm.chunk(shape, split)
        counts, displs, _ = comm.counts_displs_shape(shape, split)

        # Calculate number of local elements per process
        local_elements = [total_elements / shape[split] * counts[i] for i in range(size)]
        cum_elements = np.cumsum(local_elements)

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
    x_1 = torch.arange(*lrange, dtype=dtype)
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
    used_values = int(np.ceil(total_elements / 2))
    # increase counter but not over 128 bit
    tmp_counter += used_values
    __counter = tmp_counter & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # 128-bit mask

    return x_0, x_1, lshape, lslice


def get_state() -> Tuple[str, int, int, int, float]:
    """
    Return a tuple representing the internal state of the generator.
    The returned tuple has the following items:

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
    return (torch.log(-torch.log(1 - values ** 0.0775)) - 1.0821) * __KUNDU_INVERSE


def rand(
    *args: List[int],
    dtype: datatype = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None
) -> DNDarray:
    """
    Random values in a given shape.
    Create a :class:`~heat.core.dndarray.DNDarray`  of the given shape and populate it with random samples from a
    uniform distribution over [0, 1).

    Parameters
    ----------
    d0, d1, …, dn : List[int,...]
        The dimensions of the returned array, should all be positive. If no argument is given a single random samples is
        generated.
    dtype: datatype, optional
        The datatype of the returned values. Has to be one of
        [:class:`~heat.core.types.float32, :class:`~heat.core.types.float64`].
    split: int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device`  the array shall be allocated on, defaults to globally
        set default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    """
    # if args are not set, generate a single sample
    if not args:
        args = (1,)

    # ensure that the passed dimensions are positive integer-likes
    shape = tuple(int(ele) for ele in args)
    if not all(ele > 0 for ele in shape):
        raise ValueError("negative dimensions are not allowed")

    # make sure the remaining parameters are of proper type
    split = stride_tricks.sanitize_axis(shape, split)
    device = devices.sanitize_device(device)
    comm = communication.sanitize_comm(comm)

    # generate the random sequence
    if dtype == types.float32:
        x_0, x_1, lshape, lslice = __counter_sequence(shape, torch.int32, split, comm)
        x_0, x_1 = __threefry32(x_0, x_1)

        # combine the values into one tensor and convert them to floats
        values = __int32_to_float32(torch.stack([x_0, x_1], dim=1).flatten()[lslice]).reshape(
            lshape
        )
    elif dtype == types.float64:
        x_0, x_1, lshape, lslice = __counter_sequence(shape, torch.int64, split, comm)
        x_0, x_1 = __threefry64(x_0, x_1)

        # combine the values into one tensor and convert them to floats
        values = __int64_to_float64(torch.stack([x_0, x_1], dim=1).flatten()[lslice]).reshape(
            lshape
        )
    else:
        # Unsupported type
        raise ValueError("dtype is none of ht.float32 or ht.float64 but was {}".format(dtype))

    return DNDarray(values, shape, dtype, split, device, comm)


def randint(
    low: int,
    high: Optional[int] = None,
    size: Optional[Tuple[int]] = None,
    dtype: Optional[datatype] = None,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Random values in a given shape.
    Create a tensor of the given shape and populate it with random integer samples from a uniform distribution over
    ``[low, high)`` or ``[0, low)`` if high is not provided.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless ``high=None``, in which case this parameter is one
        above the highest such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior if ``high=None``).
    size : int or Tuple[int,...], optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then :math:`m \\times n \\times k`` samples are drawn.
        Default is None, in which case a single value is returned.
    dtype : datatype, optional
        Desired datatype of the result. Must be an integer type.
    split: int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device`  the array shall be allocated on, defaults to globally
        set default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
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
        size = (1,)
    shape = tuple(int(ele) for ele in size)
    if not all(ele > 0 for ele in shape):
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
    # generate the random sequence
    x_0, x_1, lshape, lslice = __counter_sequence(shape, dtype.torch_type(), split, comm)
    if torch_dtype is torch.int32:
        x_0, x_1 = __threefry32(x_0, x_1)
    else:  # torch.int64
        x_0, x_1 = __threefry64(x_0, x_1)

    # stack the resulting sequence and normalize to given range
    values = torch.stack([x_0, x_1], dim=1).flatten()[lslice].reshape(lshape)
    # ATTENTION: this is biased and known, bias-free rejection sampling is difficult to do in parallel
    values = (values.abs_() % span) + low

    return DNDarray(values, shape, dtype, split, device, comm)


def randn(
    *args: List[int],
    dtype: datatype = types.float32,
    split: Optional[int] = None,
    device: Optional[str] = None,
    comm: Optional[Communication] = None
) -> DNDarray:
    """
    Returns a tensor filled with random numbers from a standard normal distribution with zero mean and variance of one.

    Parameters
    ----------
    d0, d1, …, dn : List[int,...]
        The dimensions of the returned array, should be all positive.
    dtype: datatype, optional
        The datatype of the returned values. Has to be one of [:class:`~heat.core.types.float32, :class:`~heat.core.types.float64`].
    split: int, optional
        The axis along which the array is split and distributed, defaults to no distribution.
    device : str, optional
        Specifies the :class:`~heat.core.devices.Device`  the array shall be allocated on, defaults to globally
        set default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    Raises
    -------
    TypeError
        If one of d0 to dn is not an int.
    ValueError
        If one of d0 to dn is less or equal to 0.

    Examples
    --------
    >>> ht.randn(3)
    tensor([ 0.1921, -0.9635,  0.5047])
    >>> ht.randn(4, 4)
    tensor([[-1.1261,  0.5971,  0.2851,  0.9998],
            [-1.8548, -1.2574,  0.2391, -0.3302],
            [ 1.3365, -1.5212,  1.4159, -0.1671],
            [ 0.1260,  1.2126, -0.0804,  0.0907]])
    """
    # generate uniformly distributed random numbers first
    normal_tensor = rand(*args, dtype=dtype, split=split, device=device, comm=comm)
    # convert the the values to a normal distribution using the kundu transform
    normal_tensor._DNDarray__array = __kundu_transform(normal_tensor._DNDarray__array)

    return normal_tensor


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
    Set the internal state of the generator from a tuple.
    The tuple has the following items:

    1. The string ‘Threefry’,

    2. The Threefry key value, aka seed,

    3. The internal counter value,

    4. An integer ``has_gauss``, ignored (present for compatibility with numpy), optional and

    5. A float ``cached_gaussian``, ignored (present for compatibility with numpy), optional.

    Parameters
    ----------
    state : Tuple[str, int, int, int, float]

    Raises
    ------
    TypeError
        If and improper state is passed.
    ValueError
        If one of the items in the state tuple is of wrong type or value.
    """
    if not isinstance(state, tuple) or (len(state) != 3 and len(state) != 5):
        raise TypeError("state needs to be a three- or five-tuple")

    if state[0] != "Threefry":
        raise ValueError("algorithm must be 'Threefry'")

    global __seed, __counter
    __seed = int(state[1])
    __counter = int(state[2])


def __threefry32(x_0: torch.Tensor, x_1: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1]. Returns
    Two vectors with num_samples / 2 (rounded-up) pseudo random numbers. This is the 32-bit version.

    Parameters
    ----------
    x_0 : torch.Tensor
        Upper bits of the to be encoded random sequence
    x_1 : torch.Tensor
        Lower bits of the to be encoded random sequence

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2, 3"
    Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
    p. 16, 2011
    """
    samples = len(x_0)

    # Seed is > 32 bit
    seed_32 = __seed & 0x7FFFFFFF

    # set up key buffer
    ks_0 = torch.full((samples,), seed_32, dtype=torch.int32)
    ks_1 = torch.full((samples,), seed_32, dtype=torch.int32)
    ks_2 = torch.full((samples,), 466688986, dtype=torch.int32)
    ks_2 ^= ks_0
    ks_2 ^= ks_0

    # initialize output using the key
    x_0 += ks_0
    x_1 += ks_1

    # perform rounds
    # round 1
    x_0 += x_1
    x_1 = (x_1 << 13) | (x_1 >> 19)
    x_1 ^= x_0
    # round 2
    x_0 += x_1
    x_1 = (x_1 << 15) | (x_1 >> 17)
    x_1 ^= x_0
    # round 3
    x_0 += x_1
    x_1 = (x_1 << 26) | (x_1 >> 6)
    x_1 ^= x_0
    # round 4
    x_0 += x_1
    x_1 = (x_1 << 6) | (x_1 >> 26)
    x_1 ^= x_0

    # inject key
    x_0 += ks_1
    x_1 += ks_2 + 1

    # round 5
    x_0 += x_1
    x_1 = (x_1 << 17) | (x_1 >> 15)
    x_1 ^= x_0
    # round 6
    x_0 += x_1
    x_1 = (x_1 << 29) | (x_1 >> 3)
    x_1 ^= x_0
    # round 7
    x_0 += x_1
    x_1 = (x_1 << 16) | (x_1 >> 16)
    x_1 ^= x_0
    # round 8
    x_0 += x_1
    x_1 = (x_1 << 24) | (x_1 >> 8)
    x_1 ^= x_0

    # inject key
    # X_0 += ks_2; X_1 += (ks_0 + 2)
    #
    # X_0 += X_1; X_1 = (X_1 << 13) | (X_1 >> 19); X_1 ^= X_0  # round 9
    # X_0 += X_1; X_1 = (X_1 << 15) | (X_1 >> 17); X_1 ^= X_0  # round 10
    # X_0 += X_1; X_1 = (X_1 << 26) | (X_1 >>  6); X_1 ^= X_0  # round 11
    # X_0 += X_1; X_1 = (X_1 <<  6) | (X_1 >> 26); X_1 ^= X_0  # round 12

    # inject key
    x_0 += ks_0
    x_1 += ks_1 + 3

    return x_0, x_1


def __threefry64(x_0: torch.Tensor, x_1: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1].
    Returns two vectors with num_samples / 2 (rounded-up) pseudo random numbers. This is the 64-bit version.

    Parameters
    ----------
    x_0 : torch.Tensor
        Upper bits of the to be encoded random sequence
    x_1 : torch.Tensor
        Lower bits of the to be encoded random sequence

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2, 3"
    Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
    p. 16, 2011
    """
    samples = len(x_0)

    # set up key buffer
    ks_0 = torch.full((samples,), __seed, dtype=torch.int64)
    ks_1 = torch.full((samples,), __seed, dtype=torch.int64)
    ks_2 = torch.full((samples,), 2004413935125273122, dtype=torch.int64)
    ks_2 ^= ks_0
    ks_2 ^= ks_0

    # initialize output using the key
    x_0 += ks_0
    x_1 += ks_1

    # perform rounds
    # round 1
    x_0 += x_1
    x_1 = (x_1 << 16) | (x_1 >> 48)
    x_1 ^= x_0
    # round 2
    x_0 += x_1
    x_1 = (x_1 << 42) | (x_1 >> 22)
    x_1 ^= x_0
    # round 3
    x_0 += x_1
    x_1 = (x_1 << 12) | (x_1 >> 52)
    x_1 ^= x_0
    # round 4
    x_0 += x_1
    x_1 = (x_1 << 31) | (x_1 >> 33)
    x_1 ^= x_0

    # inject key
    x_0 += ks_1
    x_1 += ks_2 + 1

    # round 5
    x_0 += x_1
    x_1 = (x_1 << 16) | (x_1 >> 48)
    x_1 ^= x_0
    # round 6
    x_0 += x_1
    x_1 = (x_1 << 32) | (x_1 >> 32)
    x_1 ^= x_0
    # round 7
    x_0 += x_1
    x_1 = (x_1 << 24) | (x_1 >> 40)
    x_1 ^= x_0
    # round 8
    x_0 += x_1
    x_1 = (x_1 << 21) | (x_1 >> 43)
    x_1 ^= x_0

    # inject key
    # X_0 += ks_2; X_1 += (ks_0 + 2)
    #
    # X_0 += X_1; X_1 = (X_1 << 16) | (X_1 >> 48); X_1 ^= X_0  # round 9
    # X_0 += X_1; X_1 = (X_1 << 42) | (X_1 >> 22); X_1 ^= X_0  # round 10
    # X_0 += X_1; X_1 = (X_1 << 12) | (X_1 >> 52); X_1 ^= X_0  # round 11
    # X_0 += X_1; X_1 = (X_1 << 31) | (X_1 >> 33); X_1 ^= X_0  # round 12

    # inject key
    x_0 += ks_0
    x_1 += ks_1 + 3

    return x_0, x_1


# roll a global time-based seed
seed()
