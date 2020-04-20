import numpy as np
import time
import torch

from . import communication
from . import devices
from . import dndarray
from . import stride_tricks
from . import types


# introduce the global random state variables, will be correctly initialized at the end of file
__seed = None
__counter = None


# float conversion constants
__INT32_TO_FLOAT32 = 1.0 / 8388608.0
__INT64_TO_FLOAT64 = 1.0 / 9007199254740992.0
__KUNDU_INVERSE = 1.0 / 0.3807


def __counter_sequence(shape, dtype, split, device, comm):
    """
    Generates a sequence of numbers to be used as the "clear text" for the threefry encryption, i.e. the pseudo random
    number generator. Due to the fact that threefry always requires pairs of inputs, the input sequence may not just be
    a simple range including the global offset, but rather needs to be to independent vectors, one containing the range
    and the other having the interleaved high-bits counter in it.

    Parameters
    ----------
    shape : tuple of ints
        The global shape of the random tensor to be generated.
    dtype : torch.dtype
        The data type of the elements to be generated. Needs to be either torch.int32 or torch.int64.
    split : int or None
        The split axis along which the random number tensor is split
    device : 'str'
        Specifies the device the tensor shall be allocated on.
    comm: ht.Communication
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
            # No overlap with previous processes
            if elements == int(elements):
                # Even number of elements
                end = int(elements)
            else:
                # Odd number of elements
                end = int(elements) + 1
                lslice = slice(None, -1)
        else:
            # Overlap with previous processes
            if elements == int(elements):
                # Even number of elements
                end = int(elements) + 1
                lslice = slice(1, -1)
            else:
                # Odd number of elements
                end = int(elements) + 1
                lslice = slice(1, None)
        start = int(start)
        end += start

    # Check x_1 for overflow
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

    # Detect if x_0 needs to be increased for current values
    if end > max_count:
        if start > max_count:
            # x_0 changed in previous process, increase all values
            x_0 += 1
        else:
            # x_0 changes after reaching the overflow in this process
            x_0[-(end - max_count - 1) :] += 1

    # Correctly increase the counter variable
    used_values = int(np.ceil(total_elements / 2))
    # Increase counter but not over 128 bit
    tmp_counter += used_values
    __counter = tmp_counter & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # 128bit mask

    return x_0, x_1, lshape, lslice


def get_state():
    """
    Return a tuple representing the internal state of the generator.

    Returns
    -------
    out : tuple(str, int, int, int, float)
        The returned tuple has the following items:
            1. the string ‘Threefry’,
            2. the Threefry key value, aka seed,
            3. the internal counter value,
            4. an integer has_gauss, always set to 0 (present for compatibility with numpy) and
            5. a float cached_gaussian, always set to 0.0 (present for compatibility with numpy).
    """
    return "Threefry", __seed, __counter, 0, 0.0


def __int32_to_float32(values):
    """
    Converts a tensor of 32-bit (random) numbers to matching single-precision floating point numbers (equally 32-bit) in
    the bounded interval [0.0, 1.0). Extracts the 23 least-significant bits of the integers (0x7fffff) and sets them to
    be the mantissa of the floating point number. Interval is bound by dividing by 2^23 = 8388608.0.

    Parameters
    ----------
    values : torch.Tensor (int32)
        Values to be converted to floating points numbers in interval [0.0, 1.0).

    Returns
    -------
    floats : torch.Tensor (float32)
        Corresponding single-precision floating point numbers.
    """
    return (values & 0x7FFFFF).type(torch.float32) * __INT32_TO_FLOAT32


def __int64_to_float64(values):
    """
    Converts a tensor of 64-bit (random) numbers to matching double-precision floating point numbers (equally 64-bit) in
    the bounded interval [0.0, 1.0). Extracts the 53 least-significant bits of the integers (0x1fffffffffffff) and sets
    them to be the mantissa of the floating point number. Interval is bound by dividing by 2^53 = 9007199254740992.0.

    Parameters
    ----------
    values : torch.Tensor (int64)
        Values to be converted to floating points numbers in interval [0.0, 1.0).

    Returns
    -------
    floats : torch.Tensor (float64)
        Corresponding single-precision floating point numbers.
    """
    return (values & 0x1FFFFFFFFFFFFF).type(torch.float64) * __INT64_TO_FLOAT64


def __kundu_transform(values):
    """
    Transforms uniformly distributed floating point random values in the interval [0.0, 1.0) into normal distributed
    floating point random values with mean 0.0 and standard deviation 1.0. The algorithm makes use of the generalized
    exponential distribution transformation [1].

    Parameters
    ----------
    values : torch.Tensor
        A tensor containing uniformly distributed floating point values in the interval [0.0, 1.0).

    Returns
    -------
    normal_values : torch.Tensor
        A tensor containing the equivalent normally distributed floating point values with mean of 0.0 and standard
        deviation of 1.0.

    References
    ----------
    [1] Boiroju, N. K. and Reddy, K. M., "Generation of Standard Normal Random Numbers", Interstat, vol 5., 2012.
    """
    return (torch.log(-torch.log(1 - values ** 0.0775)) - 1.0821) * __KUNDU_INVERSE


def rand(*args, dtype=types.float32, split=None, device=None, comm=None):
    """
    Random values in a given shape.

    Create a tensor of the given shape and populate it with random samples from a uniform distribution over [0, 1).

    Parameters
    ----------
    d0, d1, …, dn : int, optional
        The dimensions of the returned array, should all be positive. If no argument is given a single random samples is
        generated.
    dtype: ht.types, optional
        The datatype of the returned values. Has to be one of [ht.float32, ht.float64]. Default is ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.dndarray, shape (d0, d1, ..., dn)
        The uniformly distributed [0.0, 1.0)-bound random values.
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
        x_0, x_1, lshape, lslice = __counter_sequence(shape, torch.int32, split, device, comm)
        x_0, x_1 = __threefry32(x_0, x_1)

        # combine the values into one tensor and convert them to floats
        values = __int32_to_float32(torch.stack([x_0, x_1], dim=1).flatten()[lslice]).reshape(
            lshape
        )
    elif dtype == types.float64:
        x_0, x_1, lshape, lslice = __counter_sequence(shape, torch.int64, split, device, comm)
        x_0, x_1 = __threefry64(x_0, x_1)

        # combine the values into one tensor and convert them to floats
        values = __int64_to_float64(torch.stack([x_0, x_1], dim=1).flatten()[lslice]).reshape(
            lshape
        )
    else:
        # Unsupported type
        raise ValueError("dtype is none of ht.float32 or ht.float64 but was {}".format(dtype))

    return dndarray.DNDarray(values, shape, dtype, split, device, comm)


def randint(low, high=None, size=None, dtype=None, split=None, device=None, comm=None):
    """
    Random values in a given shape.

    Create a tensor of the given shape and populate it with random integer samples from a uniform distribution over
    [low, high) or [0, low) if high is not provided.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless high=None, in which case this parameter is one
        above the highest such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior if high=None).
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in
        which case a single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. Must be an integer type. Defaults to ht.int64.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.dndarray, shape (d0, d1, ..., dn)
        The uniformly distributed [0.0, 1.0)-bound random values.
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
    x_0, x_1, lshape, lslice = __counter_sequence(shape, dtype.torch_type(), split, device, comm)
    if torch_dtype is torch.int32:
        x_0, x_1 = __threefry32(x_0, x_1)
    else:  # torch.int64
        x_0, x_1 = __threefry64(x_0, x_1)

    # stack the resulting sequence and normalize to given range
    values = torch.stack([x_0, x_1], dim=1).flatten()[lslice].reshape(lshape)
    # ATTENTION: this is biased and known, bias-free rejection sampling is difficult to do in parallel
    values = (values.abs_() % span) + low

    return dndarray.DNDarray(values, shape, dtype, split, device, comm)


def randn(*args, dtype=types.float32, split=None, device=None, comm=None):
    """
    Returns a tensor filled with random numbers from a standard normal distribution with zero mean and variance of one.

    Parameters
    ----------
    d0, d1, …, dn : int, optional
        The dimensions of the returned array, should be all positive.
    dtype: ht.types, optional
        The datatype of the returned values. Has to be one of [ht.float32, ht.float64]. Default is ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.dndarray, shape (d0, d1, ..., dn)
        The normal distributed random values.

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


def seed(seed=None):
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


def set_state(state):
    """
    Set the internal state of the generator from a tuple.

    Parameters
    ----------
    state : tuple(str, int, int, int, float)
        The returned tuple has the following items:
            1. the string ‘Threefry’,
            2. the Threefry key value, aka seed,
            3. the internal counter value,
            4. an integer has_gauss, ignored (present for compatibility with numpy), optional and
            5. a float cached_gaussian, ignored (present for compatibility with numpy), optional.

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


def __threefry32(X_0, X_1):
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1]. This is the
    32-bit version.

    Parameters
    ----------
    X_0 : torch.Tensor
        Upper bits of the to be encoded random sequence
    X_1 : torch.Tensor
        Lower bits of the to be encoded random sequence

    Returns
    -------
    random_numbers : tuple(torch.Tensor (int32))
        Two vectors with num_samples / 2 (rounded-up) pseudo random numbers.

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2, 3"
        Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
        p. 16, 2011
    """
    samples = len(X_0)

    # Seed is > 32 bit
    seed_32 = __seed & 0x7FFFFFFF

    # set up key buffer
    ks_0 = torch.full((samples,), seed_32, dtype=torch.int32)
    ks_1 = torch.full((samples,), seed_32, dtype=torch.int32)
    ks_2 = torch.full((samples,), 466688986, dtype=torch.int32)
    ks_2 ^= ks_0
    ks_2 ^= ks_0

    # initialize output using the key
    X_0 += ks_0
    X_1 += ks_1

    # perform rounds
    # round 1
    X_0 += X_1
    X_1 = (X_1 << 13) | (X_1 >> 19)
    X_1 ^= X_0
    # round 2
    X_0 += X_1
    X_1 = (X_1 << 15) | (X_1 >> 17)
    X_1 ^= X_0
    # round 3
    X_0 += X_1
    X_1 = (X_1 << 26) | (X_1 >> 6)
    X_1 ^= X_0
    # round 4
    X_0 += X_1
    X_1 = (X_1 << 6) | (X_1 >> 26)
    X_1 ^= X_0

    # inject key
    X_0 += ks_1
    X_1 += ks_2 + 1

    # round 5
    X_0 += X_1
    X_1 = (X_1 << 17) | (X_1 >> 15)
    X_1 ^= X_0
    # round 6
    X_0 += X_1
    X_1 = (X_1 << 29) | (X_1 >> 3)
    X_1 ^= X_0
    # round 7
    X_0 += X_1
    X_1 = (X_1 << 16) | (X_1 >> 16)
    X_1 ^= X_0
    # round 8
    X_0 += X_1
    X_1 = (X_1 << 24) | (X_1 >> 8)
    X_1 ^= X_0

    # inject key
    # X_0 += ks_2; X_1 += (ks_0 + 2)
    #
    # X_0 += X_1; X_1 = (X_1 << 13) | (X_1 >> 19); X_1 ^= X_0  # round 9
    # X_0 += X_1; X_1 = (X_1 << 15) | (X_1 >> 17); X_1 ^= X_0  # round 10
    # X_0 += X_1; X_1 = (X_1 << 26) | (X_1 >>  6); X_1 ^= X_0  # round 11
    # X_0 += X_1; X_1 = (X_1 <<  6) | (X_1 >> 26); X_1 ^= X_0  # round 12

    # inject key
    X_0 += ks_0
    X_1 += ks_1 + 3

    return X_0, X_1


def __threefry64(X_0, X_1):
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1]. This is the
    64-bit version.

    Parameters
    ----------
    X_0 : torch.Tensor
        Upper bits of the to be encoded random sequence
    X_1 : torch.Tensor
        Lower bits of the to be encoded random sequence

    Returns
    -------
    random_numbers : tuple(torch.Tensor (int64))
        Two vectors with num_samples / 2 (rounded-up) pseudo random numbers.

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2, 3"
        Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
        p. 16, 2011
    """
    samples = len(X_0)

    # set up key buffer
    ks_0 = torch.full((samples,), __seed, dtype=torch.int64)
    ks_1 = torch.full((samples,), __seed, dtype=torch.int64)
    ks_2 = torch.full((samples,), 2004413935125273122, dtype=torch.int64)
    ks_2 ^= ks_0
    ks_2 ^= ks_0

    # initialize output using the key
    X_0 += ks_0
    X_1 += ks_1

    # perform rounds
    # round 1
    X_0 += X_1
    X_1 = (X_1 << 16) | (X_1 >> 48)
    X_1 ^= X_0
    # round 2
    X_0 += X_1
    X_1 = (X_1 << 42) | (X_1 >> 22)
    X_1 ^= X_0
    # round 3
    X_0 += X_1
    X_1 = (X_1 << 12) | (X_1 >> 52)
    X_1 ^= X_0
    # round 4
    X_0 += X_1
    X_1 = (X_1 << 31) | (X_1 >> 33)
    X_1 ^= X_0

    # inject key
    X_0 += ks_1
    X_1 += ks_2 + 1

    # round 5
    X_0 += X_1
    X_1 = (X_1 << 16) | (X_1 >> 48)
    X_1 ^= X_0
    # round 6
    X_0 += X_1
    X_1 = (X_1 << 32) | (X_1 >> 32)
    X_1 ^= X_0
    # round 7
    X_0 += X_1
    X_1 = (X_1 << 24) | (X_1 >> 40)
    X_1 ^= X_0
    # round 8
    X_0 += X_1
    X_1 = (X_1 << 21) | (X_1 >> 43)
    X_1 ^= X_0

    # inject key
    # X_0 += ks_2; X_1 += (ks_0 + 2)
    #
    # X_0 += X_1; X_1 = (X_1 << 16) | (X_1 >> 48); X_1 ^= X_0  # round 9
    # X_0 += X_1; X_1 = (X_1 << 42) | (X_1 >> 22); X_1 ^= X_0  # round 10
    # X_0 += X_1; X_1 = (X_1 << 12) | (X_1 >> 52); X_1 ^= X_0  # round 11
    # X_0 += X_1; X_1 = (X_1 << 31) | (X_1 >> 33); X_1 ^= X_0  # round 12

    # inject key
    X_0 += ks_0
    X_1 += ks_1 + 3

    return X_0, X_1


# roll a global time-based seed
seed()
