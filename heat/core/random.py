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
    return 'Threefry', __seed, __counter, 0, 0.0


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
    return (values & 0x7fffff).type(torch.float32) / 8388608.0


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
    return (values & 0x1fffffffffffff).type(torch.float64) / 9007199254740992.0


def randn(*args, split=None, device=None, comm=None):
    """
    Returns a tensor filled with random numbers from a standard normal distribution with zero mean and variance of one.

    The shape of the tensor is defined by the varargs args.

    Parameters
    ----------
    d0, d1, …, dn : int, optional
        The dimensions of the returned array, should be all positive.

    Returns
    -------
    broadcast_shape : tuple of ints
        the broadcast shape

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
    # TODO: make me splitable
    # TODO: add device capabilities
    # check if all positional arguments are integers
    if not all(isinstance(_, int) for _ in args):
        raise TypeError('dimensions have to be integers')
    if not all(_ > 0 for _ in args):
        raise ValueError('negative dimension are not allowed')

    gshape = tuple(args) if args else(1,)
    split = stride_tricks.sanitize_axis(gshape, split)

    try:
        torch.randn(gshape)
    except RuntimeError as exception:
        # re-raise the exception to be consistent with numpy's exception interface
        raise ValueError(str(exception))

    # compose the local tensor
    device = devices.sanitize_device(device)
    comm = communication.sanitize_comm(comm)
    data = torch.randn(args, device=device.torch_device)

    return dndarray.DNDarray(data, gshape, types.canonical_heat_type(data.dtype), split, device, comm)


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
        raise TypeError('state needs to be a three- or five-tuple')

    if state[0] != 'Threefry':
        raise ValueError('algorithm must be "Threefry"')

    global __seed, __counter
    __seed = int(state[1])
    __counter = int(state[2])


def __threefry_32(num_samples):
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1]. This is the
    32-bit version.

    Parameters
    ----------
    num_samples : int
        Number of 32-bit pseudo random numbers to be generated.

    Returns
    -------
    random_numbers : torch.Tensor (int32)
        Vector with num_samples pseudo random numbers.

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2, 3"
        Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
        p. 16, 2011
    """
    samples = (num_samples + 1) // 2

    # set up X, i.e. output buffer
    X_0 = torch.arange(samples, dtype=torch.int32)
    X_1 = torch.arange(samples, dtype=torch.int32)
    X_0 //= torch.iinfo(torch.int32).max

    # set up key buffer
    ks_0 = torch.full((samples,), __seed, dtype=torch.int32)
    ks_1 = torch.full((samples,), __seed, dtype=torch.int32)
    ks_2 = torch.full((samples,), 466688986, dtype=torch.int32)
    ks_2 ^= ks_0
    ks_2 ^= ks_0

    # initialize output using the key
    X_0 += ks_0
    X_1 += ks_1

    # perform rounds
    X_0 += X_1; X_1 = (X_1 << 13) | (X_1 >> 19); X_1 ^= X_0  # round 1
    X_0 += X_1; X_1 = (X_1 << 15) | (X_1 >> 17); X_1 ^= X_0  # round 2
    X_0 += X_1; X_1 = (X_1 << 26) | (X_1 >>  6); X_1 ^= X_0  # round 3
    X_0 += X_1; X_1 = (X_1 << 6)  | (X_1 >> 26); X_1 ^= X_0  # round 4

    # inject key
    X_0 += ks_1; X_1 += (ks_2 + 1)

    X_0 += X_1; X_1 = (X_1 << 17) | (X_1 >> 15); X_1 ^= X_0  # round 5
    X_0 += X_1; X_1 = (X_1 << 29) | (X_1 >>  3); X_1 ^= X_0  # round 6
    X_0 += X_1; X_1 = (X_1 << 16) | (X_1 >> 16); X_1 ^= X_0  # round 7
    X_0 += X_1; X_1 = (X_1 << 24) | (X_1 >>  8); X_1 ^= X_0  # round 8

    # inject key
    X_0 += ks_2; X_1 += (ks_0 + 2)

    X_0 += X_1; X_1 = (X_1 << 13) | (X_1 >> 19); X_1 ^= X_0  # round 9
    X_0 += X_1; X_1 = (X_1 << 15) | (X_1 >> 17); X_1 ^= X_0  # round 10
    X_0 += X_1; X_1 = (X_1 << 26) | (X_1 >>  6); X_1 ^= X_0  # round 11
    X_0 += X_1; X_1 = (X_1 <<  6) | (X_1 >> 26); X_1 ^= X_0  # round 12

    # inject key
    X_0 += ks_0; X_1 += (ks_1 + 3)

    return X_0, X_1


def __threefry64(num_samples):
    """
    Counter-based pseudo random number generator. Based on a 12-round Threefry "encryption" algorithm [1]. This is the
    64-bit version.

    Parameters
    ----------
    num_samples : int
        Number of 64-bit pseudo random numbers to be generated.

    Returns
    -------
    random_numbers : torch.Tensor (int64)
        Vector with num_samples pseudo random numbers.

    References
    ----------
    [1] Salmon, John K., Moraes, Mark A., Dror, Ron O. and Shaw, David E., "Parallel random numbers: as easy as 1, 2, 3"
        Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis,
        p. 16, 2011
    """
    samples = (num_samples + 1) // 2

    # set up X, i.e. output buffer
    X_0 = torch.arange(samples, dtype=torch.int64)
    X_1 = torch.arange(samples, dtype=torch.int64)
    X_0 //= torch.iinfo(torch.int64).max

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
    X_0 += X_1; X_1 = (X_1 << 16) | (X_1 >> 48); X_1 ^= X_0  # round 1
    X_0 += X_1; X_1 = (X_1 << 42) | (X_1 >> 22); X_1 ^= X_0  # round 2
    X_0 += X_1; X_1 = (X_1 << 12) | (X_1 >> 52); X_1 ^= X_0  # round 3
    X_0 += X_1; X_1 = (X_1 << 31) | (X_1 >> 33); X_1 ^= X_0  # round 4
    # inject key
    X_0 += ks_1; X_1 += (ks_2 + 1)

    X_0 += X_1; X_1 = (X_1 << 16) | (X_1 >> 48); X_1 ^= X_0  # round 5
    X_0 += X_1; X_1 = (X_1 << 32) | (X_1 >> 32); X_1 ^= X_0  # round 6
    X_0 += X_1; X_1 = (X_1 << 24) | (X_1 >> 40); X_1 ^= X_0  # round 7
    X_0 += X_1; X_1 = (X_1 << 21) | (X_1 >> 43); X_1 ^= X_0  # round 8

    # inject key
    X_0 += ks_2; X_1 += (ks_0 + 2)

    X_0 += X_1; X_1 = (X_1 << 16) | (X_1 >> 48); X_1 ^= X_0  # round 9
    X_0 += X_1; X_1 = (X_1 << 42) | (X_1 >> 22); X_1 ^= X_0  # round 10
    X_0 += X_1; X_1 = (X_1 << 12) | (X_1 >> 52); X_1 ^= X_0  # round 11
    X_0 += X_1; X_1 = (X_1 << 31) | (X_1 >> 33); X_1 ^= X_0  # round 12

    # inject key
    X_0 += ks_0; X_1 += (ks_1 + 3)

    return X_0, X_1


def uniform(low=0.0, high=1.0, size=None, device=None, comm=None):
    # TODO: comment me
    # TODO: test me
    # TODO: make me splitable
    # TODO: add device capabilities
    if size is None:
        size = (1,)

    device = devices.sanitize_device(device)
    comm = communication.sanitize_comm(comm)
    data = torch.rand(*size, device=device.torch_device) * (high - low) + low

    return dndarray.DNDarray(data, size, types.float32, None, device, comm)


# roll a global time-based seed
seed()
