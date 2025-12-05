Module heat.core.random
=======================
Provides parallel random number generators (pRNG)
Two options are aviable:

1.  Batchparallel RNG (default):
    This is a simple, fast, and (weakly) reproducible random number generator (RNG) that is based on the idea of a global seed
    that results in process-local seeds for each MPI-process; then, on each MPI-process torch's RNG is used with these process-local seeds.
    To reproduce results, the global seed needs to be set to the same value and the number of MPI-processes needs to be the same (=weak reproducibility).

2.  Threefry RNG:
    This is a fully reproducible parallel RNG that is based on the Threefry encryption algorithm.
    It is slower than the batchparallel RNG and limited to generating random DNDarrays with less than maxint32 many entries.
    However, unlike batchparallel RNG it ensures full reproducibility even for different numbers of MPI-processes.

Functions
---------

`get_state() ‑> Tuple[str, int, int, int, float]`
:   Return a tuple representing the internal state of the generator. The returned tuple has the following items:

    1. The string 'Batchparallel' or ‘Threefry’, describing the type of random number generator,

    2. The seed. For batchparallel RNG this refers to the global seed. For Threefry RNG the seed is the key value,

    3. The local seed (for batchparallel RNG), or the internal counter value (for Threefry RNG), respectively,

    4. An integer has_gauss, always set to 0 (present for compatibility with numpy), and

    5. A float cached_gaussian, always set to 0.0 (present for compatibility with numpy).

`normal(mean: Union[float, DNDarray] = 0.0, std: Union[float, DNDarray] = 1.0, shape: Optional[Tuple[int, ...]] = None, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Returns an array filled with random numbers from a normal distribution whose mean and standard deviation are given.
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
    >>> ht.random.normal(ht.array([-1, 2]), ht.array([0.5, 2]), (2,))
    DNDarray([-1.4669,  1.6596], dtype=ht.float64, device=cpu:0, split=None)

`permutation(x: Union[int, DNDarray], **kwargs) ‑> heat.core.dndarray.DNDarray`
:   Randomly permute a sequence, or return a permuted range. If ``x`` is a multi-dimensional array, it is only shuffled
    along its first index.

    Parameters
    ----------
    x : int or DNDarray
        If ``x`` is an integer, call :func:`heat.random.randperm <heat.core.random.randperm>`. If ``x`` is an array,
        make a copy and shuffle the elements randomly.

    kwargs : dict, optional
        Additional keyword arguments passed to :func:`heat.random.randperm <heat.core.random.randperm>` if ``x`` is an integer.

    See Also
    --------
    :func:`heat.random.randperm <heat.core.random.randperm>` for randomly permuted ranges.

    Examples
    --------
    >>> ht.random.permutation(10)
    DNDarray([9, 1, 5, 4, 8, 2, 7, 6, 3, 0], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.random.permutation(ht.array([1, 4, 9, 12, 15]))
    DNDarray([ 9,  1, 12,  4, 15], dtype=ht.int64, device=cpu:0, split=None)
    >>> arr = ht.arange(9).reshape((3, 3))
    >>> ht.random.permutation(arr)
    DNDarray([[3, 4, 5],
              [6, 7, 8],
              [0, 1, 2]], dtype=ht.int32, device=cpu:0, split=None)

    Notes
    -----
    This routine makes usage of torch's RNG to generate an array of the permuted indices of axis 0.
    Thus, the array containing these indices needs to fit into the memory of a single MPI-process.

`rand(*d: int, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[Device] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Random values in a given shape. Create a :class:`~heat.core.dndarray.DNDarray` of the given shape and populate it
    with random samples from a uniform distribution over :math:`[0, 1)`.

    Parameters
    ----------
    *d : int, optional
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

`randint(low: int, high: Optional[int] = None, size: Optional[Union[int, Tuple[int]]] = None, dtype: Optional[Type[datatype]] = heat.core.types.int32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Random values in a given shape. Create a tensor of the given shape and populate it with random integer samples from
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
    ------
    TypeError
        If one of low or high is not an int.
    ValueError
        If low >= high, dimensions are negative or the passed datatype is not an integer.

    Examples
    --------
    >>> ht.randint(3)
    DNDarray([4, 101, 16], dtype=ht.int32, device=cpu:0, split=None)

`randn(*d: int, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Returns a tensor filled with random numbers from a standard normal distribution with zero mean and variance of one.

    Parameters
    ----------
    *d : int, optional
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
    ------
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

`random(shape: Optional[Tuple[int]] = None, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None)`
:   Populates a :class:`~heat.core.dndarray.DNDarray` of the given shape with random samples from a continuous uniform
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

`random_integer(low: int, high: Optional[int] = None, size: Optional[Union[int, Tuple[int]]] = None, dtype: Optional[Type[datatype]] = heat.core.types.int32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Alias for :func:`heat.random.randint <heat.core.random.randint>`.

`random_sample(shape: Optional[Tuple[int]] = None, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None)`
:   Alias for :func:`heat.random.random <heat.core.random.random>`.

`randperm(n: int, dtype: Type[datatype] = heat.core.types.int64, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Returns a random permutation of integers from :math:`0` to :math:`n - 1`.

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
    ------
    TypeError
        If ``n`` is not an integer.

    Examples
    --------
    >>> ht.random.randperm(4)
    DNDarray([2, 3, 1, 0], dtype=ht.int64, device=cpu:0, split=None)

    Notes
    -----
    This routine makes usage of torch's RNG. Thus, the resulting array needs to fit into the memory of a single MPI-process.

`ranf(shape: Optional[Tuple[int]] = None, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None)`
:   Alias for :func:`heat.random.random <heat.core.random.random>`.

`sample(shape: Optional[Tuple[int]] = None, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None)`
:   Alias for :func:`heat.random.random <heat.core.random.random>`.

`seed(seed: Optional[int] = None)`
:   Seed the random number generator.

    Parameters
    ----------
    seed : int, optional
        Value to seed the algorithm with, if not set a time-based seed is generated.

`set_state(state: Tuple[str, int, int, int, int, float])`
:   Set the internal state of the generator from a tuple. The tuple has the following items:

    1. The string 'Batchparallel' or ‘Threefry’, describing the type of random number generator,

    2. The seed. For batchparallel RNG this refers to the global seed. For Threefry RNG the seed is the key value,

    3. The local seed (for batchparallel RNG), or the internal counter value (for Threefry RNG), respectively,
       (For batchparallel RNG, this value is ignored if a global seed is provided. If you want to prescribe a process-local
       seed manually, you need to set the global seed to None.)

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

`standard_normal(shape: Optional[Tuple[int, ...]] = None, dtype: Type[datatype] = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Returns an array filled with random numbers from a standard normal distribution with zero mean and variance of one.

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
