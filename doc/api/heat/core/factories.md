Module heat.core.factories
==========================
Provides high-level DNDarray initialization functions

Functions
---------

`arange(*args: int | float, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, device: str | heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)`` (in other words, the interval including `start`
    but excluding `stop`). For integer arguments the function is equivalent to the Python built-in `range
    <http://docs.python.org/lib/built-in-funcs.html>`_ function, but returns a array rather than a list.
    When using a non-integer step, such as 0.1, the results may be inconsistent due to being subject to numerical
    rounding. In the cases the usage of :func:`linspace` is recommended.
    For floating point arguments, the length of the result is :math:`\lceil(stop-start)/step\rceil`.
    Again, due to floating point rounding, this rule may result in the last element of `out` being greater than `stop`
    by machine epsilon.

    Parameters
    ----------
    *args : int or float, optional
        Positional arguments defining the interval. Can be:
        - A single argument: interpreted as `stop`, with `start=0` and `step=1`.
        - Two arguments: interpreted as `start` and `stop`, with `step=1`.
        - Three arguments: interpreted as `start`, `stop`, and `step`.
        The function raises a `TypeError` if more than three arguments are provided.
    dtype : datatype, optional
        The type of the output array.  If `dtype` is not given, it is automatically inferred from the other input
        arguments.
    split: int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str, optional
        Specifies the device the array shall be allocated on, defaults to globally set default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    See Also
    --------
    :func:`linspace` : Evenly spaced numbers with careful handling of endpoints.

    Examples
    --------
    >>> ht.arange(3)
    DNDarray([0, 1, 2], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.arange(3.0)
    DNDarray([0., 1., 2.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.arange(3, 7)
    DNDarray([3, 4, 5, 6], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.arange(3, 7, 2)
    DNDarray([3, 5], dtype=ht.int32, device=cpu:0, split=None)

`array(obj: Iterable, dtype: Type[heat.core.types.datatype] | None = None, copy: bool | None = None, ndmin: int = 0, order: str = 'C', split: int | None = None, is_split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None) ‑> heat.core.dndarray.DNDarray`
:   Create a :class:`~heat.core.dndarray.DNDarray`.

    Parameters
    ----------
    obj : array_like
        A tensor or array, any object exposing the array interface, an object whose ``__array__`` method returns an
        array, or any (nested) sequence.
    dtype : datatype, optional
        The desired data-type for the array. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the :func:`~heat.core.dndarray.astype` method.
    copy : bool, optional
        If ``True``, the input object is copied.
        If ``False``, input which supports the buffer protocol is never copied.
        If ``None`` (default), the function reuses the existing memory buffer if possible, and copies otherwise.
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have. Ones will, if needed, be
        attached to the shape if ``ndim > 0`` and prefaced in case of ``ndim < 0`` to meet the requirement.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).
    split : int or None, optional
        The axis along which the passed array content ``obj`` is split and distributed in memory. Mutually exclusive
        with ``is_split``.
    is_split : int or None, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. Useful for
        interfacing with other distributed-memory code. The shape of the global array is automatically inferred.
        Mutually exclusive with ``split``.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on (i.e. globally set default
        device).
    comm : Communication, optional
        Handle to the nodes holding distributed array chunks.

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.
    ValueError
        If ``copy`` is False but a copy is necessary to satisfy other requirements (e.g. different dtype, device, etc.).
    TypeError
        If the input object cannot be converted to a torch.Tensor, hence it cannot be converted to a :class:`~heat.core.dndarray.DNDarray`.

    Examples
    --------
    >>> ht.array([1, 2, 3])
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.array([1, 2, 3.0])
    DNDarray([1., 2., 3.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.array([[1, 2], [3, 4]])
    DNDarray([[1, 2],
              [3, 4]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.array([1, 2, 3], ndmin=2)
    DNDarray([[1],
              [2],
              [3]], dtype=ht.int64, device=cpu:0, split=None)
    >>> ht.array([1, 2, 3], dtype=float)
    DNDarray([1., 2., 3.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.array([1, 2, 3, 4], split=0)
    DNDarray([1, 2, 3, 4], dtype=ht.int64, device=cpu:0, split=0)
    >>> if ht.MPI_WORLD.rank == 0
    >>>     a = ht.array([1, 2], is_split=0)
    >>> else:
    >>>     a = ht.array([3, 4], is_split=0)
    >>> a
    DNDarray([1, 2, 3, 4], dtype=ht.int64, device=cpu:0, split=0)
    >>> a = np.arange(2 * 3).reshape(2, 3)
    >>> a
    array([[ 0,  1,  2],
           [ 3,  4,  5]])
    >>> a.strides
    (24, 8)
    >>> b = ht.array(a)
    >>> b
    DNDarray([[0, 1, 2],
              [3, 4, 5]], dtype=ht.int64, device=cpu:0, split=None)
    >>> b.strides
    (24, 8)
    >>> b.larray.untyped_storage()
     0
     1
     2
     3
     4
     5
    [torch.LongStorage of size 6]
    >>> c = ht.array(a, order="F")
    >>> c
    DNDarray([[0, 1, 2],
              [3, 4, 5]], dtype=ht.int64, device=cpu:0, split=None)
    >>> c.strides
    (8, 16)
    >>> c.larray.untyped_storage()
     0
     3
     1
     4
     2
     5
    [torch.LongStorage of size 6]
    >>> a = np.arange(4 * 3).reshape(4, 3)
    >>> a.strides
    (24, 8)
    >>> b = ht.array(a, order="F", split=0)
    >>> b
    DNDarray([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]], dtype=ht.int64, device=cpu:0, split=0)
    >>> b.strides
    [0/2] (8, 16)
    [1/2] (8, 16)
    >>> b.larray.untyped_storage()
    [0/2] 0
          3
          1
          4
          2
          5
         [torch.LongStorage of size 6]
    [1/2] 6
          9
          7
          10
          8
          11
         [torch.LongStorage of size 6]

`asarray(obj: Iterable, dtype: Type[heat.core.types.datatype] | None = None, copy: bool | None = None, order: str = 'C', is_split: bool | None = None, device: str | heat.core.devices.Device | None = None) ‑> heat.core.dndarray.DNDarray`
:   Convert ``obj`` to a DNDarray. If ``obj`` is a `DNDarray` or `Tensor` with the same `dtype` and `device` or if the
    data is an `ndarray` of the corresponding ``dtype`` and the ``device`` is the CPU, no copy will be performed.

    Parameters
    ----------
    obj : iterable
        Input data, in any form that can be converted to an array. This includes e.g. lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ndarrays.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.
    copy : bool, optional
        If ``True``, then the object is copied.  If ``False``, the object is not copied and a ``ValueError`` is
        raised in the case a copy would be necessary. If ``None``, a copy will only be made if `obj` is a nested
        sequence or if a copy is needed to satisfy any of the other requirements, e.g. ``dtype``.
    order: str, optional
        Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.
    is_split : None or int, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all MPI processes. Useful for
        interfacing with other HPC code. The shape of the global tensor is automatically inferred.
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on. By default, it is inferred from the input data.

    Examples
    --------
    >>> a = [1, 2]
    >>> ht.asarray(a)
    DNDarray([1, 2], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = np.array([1, 2, 3])
    >>> n = ht.asarray(a)
    >>> n
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> n[0] = 0
    >>> a
    DNDarray([0, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = torch.tensor([1, 2, 3])
    >>> t = ht.asarray(a)
    >>> t
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> t[0] = 0
    >>> a
    DNDarray([0, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = ht.array([1, 2, 3, 4], dtype=ht.float32)
    >>> ht.asarray(a, dtype=ht.float32) is a
    True
    >>> ht.asarray(a, dtype=ht.float64) is a
    False

`empty(shape: int | Sequence[int], dtype: Type[heat.core.types.datatype] = heat.core.types.float32, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Returns a new uninitialized :class:`~heat.core.dndarray.DNDarray` of given shape and data type. May be allocated
    split up across multiple nodes along the specified axis.

    Parameters
    ----------
    shape : int or Sequence[int,...]
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : datatype
        The desired HeAT data type for the array.
    split: int, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device`. the array shall be allocated on, defaults to globally set
        default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> ht.empty(3)
    DNDarray([0., 0., 0.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.empty(3, dtype=ht.int)
    DNDarray([59140784,        0, 59136816], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.empty(
    ...     (
    ...         2,
    ...         3,
    ...     )
    ... )
    DNDarray([[-1.7206e-10,  4.5905e-41, -1.7206e-10],
              [ 4.5905e-41,  4.4842e-44,  0.0000e+00]], dtype=ht.float32, device=cpu:0, split=None)

`empty_like(a: heat.core.dndarray.DNDarray, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Returns a new uninitialized :class:`~heat.core.dndarray.DNDarray` with the same type, shape and data distribution
    of given object. Data type, data distribution axis, and device can be explicitly overridden.

    Parameters
    ----------
    a : DNDarray
        The shape, data-type, split axis and device of ``a`` define these same attributes of the returned array. Uninitialized array with
        the same shape, type, split axis and device as ``a`` unless overriden.
    dtype : datatype, optional
        Overrides the data type of the result.
    split: int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> x = ht.ones(
    ...     (
    ...         2,
    ...         3,
    ...     )
    ... )
    >>> x
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.empty_like(x)
    DNDarray([[-1.7205e-10,  4.5905e-41,  7.9442e-37],
              [ 0.0000e+00,  4.4842e-44,  0.0000e+00]], dtype=ht.float32, device=cpu:0, split=None)

`eye(shape: int | Sequence[int], dtype: Type[heat.core.types.datatype] = heat.core.types.float32, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Returns a new 2-D :class:`~heat.core.dndarray.DNDarray` with ones on the diagonal and zeroes elsewhere, i.e. an
    identity matrix.

    Parameters
    ----------
    shape : int or Sequence[int,...]
        The shape of the data-type. If only one number is provided, returning array will be square with that size. In
        other cases, the first value represents the number rows, the second the number of columns.
    dtype : datatype, optional
        Overrides the data type of the result.
    split : int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> ht.eye(2)
    DNDarray([[1., 0.],
              [0., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.eye((2, 3), dtype=ht.int32)
    DNDarray([[1, 0, 0],
              [0, 1, 0]], dtype=ht.int32, device=cpu:0, split=None)

`from_partition_dict(parted: dict, comm: heat.core.communication.Communication | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return a newly created DNDarray constructed from the '__partitioned__' attributed of the input object.
    Memory of local partitions will be shared (zero-copy) as long as supported by data objects.
    Currently supports numpy ndarrays and torch tensors as data objects.
    Current limitations:
      * Partitions must be ordered in the partition-grid by rank
      * Only one split-axis
      * Only one partition per rank
      * Only SPMD-style __partitioned__

    Parameters
    ----------
    parted : dict
        A partition dictionary used to create the new DNDarray
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    See Also
    --------
    :func:`ht.core.DNDarray.create_partition_interface <ht.core.DNDarray.create_partition_interface>`.

    Raises
    ------
    AttributeError
        If not hasattr(x, "__partitioned__") or if underlying data has no dtype.
    TypeError
        If it finds an unsupported array types
    RuntimeError
        If other unsupported content is found.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.ones((44, 55), split=0)
    >>> b = ht.from_partition_dict(a.__partitioned__)
    >>> assert (a == b).all()
    >>> a[40] = 4711
    >>> assert (a == b).all()

`from_partitioned(x, comm: heat.core.communication.Communication | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return a newly created DNDarray constructed from the '__partitioned__' attributed of the input object.
    Memory of local partitions will be shared (zero-copy) as long as supported by data objects.
    Currently supports numpy ndarrays and torch tensors as data objects.
    Current limitations:
      * Partitions must be ordered in the partition-grid by rank
      * Only one split-axis
      * Only one partition per rank
      * Only SPMD-style __partitioned__

    Parameters
    ----------
    x : object
        Requires x.__partitioned__
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    See Also
    --------
    :func:`ht.core.DNDarray.create_partition_interface <ht.core.DNDarray.create_partition_interface>`.

    Raises
    ------
    AttributeError
        If not hasattr(x, "__partitioned__") or if underlying data has no dtype.
    TypeError
        If it finds an unsupported array types
    RuntimeError
        If other unsupported content is found.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.ones((44, 55), split=0)
    >>> b = ht.from_partitioned(a)
    >>> assert (a == b).all()
    >>> a[40] = 4711
    >>> assert (a == b).all()

`full(shape: int | Sequence[int], fill_value: int | float, dtype: Type[heat.core.types.datatype] = heat.core.types.float32, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Return a new :class:`~heat.core.dndarray.DNDarray` of given shape and type, filled with ``fill_value``.

    Parameters
    ----------
    shape : int or Sequence[int,...]
        Shape of the new array, e.g., (2, 3) or 2.
    fill_value : scalar
        Fill value.
    dtype : datatype, optional
        The desired data-type for the array
    split: int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> ht.full((2, 2), ht.inf)
    DNDarray([[inf, inf],
              [inf, inf]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.full((2, 2), 10)
    DNDarray([[10., 10.],
              [10., 10.]], dtype=ht.float32, device=cpu:0, split=None)

`full_like(a: heat.core.dndarray.DNDarray, fill_value: int | float, dtype: Type[heat.core.types.datatype] = heat.core.types.float32, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Return a full :class:`~heat.core.dndarray.DNDarray` with the same shape and type as a given array. Data type, data distribution axis, and device can be explicitly overridden.

    Parameters
    ----------
    a : DNDarray
        The shape, data-type, split axis and device of ``a`` define these same attributes of the returned array.
    fill_value : scalar
        Fill value.
    dtype : datatype, optional
        The data type of the result, defaults to `a.dtype`.
    split: int or None, optional
        The axis along which the array is split and distributed; defaults to `a.split`.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to `a.device`.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> x = ht.zeros(
    ...     (
    ...         2,
    ...         3,
    ...     )
    ... )
    >>> x
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.full_like(x, 1.0)
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)

`linspace(start: int | float, stop: int | float, num: int = 50, endpoint: bool = True, retstep: bool = False, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None) ‑> Tuple[heat.core.dndarray.DNDarray, float]`
:   Returns num evenly spaced samples, calculated over the interval ``[start, stop]``. The endpoint of the interval can
    optionally be excluded. There are num equally spaced samples in the closed interval ``[start, stop]`` or the
    half-open interval ``[start, stop)`` (depending on whether endpoint is ``True`` or ``False``).

    Parameters
    ----------
    start: scalar or scalar-convertible
        The starting value of the sample interval, maybe a sequence if convertible to scalar
    stop: scalar or scalar-convertible
        The end value of the sample interval, unless is set to False. In that case, the sequence consists of all but the
        last of ``num+1`` evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint
        is ``False``.
    num: int, optional
        Number of samples to generate, defaults to 50. Must be non-negative.
    endpoint: bool, optional
        If ``True``, stop is the last sample, otherwise, it is not included.
    retstep: bool, optional
        If ``True``, return (samples, step), where step is the spacing between samples.
    dtype: dtype, optional
        The type of the output array.
    split: int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    Examples
    --------
    >>> ht.linspace(2.0, 3.0, num=5)
    DNDarray([2.0000, 2.2500, 2.5000, 2.7500, 3.0000], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.linspace(2.0, 3.0, num=5, endpoint=False)
    DNDarray([2.0000, 2.2000, 2.4000, 2.6000, 2.8000], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.linspace(2.0, 3.0, num=5, retstep=True)
    (DNDarray([2.0000, 2.2500, 2.5000, 2.7500, 3.0000], dtype=ht.float32, device=cpu:0, split=None), 0.25)

`logspace(start: int | float, stop: int | float, num: int = 50, endpoint: bool = True, base: float = 10.0, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return numbers spaced evenly on a log scale. In linear space, the sequence starts at ``base**start`` (``base`` to
    the power of ``start``) and ends with ``base**stop`` (see ``endpoint`` below).

    Parameters
    ----------
    start : scalar or scalar-convertible
        ``base**start`` is the starting value of the sequence.
    stop : scalar or scalar-convertible
        ``base**stop`` is the final value of the sequence, unless `endpoint` is ``False``.  In that case, ``num+1``
        values are spaced over the interval in log-space, of which all but the last (a sequence of length ``num``) are
        returned.
    num : int, optional
        Number of samples to generate.
    endpoint : bool, optional
        If ``True``, `stop` is the last sample. Otherwise, it is not included.
    base : float, optional
        The base of the log space. The step size between the elements in :math:`ln(samples) / ln(base)` (or
        :math:`base(samples)`) is uniform.
    dtype : datatype, optional
        The type of the output array.  If ``dtype`` is not given, infer the data type from the other input arguments.
    split: int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.

    See Also
    --------
    :func:`arange` : Similar to :func:`linspace`, with the step size specified instead of the
        number of samples. Note that, when used with a float endpoint, the endpoint may or may not be included.

    :func:`linspace` : Similar to ``logspace``, but with the samples uniformly distributed in linear space, instead of
        log space.

    Examples
    --------
    >>> ht.logspace(2.0, 3.0, num=4)
    DNDarray([ 100.0000,  215.4434,  464.1590, 1000.0000], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.logspace(2.0, 3.0, num=4, endpoint=False)
    DNDarray([100.0000, 177.8279, 316.2278, 562.3413], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.logspace(2.0, 3.0, num=4, base=2.0)
    DNDarray([4.0000, 5.0397, 6.3496, 8.0000], dtype=ht.float32, device=cpu:0, split=None)

`meshgrid(*arrays: Sequence[heat.core.dndarray.DNDarray], indexing: str = 'xy') ‑> List[heat.core.dndarray.DNDarray]`
:   Returns coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays : Sequence[ DNDarray ]
        one-dimensional arrays representing grid coordinates. If exactly one vector is distributed, the returned matrices will
        be distributed along the axis equal to the index of this vector in the input list.
    indexing : str, optional
        Cartesian ‘xy’ or matrix ‘ij’ indexing of output. It is ignored if zero or one one-dimensional arrays are provided. Default: 'xy' .

    Raises
    ------
    ValueError
        If `indexing` is not 'xy' or 'ij'.
    ValueError
        If more than one input vector is distributed.

    Examples
    --------
    >>> x = ht.arange(4)
    >>> y = ht.arange(3)
    >>> xx, yy = ht.meshgrid(x, y)
    >>> xx
    DNDarray([[0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3]], dtype=ht.int32, device=cpu:0, split=None)
    >>> yy
    DNDarray([[0, 0, 0, 0],
          [1, 1, 1, 1],
          [2, 2, 2, 2]], dtype=ht.int32, device=cpu:0, split=None)

`ones(shape: int | Sequence[int], dtype: Type[heat.core.types.datatype] = heat.core.types.float32, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Returns a new :class:`~heat.core.dndarray.DNDarray` of given shape and data type filled with one. May be allocated
    split up across multiple nodes along the specified axis.

    Parameters
    ----------
    shape : int or Sequence[int,...]
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : datatype, optional
        The desired HeAT data type for the array.
    split : int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> ht.ones(3)
    DNDarray([1., 1., 1.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.ones(3, dtype=ht.int)
    DNDarray([1, 1, 1], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.ones(
    ...     (
    ...         2,
    ...         3,
    ...     )
    ... )
    DNDarray([[1., 1., 1.],
          [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)

`ones_like(a: heat.core.dndarray.DNDarray, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Returns a new :class:`~heat.core.dndarray.DNDarray` filled with ones with the same type,
    shape, data distribution and device of the input object. Data type, data distribution axis, and device can be explicitly overridden.

    Parameters
    ----------
    a : DNDarray
        The shape, data-type, split axis and device of ``a`` define these same attributes of the returned array.
    dtype : datatype, optional
        Overrides the data type of the result.
    split: int or None, optional
        The axis along which the array is split and distributed; defaults to `a.split`.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to `a.device`.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> x = ht.zeros(
    ...     (
    ...         2,
    ...         3,
    ...     )
    ... )
    >>> x
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.ones_like(x)
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)

`zeros(shape: int | Sequence[int], dtype: Type[heat.core.types.datatype] = heat.core.types.float32, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Returns a new :class:`~heat.core.dndarray.DNDarray` of given shape and data type filled with zero values.
    May be allocated split up across multiple nodes along the specified axis.

    Parameters
    ----------
    shape : int or Sequence[int,...]
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : datatype
        The desired HeAT data type for the array.
    split: int or None, optional
        The axis along which the array is split and distributed; ``None`` means no distribution.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> ht.zeros(3)
    DNDarray([0., 0., 0.], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.zeros(3, dtype=ht.int)
    DNDarray([0, 0, 0], dtype=ht.int32, device=cpu:0, split=None)
    >>> ht.zeros(
    ...     (
    ...         2,
    ...         3,
    ...     )
    ... )
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)

`zeros_like(a: heat.core.dndarray.DNDarray, dtype: Type[heat.core.types.datatype] | None = None, split: int | None = None, device: heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, order: str = 'C') ‑> heat.core.dndarray.DNDarray`
:   Returns a new :class:`~heat.core.dndarray.DNDarray` filled with zeros with the same type, shape, data
    distribution, and device of the input object. Data type, data distribution axis, and device can be explicitly overridden.

    Parameters
    ----------
    a : DNDarray
        The shape, data-type, split axis, and device  of ``a`` define these same attributes of the returned array.
    dtype : datatype, optional
        Overrides the data type of the result.
    split: int or None, optional
        The axis along which the array is split and distributed; defaults to `a.split`.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to `a.device`.
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.

    Examples
    --------
    >>> x = ht.ones(
    ...     (
    ...         2,
    ...         3,
    ...     )
    ... )
    >>> x
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.zeros_like(x)
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
