import numpy as np
import torch

from .communication import MPI, sanitize_comm
from .stride_tricks import sanitize_axis, sanitize_shape
from . import devices
from . import dndarray
from . import types

__all__ = [
    "arange",
    "array",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "logspace",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
]


def arange(*args, dtype=None, split=None, device=None, comm=None):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)`` (in other words, the interval including `start`
    but excluding `stop`). For integer arguments the function is equivalent to the Python built-in `range
    <http://docs.python.org/lib/built-in-funcs.html>`_ function, but returns a tensor rather than a list.

    When using a non-integer step, such as 0.1, the results will often not be consistent. It is better to use
    ``linspace`` for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except in some cases where `step` is not an integer
        and floating point round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance between two adjacent values, ``out[i+1] -
        out[i]``. The default step size is 1. If `step` is specified as a position argument, `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data type from the other input arguments.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    arange : ht.DNDarraytensor
        1D heat tensor of evenly spaced values.

        For floating point arguments, the length of the result is ``ceil((stop - start)/step)``. Because of floating
        point overflow, this rule may result in the last element of `out` being greater than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.

    Examples
    --------
    >>> ht.arange(3)
    tensor([0, 1, 2])
    >>> ht.arange(3.0)
    tensor([ 0.,  1.,  2.])
    >>> ht.arange(3, 7)
    tensor([3, 4, 5, 6])
    >>> ht.arange(3, 7, 2)
    tensor([3, 5])
    """
    num_of_param = len(args)

    # check if all positional arguments are integers
    all_ints = all([isinstance(_, int) for _ in args])

    # set start, stop, step, num according to *args
    if num_of_param == 1:
        if dtype is None:
            # use int32 as default instead of int64 used in numpy
            dtype = types.int32
        start = 0
        stop = int(np.ceil(args[0]))
        step = 1
        num = stop
    elif num_of_param == 2:
        if dtype is None:
            dtype = types.int32 if all_ints else types.float32
        start = args[0]
        stop = args[1]
        step = 1
        num = int(np.ceil(stop - start))
    elif num_of_param == 3:
        if dtype is None:
            dtype = types.int32 if all_ints else types.float32
        start = args[0]
        stop = args[1]
        step = args[2]
        num = int(np.ceil((stop - start) / step))
    else:
        raise TypeError(
            "function takes minimum one and at most 3 positional arguments ({} given)".format(
                num_of_param
            )
        )

    # sanitize device and comm
    device = devices.sanitize_device(device)
    comm = sanitize_comm(comm)

    gshape = (num,)
    split = sanitize_axis(gshape, split)
    offset, lshape, _ = comm.chunk(gshape, split)

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step
    data = torch.arange(start, stop, step, device=device.torch_device)

    htype = types.canonical_heat_type(dtype)
    data = data.type(htype.torch_type())

    return dndarray.DNDarray(data, gshape, htype, split, device, comm)


def array(obj, dtype=None, copy=True, ndmin=0, split=None, is_split=None, device=None, comm=None):
    """
    Create a tensor.
    Parameters
    ----------
    obj : array_like
        A tensor or array, any object exposing the array interface, an object whose __array__ method returns an array,
        or any (nested) sequence.
    dtype : dtype, optional
        The desired data-type for the array. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied. Otherwise, a copy will only be made if obj is a nested sequence or
        if a copy is needed to satisfy any of the other requirements, e.g. dtype.
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have. Ones will, if needed, be
        attached to the shape if  ndim>0  and prefaced in case of ndim<0 to meet the requirement.
    split : None or int, optional
        The axis along which the passed array content obj is split and distributed in memory. Mutually exclusive with
        is_split.
    is_split : None or int, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. Useful for
        interfacing with other HPC code. The shape of the global tensor is automatically inferred. Mutually exclusive
        with split.
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed tensor chunks.

    Returns
    -------
    out : ht.DNDarray
        A tensor object satisfying the specified requirements.

    Examples
    --------
    >>> ht.array([1, 2, 3])
    tensor([1, 2, 3])

    Upcasting:
    >>> ht.array([1, 2, 3.0])
    tensor([ 1.,  2.,  3.])

    More than one dimension:
    >>> ht.array([[1, 2], [3, 4]])
    tensor([[1, 2],
            [3, 4]])

    Minimum dimensions given:
    >>> ht.array([1, 2, 3], ndmin=2)
    tensor([[1, 2, 3]])

    Type provided:
    >>> ht.array([1, 2, 3], dtype=float)
    tensor([ 1.0, 2.0, 3.0])

    Split data:
    >>> ht.array([1, 2, 3, 4], split=0)
    (0/2) tensor([1, 2])
    (1/2) tensor([3, 4])

    Pre-split data:
    (0/2) >>> ht.array([1, 2], is_split=0)
    (1/2) >>> ht.array([3, 4], is_split=0)
    (0/2) tensor([1, 2, 3, 4])
    (1/2) tensor([1, 2, 3, 4])
    """
    # extract the internal tensor in case of a heat tensor
    if isinstance(obj, dndarray.DNDarray):
        obj = obj._DNDarray__array

    # sanitize the data type
    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    # initialize the array
    if bool(copy):
        if isinstance(obj, torch.Tensor):
            obj = obj.clone().detach()
        else:
            try:
                obj = torch.tensor(obj, dtype=dtype.torch_type() if dtype is not None else None)
            except RuntimeError:
                raise TypeError("invalid data of type {}".format(type(obj)))

    # infer dtype from obj if not explicitly given
    if dtype is None:
        dtype = types.canonical_heat_type(obj.dtype)

    # sanitize minimum number of dimensions
    if not isinstance(ndmin, int):
        raise TypeError("expected ndmin to be int, but was {}".format(type(ndmin)))

    # reshape the object to encompass additional dimensions
    ndmin_abs = abs(ndmin) - len(obj.shape)
    if ndmin_abs > 0 and ndmin > 0:
        obj = obj.reshape(obj.shape + ndmin_abs * (1,))
    if ndmin_abs > 0 > ndmin:
        obj = obj.reshape(ndmin_abs * (1,) + obj.shape)

    # sanitize the split axes, ensure mutual exclusiveness
    split = sanitize_axis(obj.shape, split)
    is_split = sanitize_axis(obj.shape, is_split)
    if split is not None and is_split is not None:
        raise ValueError("split and is_split are mutually exclusive parameters")

    # sanitize device and object
    device = devices.sanitize_device(device)
    comm = sanitize_comm(comm)

    # determine the local and the global shape, if not split is given, they are identical
    lshape = np.array(obj.shape)
    gshape = lshape.copy()

    # content shall be split, chunk the passed data object up
    if split is not None:
        _, _, slices = comm.chunk(obj.shape, split)
        obj = obj[slices].clone()
    # check with the neighboring rank whether the local shape would fit into a global shape
    elif is_split is not None:
        if comm.rank < comm.size - 1:
            comm.Isend(lshape, dest=comm.rank + 1)
        if comm.rank != 0:
            # look into the message of the neighbor to see whether the shape length fits
            status = MPI.Status()
            comm.Probe(source=comm.rank - 1, status=status)
            length = status.Get_count() // lshape.dtype.itemsize

            # the number of shape elements does not match with the 'left' rank
            if length != len(lshape):
                discard_buffer = np.empty(length)
                comm.Recv(discard_buffer, source=comm.rank - 1)
                gshape[is_split] = np.iinfo(gshape.dtype).min
            else:
                # check whether the individual shape elements match
                comm.Recv(gshape, source=comm.rank - 1)
                for i in range(length):
                    if i == is_split:
                        continue
                    elif lshape[i] != gshape[i] and lshape[i] - 1 != gshape[i]:
                        gshape[is_split] = np.iinfo(gshape.dtype).min

        # sum up the elements along the split dimension
        reduction_buffer = np.array(gshape[is_split])
        comm.Allreduce(MPI.IN_PLACE, reduction_buffer, MPI.SUM)
        if reduction_buffer < 0:
            raise ValueError("unable to construct tensor, shape of local data chunk does not match")
        ttl_shape = np.array(obj.shape)
        ttl_shape[is_split] = lshape[is_split]
        comm.Allreduce(MPI.IN_PLACE, ttl_shape, MPI.SUM)
        gshape[is_split] = ttl_shape[is_split]
        split = is_split

    return dndarray.DNDarray(obj, tuple(int(ele) for ele in gshape), dtype, split, device, comm)


def empty(shape, dtype=types.float32, split=None, device=None, comm=None):
    """
    Returns a new uninitialized array of given shape and data type. May be allocated split up across multiple
    nodes along the specified axis.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of zeros with given shape, data type and node distribution.

    Examples
    --------
    >>> ht.empty(3)
    tensor([ 0.0000e+00, -2.0000e+00,  3.3113e+35])

    >>> ht.empty(3, dtype=ht.int)
    tensor([ 0.0000e+00, -2.0000e+00,  3.3113e+35])

    >>> ht.empty((2, 3,))
    tensor([[ 0.0000e+00, -2.0000e+00,  3.3113e+35],
            [ 3.6902e+19,  1.2096e+04,  7.1846e+22]])
    """
    return __factory(shape, dtype, split, torch.empty, device, comm)


def empty_like(a, dtype=None, split=None, device=None, comm=None):
    """
    Returns a new uninitialized array with the same type, shape and data distribution of given object. Data type and
    data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
        Uninitialized tensor with the same shape, type and split axis as 'a' unless overriden.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        A new uninitialized array.

    Examples
    --------
    >>> x = ht.ones((2, 3,))
    >>> x
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

    >>> ht.empty_like(x)
    tensor([[ 0.0000e+00, -2.0000e+00,  3.3113e+35],
            [ 3.6902e+19,  1.2096e+04,  7.1846e+22]])
    """
    return __factory_like(a, dtype, split, empty, device, comm)


def eye(shape, dtype=types.float32, split=None, device=None, comm=None):
    """
    Returns a new 2-D tensor with ones on the diagonal and zeroes elsewhere.

    Parameters
    ----------
    shape : int or tuple of ints
            The shape of the data-type. If only one number is provided, returning tensor will be square with that size.
            In other cases, the first value represents the number rows, the second the number of columns.
    dtype : ht.dtype, optional
            Overrides the data type of the result.
    split : int, optional
            The axis along which the tensor is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
            Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm : Communication, optional
            Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        An identity matrix.

    Examples
    --------
    >>> import heat as ht
    >>> ht.eye(2)
    tensor([[1., 0.],
            [0., 1.]])

    >>> ht.eye((2, 3), dtype=ht.int32)
    tensor([[1, 0, 0],
            [0, 1, 0]], dtype=torch.int32)
    """
    # Determine the actual size of the resulting data
    gshape = shape
    if isinstance(gshape, int):
        gshape = (gshape, gshape)
    if len(gshape) == 1:
        gshape = gshape * 2

    split = sanitize_axis(gshape, split)
    device = devices.sanitize_device(device)
    comm = sanitize_comm(comm)
    offset, lshape, _ = comm.chunk(gshape, split)

    # start by creating tensor filled with zeroes
    data = torch.zeros(
        lshape, dtype=types.canonical_heat_type(dtype).torch_type(), device=device.torch_device
    )

    # insert ones at the correct positions
    for i in range(min(lshape)):
        pos_x = i if split == 0 else i + offset
        pos_y = i if split == 1 else i + offset
        data[pos_x][pos_y] = 1

    return dndarray.DNDarray(
        data, gshape, types.canonical_heat_type(data.dtype), split, device, comm
    )


def __factory(shape, dtype, split, local_factory, device, comm):
    """
    Abstracted factory function for HeAT tensor initialization.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split : int
        The axis along which the array is split and distributed.
    local_factory : function
        Function that creates the local PyTorch tensor for the HeAT tensor.
    device : str or None
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of ones with given shape, data type and node distribution.
    """
    # clean the user input
    shape = sanitize_shape(shape)
    dtype = types.canonical_heat_type(dtype)
    split = sanitize_axis(shape, split)
    device = devices.sanitize_device(device)
    comm = sanitize_comm(comm)

    # chunk the shape if necessary
    _, local_shape, _ = comm.chunk(shape, split)
    # create the torch data using the factory function
    data = local_factory(local_shape, dtype=dtype.torch_type(), device=device.torch_device)

    return dndarray.DNDarray(data, shape, dtype, split, device, comm)


def __factory_like(a, dtype, split, factory, device, comm, **kwargs):
    """
    Abstracted '...-like' factory function for HeAT tensor initialization

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    factory : function
        Function that creates a HeAT tensor.
    device : str or None
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of ones with given shape, data type and node distribution that is like a
    """
    # determine the global shape of the object to create
    # attempt in this order: shape property, length of object or default shape (1,)
    try:
        shape = a.shape
    except AttributeError:
        try:
            shape = (len(a),)
        except TypeError:
            shape = (1,)

    # infer the data type, otherwise default to float32
    if dtype is None:
        try:
            dtype = types.heat_type_of(a)
        except TypeError:
            dtype = types.float32

    # infer split axis
    if split is None:
        try:
            split = a.split if not isinstance(a, str) else None
        except AttributeError:
            # do not split at all
            pass

    # use the default communicator, if not set
    comm = sanitize_comm(comm)

    return factory(shape, dtype=dtype, split=split, device=device, comm=comm, **kwargs)


def full(shape, fill_value, dtype=types.float32, split=None, device=None, comm=None):
    """
    Return a new array of given shape and type, filled with fill_value.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., (2, 3) or 2.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of fill_value with the given shape, dtype and split.

    Examples
    --------
    >>> ht.full((2, 2), np.inf)
    tensor([[ inf,  inf],
            [ inf,  inf]])
    >>> ht.full((2, 2), 10)
    tensor([[10, 10],
            [10, 10]])
    """

    def local_factory(*args, **kwargs):
        return torch.full(*args, fill_value=fill_value, **kwargs)

    return __factory(shape, dtype, split, local_factory, device, comm)


def full_like(a, fill_value, dtype=types.float32, split=None, device=None, comm=None):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    fill_value : scalar
        Fill value.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of fill_value with the same shape and type as a.

    Examples
    --------
    >>> x = ht.zeros((2, 3,))
    >>> x
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

    >>> ht.full_like(a, 1.0)
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """
    return __factory_like(a, dtype, split, full, device, comm, fill_value=fill_value)


def linspace(
    start,
    stop,
    num=50,
    endpoint=True,
    retstep=False,
    dtype=None,
    split=None,
    device=None,
    comm=None,
):
    """
    Returns num evenly spaced samples, calculated over the interval [start, stop]. The endpoint of the interval can
    optionally be excluded.

    Parameters
    ----------
    start: scalar, scalar-convertible
        The starting value of the sample interval, maybe a sequence if convertible to scalar
    stop: scalar, scalar-convertible
        The end value of the sample interval, unless is set to False. In that case, the sequence consists of all but the
        last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint
        is False.
    num: int, optional
        Number of samples to generate, defaults to 50. Must be non-negative.
    endpoint: bool, optional
        If True, stop is the last sample, otherwise, it is not included. Defaults to True.
    retstep: bool, optional
        If True, return (samples, step), where step is the spacing between samples.
    dtype: dtype, optional
        The type of the output array.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    samples: ht.DNDarray
        There are num equally spaced samples in the closed interval [start, stop] or the half-open interval
        [start, stop) (depending on whether endpoint is True or False).
    step: float, optional
        Size of spacing between samples, only returned if retstep is True.

    Examples
    --------
    >>> ht.linspace(2.0, 3.0, num=5)
    tensor([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    >>> ht.linspace(2.0, 3.0, num=5, endpoint=False)
    tensor([ 2. ,  2.2,  2.4,  2.6,  2.8])
    >>> ht.linspace(2.0, 3.0, num=5, retstep=True)
    (array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)
    """
    # sanitize input parameters
    start = float(start)
    stop = float(stop)
    num = int(num)
    if num <= 0:
        raise ValueError(
            "number of samples 'num' must be non-negative integer, but was {}".format(num)
        )
    step = (stop - start) / max(1, num - 1 if endpoint else num)

    # sanitize device and comm
    device = devices.sanitize_device(device)
    comm = sanitize_comm(comm)

    # infer local and global shapes
    gshape = (num,)
    split = sanitize_axis(gshape, split)
    offset, lshape, _ = comm.chunk(gshape, split)

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step - step
    data = torch.linspace(start, stop, lshape[0], device=device.torch_device)
    if dtype is not None:
        data = data.type(types.canonical_heat_type(dtype).torch_type())

    # construct the resulting global tensor
    ht_tensor = dndarray.DNDarray(
        data, gshape, types.canonical_heat_type(data.dtype), split, device, comm
    )

    if retstep:
        return ht_tensor, step
    return ht_tensor


def logspace(
    start, stop, num=50, endpoint=True, base=10.0, dtype=None, split=None, device=None, comm=None
):
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence.
    stop : array_like
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    samples : ht.DNDarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.

    Examples
    --------
    >>> ht.logspace(2.0, 3.0, num=4)
    tensor([ 100.0000,  215.4434,  464.1590, 1000.0000])
    >>> ht.logspace(2.0, 3.0, num=4, endpoint=False)
    tensor([100.0000, 177.8279, 316.2278, 562.3413])
    >>> ht.logspace(2.0, 3.0, num=4, base=2.0)
    tensor([4.0000, 5.0397, 6.3496, 8.0000])
    """
    y = linspace(start, stop, num=num, endpoint=endpoint, split=split, device=device, comm=comm)
    if dtype is None:
        return pow(base, y)
    return pow(base, y).astype(dtype, copy=False)


def ones(shape, dtype=types.float32, split=None, device=None, comm=None):
    """
    Returns a new array of given shape and data type filled with one values. May be allocated split up across multiple
    nodes along the specified axis.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split : int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of ones with given shape, data type and node distribution.

    Examples
    --------
    >>> ht.ones(3)
    tensor([1., 1., 1.])

    >>> ht.ones(3, dtype=ht.int)
    tensor([1, 1, 1])

    >>> ht.ones((2, 3,))
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """
    return __factory(shape, dtype, split, torch.ones, device, comm)


def ones_like(a, dtype=None, split=None, device=None, comm=None):
    """
    Returns a new array filled with ones with the same type, shape and data distribution of given object. Data type and
    data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of ones with the same shape, type and split axis as 'a' unless overriden.

    Examples
    --------
    >>> x = ht.zeros((2, 3,))
    >>> x
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

    >>> ht.ones_like(a)
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """
    return __factory_like(a, dtype, split, ones, device, comm)


def zeros(shape, dtype=types.float32, split=None, device=None, comm=None):
    """
    Returns a new array of given shape and data type filled with zero values. May be allocated split up across multiple
    nodes along the specified axis.

    Parameters
    ----------
    shape : int or sequence of ints
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : ht.dtype
        The desired HeAT data type for the array, defaults to ht.float32.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of zeros with given shape, data type and node distribution.

    Examples
    --------
    >>> ht.zeros(3)
    tensor([0., 0., 0.])

    >>> ht.zeros(3, dtype=ht.int)
    tensor([0, 0, 0])

    >>> ht.zeros((2, 3,))
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    """
    return __factory(shape, dtype, split, torch.zeros, device, comm)


def zeros_like(a, dtype=None, split=None, device=None, comm=None):
    """
    Returns a new array filled with zeros with the same type, shape and data distribution of given object. Data type and
    data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : object
        The shape and data-type of 'a' define these same attributes of the returned array.
    dtype : ht.dtype, optional
        Overrides the data type of the result.
    split: int, optional
        The axis along which the array is split and distributed, defaults to None (no distribution).
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on, defaults to None (i.e. globally set default device).
    comm: Communication, optional
        Handle to the nodes holding distributed parts or copies of this tensor.

    Returns
    -------
    out : ht.DNDarray
        Array of zeros with the same shape, type and split axis as 'a' unless overriden.

    Examples
    --------
    >>> x = ht.ones((2, 3,))
    >>> x
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

    >>> ht.zeros_like(x)
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    """
    return __factory_like(a, dtype, split, zeros, device, comm)
