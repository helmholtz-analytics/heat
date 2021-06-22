"""Provides high-level DNDarray initialization functions"""

import numpy as np
import torch
import warnings

from typing import Callable, Iterable, Optional, Sequence, Tuple, Type, Union, List

from .communication import MPI, sanitize_comm, Communication
from .devices import Device
from .dndarray import DNDarray
from .memory import sanitize_memory_layout
from .sanitation import sanitize_in, sanitize_sequence
from .stride_tricks import sanitize_axis, sanitize_shape
from .types import datatype

from . import devices
from . import types


__all__ = [
    "arange",
    "array",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "full",
    "full_like",
    "linspace",
    "logspace",
    "meshgrid",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
]


def arange(
    *args: Union[int, float],
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    device: Optional[Union[str, Device]] = None,
    comm: Optional[Communication] = None
) -> DNDarray:
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)`` (in other words, the interval including `start`
    but excluding `stop`). For integer arguments the function is equivalent to the Python built-in `range
    <http://docs.python.org/lib/built-in-funcs.html>`_ function, but returns a array rather than a list.
    When using a non-integer step, such as 0.1, the results may be inconsistent due to being subject to numerical
    rounding. In the cases the usage of :func:`linspace` is recommended.
    For floating point arguments, the length of the result is :math:`\\lceil(stop-start)/step\\rceil`.
    Again, due to floating point rounding, this rule may result in the last element of `out` being greater than `stop`
    by machine epsilon.

    Parameters
    ----------
    start : scalar, optional
        Start of interval.  The interval includes this value.  The default start value is 0.
    stop : scalar
        End of interval.  The interval does not include this value, except in some cases where ``step`` is not an
        integer and floating point round-off affects the length of ``out``.
    step : scalar, optional
        Spacing between values.  For any output ``out``, this is the distance between two adjacent values,
        ``out[i+1]-out[i]``. The default step size is 1. If ``step`` is specified as a position argument, ``start``
        must also be given.
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
    """
    num_of_param = len(args)

    # check if all positional arguments are integers
    all_ints = all([isinstance(_, int) for _ in args])

    # set start, stop, step, num according to *args
    if num_of_param == 1:
        if dtype is None:
            # use int32 as default instead of int64 used in numpy
            dtype = types.int32 if all_ints else types.float32
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
    balanced = True

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step
    data = torch.arange(start, stop, step, device=device.torch_device)

    htype = types.canonical_heat_type(dtype)
    data = data.type(htype.torch_type())

    return DNDarray(data, gshape, htype, split, device, comm, balanced)


def array(
    obj: Iterable,
    dtype: Optional[Type[datatype]] = None,
    copy: bool = True,
    ndmin: int = 0,
    order: str = "C",
    split: Optional[int] = None,
    is_split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Create a :class:`~heat.core.dndarray.DNDarray`.

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
        If ``True`` (default), then the object is copied. Otherwise, a copy will only be made if obj is a nested
        sequence or if a copy is needed to satisfy any of the other requirements, e.g. ``dtype``.
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
    >>> b.larray.storage()
     0
     1
     2
     3
     4
     5
    [torch.LongStorage of size 6]
    >>> c = ht.array(a, order='F')
    >>> c
    DNDarray([[0, 1, 2],
              [3, 4, 5]], dtype=ht.int64, device=cpu:0, split=None)
    >>> c.strides
    (8, 16)
    >>> c.larray.storage()
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
    >>> b = ht.array(a, order='F', split=0)
    >>> b
    DNDarray([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]], dtype=ht.int64, device=cpu:0, split=0)
    >>> b.strides
    [0/2] (8, 16)
    [1/2] (8, 16)
    >>> b.larray.storage()
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
    """
    # array already exists; no copy
    if (
        isinstance(obj, DNDarray)
        and not copy
        and (dtype is None or dtype == obj.dtype)
        and (split is None or split == obj.split)
        and (is_split is None or is_split == obj.split)
        and (device is None or device == obj.device)
    ):
        return obj

    # extract the internal tensor in case of a heat tensor
    if isinstance(obj, DNDarray):
        obj = obj.larray

    # sanitize the data type
    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    # sanitize device
    if device is not None:
        device = devices.sanitize_device(device)

    # initialize the array
    if bool(copy):
        if isinstance(obj, torch.Tensor):
            # TODO: watch out. At the moment clone() implies losing the underlying memory layout.
            # pytorch fix in progress
            obj = obj.clone().detach()
        else:
            try:
                obj = torch.tensor(
                    obj,
                    dtype=dtype.torch_type() if dtype is not None else None,
                    device=device.torch_device
                    if device is not None
                    else devices.get_device().torch_device,
                )
            except RuntimeError:
                raise TypeError("invalid data of type {}".format(type(obj)))
    else:
        if not isinstance(obj, DNDarray):
            obj = torch.as_tensor(
                obj,
                dtype=dtype.torch_type() if dtype is not None else None,
                device=device.torch_device
                if device is not None
                else devices.get_device().torch_device,
            )

    # infer dtype from obj if not explicitly given
    if dtype is None:
        dtype = types.canonical_heat_type(obj.dtype)
    else:
        torch_dtype = dtype.torch_type()
        if obj.dtype != torch_dtype:
            obj = obj.type(torch_dtype)

    # infer device from obj if not explicitly given
    if device is None:
        device = devices.sanitize_device(obj.device.type)

    if str(obj.device) != device.torch_device:
        warnings.warn(
            "Array 'obj' is not on device '{}'. It will be copied to it.".format(device),
            UserWarning,
        )
        obj = obj.to(device.torch_device)

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

    # sanitize comm object
    comm = sanitize_comm(comm)

    # determine the local and the global shape. If split is None, they are identical
    gshape = list(obj.shape)
    lshape = gshape.copy()
    balanced = True

    # content shall be split, chunk the passed data object up
    if split is not None:
        _, _, slices = comm.chunk(gshape, split)
        obj = obj[slices].clone()
        obj = sanitize_memory_layout(obj, order=order)
    # check with the neighboring rank whether the local shape would fit into a global shape
    elif is_split is not None:
        gshape = np.array(gshape)
        lshape = np.array(lshape)
        obj = sanitize_memory_layout(obj, order=order)
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
        # compare to calculated balanced lshape (cf. dndarray.is_balanced())
        gshape = tuple(int(ele) for ele in gshape)
        lshape = tuple(int(ele) for ele in lshape)
        _, _, chk = comm.chunk(gshape, split)
        test_lshape = tuple([x.stop - x.start for x in chk])
        match = 1 if test_lshape == lshape else 0
        gmatch = comm.allreduce(match, MPI.SUM)
        if gmatch != comm.size:
            balanced = False

    elif split is None and is_split is None:
        obj = sanitize_memory_layout(obj, order=order)

    return DNDarray(obj, tuple(gshape), dtype, split, device, comm, balanced)


def asarray(
    obj: Iterable,
    dtype: Optional[Type[datatype]] = None,
    order: str = "C",
    is_split: Optional[bool] = None,
    device: Optional[Union[str, Device]] = None,
) -> DNDarray:
    """
    Convert ``obj`` to a DNDarray. If ``obj`` is a `DNDarray` or `Tensor` with the same `dtype` and `device` or if the
    data is an `ndarray` of the corresponding ``dtype`` and the ``device`` is the CPU, no copy will be performed.

    Parameters
    ----------
    obj : iterable
        Input data, in any form that can be converted to an array. This includes e.g. lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ndarrays.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.
    order: str, optional
        Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.
    is_split : None or int, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. Useful for
        interfacing with other HPC code. The shape of the global tensor is automatically inferred.
    device : str, ht.Device or None, optional
        Specifies the device the tensor shall be allocated on. By default, it is inferred from the input data.

    Examples
    --------
    >>> a = [1,2]
    >>> ht.asarray(a)
    DNDarray([1, 2], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = np.array([1,2,3])
    >>> n = ht.asarray(a)
    >>> n
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> n[0] = 0
    >>> a
    DNDarray([0, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = torch.tensor([1,2,3])
    >>> t = ht.asarray(a)
    >>> t
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> t[0] = 0
    >>> a
    DNDarray([0, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a = ht.array([1,2,3,4], dtype=ht.float32)
    >>> ht.asarray(a, dtype=ht.float32) is a
    True
    >>> ht.asarray(a, dtype=ht.float64) is a
    False
    """
    return array(obj, dtype=dtype, copy=False, order=order, is_split=is_split, device=device)


def empty(
    shape: Union[int, Sequence[int]],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Returns a new uninitialized :class:`~heat.core.dndarray.DNDarray` of given shape and data type. May be allocated
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
    >>> ht.empty((2, 3,))
    DNDarray([[-1.7206e-10,  4.5905e-41, -1.7206e-10],
              [ 4.5905e-41,  4.4842e-44,  0.0000e+00]], dtype=ht.float32, device=cpu:0, split=None)
    """
    # TODO: implement 'K' option when torch.clone() fix to preserve memory layout is released.
    return __factory(shape, dtype, split, torch.empty, device, comm, order)


def empty_like(
    a: DNDarray,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Returns a new uninitialized :class:`~heat.core.dndarray.DNDarray` with the same type, shape and data distribution
    of given object. Data type and data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : DNDarray
        The shape and data-type of ``a`` define these same attributes of the returned array. Uninitialized array with
        the same shape, type and split axis as ``a`` unless overriden.
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
    >>> x = ht.ones((2, 3,))
    >>> x
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.empty_like(x)
    DNDarray([[-1.7205e-10,  4.5905e-41,  7.9442e-37],
              [ 0.0000e+00,  4.4842e-44,  0.0000e+00]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return __factory_like(a, dtype, split, empty, device, comm, order=order)


def eye(
    shape: Union[int, Sequence[int]],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Returns a new 2-D :class:`~heat.core.dndarray.DNDarray` with ones on the diagonal and zeroes elsewhere, i.e. an
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
    """
    # TODO: implement 'K' option when torch.clone() fix to preserve memory layout is released.
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
    balanced = True

    # start by creating tensor filled with zeroes
    data = torch.zeros(
        lshape, dtype=types.canonical_heat_type(dtype).torch_type(), device=device.torch_device
    )

    # insert ones at the correct positions
    for i in range(min(lshape)):
        pos_x = i if split == 0 else i + offset
        pos_y = i if split == 1 else i + offset
        if pos_x >= lshape[0] or pos_y >= lshape[1]:
            break
        data[pos_x][pos_y] = 1

    data = sanitize_memory_layout(data, order=order)

    return DNDarray(
        data, gshape, types.canonical_heat_type(data.dtype), split, device, comm, balanced
    )


def __factory(
    shape: Union[int, Sequence[int]],
    dtype: Type[datatype],
    split: Optional[int],
    local_factory: Callable,
    device: Device,
    comm: Communication,
    order: str,
) -> DNDarray:
    """
    Abstracted factory function for HeAT :class:`~heat.core.dndarray.DNDarray` initialization.

    Parameters
    ----------
    shape : int or Sequence[ints,...]
        Desired shape of the output array, e.g. 1 or (1, 2, 3,).
    dtype : datatype
        The desired HeAT data type for the array, defaults to ht.float32.
    split : int or None
        The axis along which the array is split and distributed.
    local_factory : callable
        Function that creates the local PyTorch tensor for the DNDarray.
    device : Device
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm : Communication
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.
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
    data = sanitize_memory_layout(data, order=order)

    return DNDarray(data, shape, dtype, split, device, comm, balanced=True)


def __factory_like(
    a: DNDarray,
    dtype: Type[datatype],
    split: Optional[int],
    factory: Callable,
    device: Device,
    comm: Communication,
    order: str = "C",
    **kwargs
) -> DNDarray:
    """
    Abstracted '...-like' factory function for HeAT :class:`~heat.core.dndarray.DNDarray` initialization

    Parameters
    ----------
    a : DNDarray
        The shape and data-type of ``a`` define these same attributes of the returned array.
    dtype : datatype
        The desired HeAT data type for the array.
    split: int or None, optional
        The axis along which the array is split and distributed, defaults to no distribution).
    factory : function
        Function that creates a DNDarray.
    device : str
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on, defaults to globally set
        default device.
    comm: Communication
        Handle to the nodes holding distributed parts or copies of this array.
    order: str, optional
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).

    Raises
    ------
    NotImplementedError
        If order is one of the NumPy options ``'K'`` or ``'A'``.
    """
    # TODO: implement 'K' option when torch.clone() fix to preserve memory layout is released.
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

    return factory(shape, dtype=dtype, split=split, device=device, comm=comm, order=order, **kwargs)


def full(
    shape: Union[int, Sequence[int]],
    fill_value: Union[int, float],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Return a new :class:`~heat.core.dndarray.DNDarray` of given shape and type, filled with ``fill_value``.

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
    """

    def local_factory(*args, **kwargs):
        return torch.full(*args, fill_value=fill_value, **kwargs)

    # Will be redundant with PyTorch 1.7
    if isinstance(fill_value, complex):
        dtype = types.complex64

    return __factory(shape, dtype, split, local_factory, device, comm, order=order)


def full_like(
    a: DNDarray,
    fill_value: Union[int, float],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Return a full :class:`~heat.core.dndarray.DNDarray` with the same shape and type as a given array.

    Parameters
    ----------
    a : DNDarray
        The shape and data-type of ``a`` define these same attributes of the returned array.
    fill_value : scalar
        Fill value.
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
    >>> x = ht.zeros((2, 3,))
    >>> x
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.full_like(x, 1.0)
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return __factory_like(a, dtype, split, full, device, comm, fill_value=fill_value, order=order)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> Tuple[DNDarray, float]:
    """
    Returns num evenly spaced samples, calculated over the interval ``[start, stop]``. The endpoint of the interval can
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
    balanced = True

    # compose the local tensor
    start += offset * step
    stop = start + lshape[0] * step - step
    data = torch.linspace(start, stop, lshape[0], device=device.torch_device)
    if dtype is not None:
        data = data.type(types.canonical_heat_type(dtype).torch_type())

    # construct the resulting global tensor
    ht_tensor = DNDarray(
        data, gshape, types.canonical_heat_type(data.dtype), split, device, comm, balanced
    )

    if retstep:
        return ht_tensor, step
    return ht_tensor


def logspace(
    start: Union[int, float],
    stop: Union[int, float],
    num: int = 50,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> DNDarray:
    """
    Return numbers spaced evenly on a log scale. In linear space, the sequence starts at ``base**start`` (``base`` to
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
    """
    y = linspace(start, stop, num=num, endpoint=endpoint, split=split, device=device, comm=comm)
    if dtype is None:
        return pow(base, y)
    return pow(base, y).astype(dtype, copy=False)


def meshgrid(*arrays: Sequence[DNDarray], indexing: str = "xy") -> List[DNDarray]:
    """
    Returns coordinate matrices from coordinate vectors.

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
    >>> xx, yy = ht.meshgrid(x,y)
    >>> xx
    DNDarray([[0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3]], dtype=ht.int32, device=cpu:0, split=None)
    >>> yy
    DNDarray([[0, 0, 0, 0],
          [1, 1, 1, 1],
          [2, 2, 2, 2]], dtype=ht.int32, device=cpu:0, split=None)
    """
    splitted = None

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    if len(arrays) == 0:
        return []

    arrays = sanitize_sequence(arrays)

    for idx, array in enumerate(arrays):
        sanitize_in(array)
        if array.split is not None:
            if splitted is not None:
                raise ValueError("split != None are not supported.")
            splitted = idx

    # pytorch does not support the indexing keyword: switch vectors
    if indexing == "xy" and len(arrays) > 1:
        arrays[0], arrays[1] = arrays[1], arrays[0]
        if splitted == 0:
            arrays[0] = arrays[0].resplit(0)
            arrays[1] = arrays[1].resplit(None)
        elif splitted == 1:
            arrays[0] = arrays[0].resplit(None)
            arrays[1] = arrays[1].resplit(0)

    grids = torch.meshgrid(*(array.larray for array in arrays))

    # pytorch does not support indexing keyword: switch back
    if indexing == "xy" and len(arrays) > 1:
        grids = list(grids)
        grids[0], grids[1] = grids[1], grids[0]

    shape = tuple(array.size for array in arrays)

    return list(
        DNDarray(
            array=grid,
            gshape=shape,
            dtype=types.heat_type_of(grid),
            split=splitted,
            device=devices.sanitize_device(grid.device.type),
            comm=sanitize_comm(None),
            balanced=True,
        )
        for grid in grids
    )


def ones(
    shape: Union[int, Sequence[int]],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Returns a new :class:`~heat.core.dndarray.DNDarray` of given shape and data type filled with one. May be allocated
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
    >>> ht.ones((2, 3,))
    DNDarray([[1., 1., 1.],
          [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    # TODO: implement 'K' option when torch.clone() fix to preserve memory layout is released.
    return __factory(shape, dtype, split, torch.ones, device, comm, order)


def ones_like(
    a: DNDarray,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Returns a new :class:`~heat.core.dndarray.DNDarray` filled with ones with the same type,
    shape and data distribution of given object. Data type and data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : DNDarray
        The shape and data-type of ``a`` define these same attributes of the returned array.
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
    >>> x = ht.zeros((2, 3,))
    >>> x
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.ones_like(x)
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    return __factory_like(a, dtype, split, ones, device, comm, order=order)


def zeros(
    shape: Union[int, Sequence[int]],
    dtype: Type[datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Returns a new :class:`~heat.core.dndarray.DNDarray` of given shape and data type filled with zero values.
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
    >>> ht.zeros((2, 3,))
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    # TODO: implement 'K' option when torch.clone() fix to preserve memory layout is released.
    return __factory(shape, dtype, split, torch.zeros, device, comm, order=order)


def zeros_like(
    a: DNDarray,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Returns a new :class:`~heat.core.dndarray.DNDarray` filled with zeros with the same type, shape and data
    distribution of given object. Data type and data distribution strategy can be explicitly overriden.

    Parameters
    ----------
    a : DNDarray
        The shape and data-type of ``a`` define these same attributes of the returned array.
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
    >>> x = ht.ones((2, 3,))
    >>> x
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.zeros_like(x)
    DNDarray([[0., 0., 0.],
              [0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=None)
    """
    # TODO: implement 'K' option when torch.clone() fix to preserve memory layout is released.
    return __factory_like(a, dtype, split, zeros, device, comm, order=order)
