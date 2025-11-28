"""Provides high-level DCSR_matrix initialization functions"""

import torch
import numpy as np
from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse import csc_matrix as scipy_csc_matrix

from typing import Optional, Type, Iterable
import warnings

from ..core import devices
from ..core import types
from ..core.communication import MPI, sanitize_comm, Communication
from ..core.devices import Device
from ..core.types import datatype

from .dcsx_matrix import DCSC_matrix, DCSR_matrix, __DCSX_matrix

__all__ = [
    "sparse_csr_matrix",
    "sparse_csc_matrix",
]


def sparse_csr_matrix(
    obj: Iterable,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    is_split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> DCSR_matrix:
    """
    Create a :class:`~heat.sparse.DCSR_matrix`.

    Parameters
    ----------
    obj : array_like
        A tensor or array, any object exposing the array interface, an object whose ``__array__`` method returns an
        array, or any (nested) sequence. Sparse tensor that needs to be distributed.
    dtype : datatype, optional
        The desired data-type for the sparse matrix. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the :func:`~heat.sparse.DCSR_matrix.astype` method.
    split : int or None, optional
        The axis along which the passed array content ``obj`` is split and distributed in memory. DCSR_matrix only supports
        distribution along axis 0. Mutually exclusive with ``is_split``.
    is_split : int or None, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. DCSR_matrix only
        supports distribution along axis 0. Useful for interfacing with other distributed-memory code. The shape of the global
        array is automatically inferred. Mutually exclusive with ``split``.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on (i.e. globally set default
        device).
    comm : Communication, optional
        Handle to the nodes holding distributed array chunks.

    Raises
    ------
    ValueError
        If split and is_split parameters are not one of 0 or None.

    Examples
    --------
    Create a :class:`~heat.sparse.DCSR_matrix` from :class:`torch.Tensor` (layout ==> torch.sparse_csr)
    >>> indptr = torch.tensor([0, 2, 3, 6])
    >>> indices = torch.tensor([0, 2, 2, 0, 1, 2])
    >>> data = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
    >>> torch_sparse_csr = torch.sparse_csr_tensor(indptr, indices, data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(torch_sparse_csr, split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.DCSR_matrix` from :class:`scipy.sparse.csr_matrix`
    >>> scipy_sparse_csr = scipy.sparse.csr_matrix((data, indices, indptr))
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(scipy_sparse_csr, split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6], dtype=torch.int32), indices: tensor([0, 2, 2, 0, 1, 2], dtype=torch.int32), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.DCSR_matrix` using data that is already distributed (with `is_split`)
    >>> indptrs = [torch.tensor([0, 2, 3]), torch.tensor([0, 3])]
    >>> indices = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 2])]
    >>> data = [torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([4, 5, 6], dtype=torch.float)]
    >>> rank = ht.MPI_WORLD.rank
    >>> local_indptr = indptrs[rank]
    >>> local_indices = indices[rank]
    >>> local_data = data[rank]
    >>> local_torch_sparse_csr = torch.sparse_csr_tensor(local_indptr, local_indices, local_data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(local_torch_sparse_csr, is_split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.DCSR_matrix` from List
    >>> ht.sparse.sparse_csr_matrix([[0, 0, 1], [1, 0, 2], [0, 0, 3]])
    (indptr: tensor([0, 1, 3, 4]), indices: tensor([2, 0, 2, 2]), data: tensor([1, 1, 2, 3]), dtype=ht.int64, device=cpu:0, split=None)
    """
    return __sparse_matrix(obj, dtype, split, is_split, device, comm, orientation="row")


def sparse_csc_matrix(
    obj: Iterable,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    is_split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> DCSC_matrix:
    """
    Create a :class:`~heat.sparse.DCSC_matrix`.

    Parameters
    ----------
    obj : array_like
        A tensor or array, any object exposing the array interface, an object whose ``__array__`` method returns an
        array, or any (nested) sequence. Sparse tensor that needs to be distributed.
    dtype : datatype, optional
        The desired data-type for the sparse matrix. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the :func:`~heat.sparse.DCSC_matrix.astype` method.
    split : int or None, optional
        The axis along which the passed array content ``obj`` is split and distributed in memory. DCSC_matrix only supports
        distribution along axis 1. Mutually exclusive with ``is_split``.
    is_split : int or None, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. DCSC_matrix only
        supports distribution along axis 1. Useful for interfacing with other distributed-memory code. The shape of the global
        array is automatically inferred. Mutually exclusive with ``split``.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on (i.e. globally set default
        device).
    comm : Communication, optional
        Handle to the nodes holding distributed array chunks.

    Raises
    ------
    ValueError
        If split and is_split parameters are not one of 1 or None.

    Examples
    --------
    Create a :class:`~heat.sparse.DCSC_matrix` from :class:`torch.Tensor` (layout ==> torch.sparse_csc)
    >>> indptr = torch.tensor([0, 2, 3, 6])
    >>> indices = torch.tensor([0, 2, 2, 0, 1, 2])
    >>> data = torch.tensor([1.0, 4.0, 5.0, 2.0, 3.0, 6.0], dtype=torch.float)
    >>> torch_sparse_csc = torch.sparse_csc_tensor(indptr, indices, data)
    >>> heat_sparse_csc = ht.sparse.sparse_csc_matrix(torch_sparse_csc, split=1)
    >>> heat_sparse_csc
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 4., 5., 2., 3., 6.]), dtype=ht.float32, device=cpu:0, split=1)

    Create a :class:`~heat.sparse.DCSC_matrix` from :class:`scipy.sparse.csc_matrix`
    >>> scipy_sparse_csc = scipy.sparse.csc_matrix((data, indices, indptr))
    >>> heat_sparse_csc = ht.sparse.sparse_csc_matrix(scipy_sparse_csc, split=1)
    >>> heat_sparse_csc
    (indptr: tensor([0, 2, 3, 6], dtype=torch.int32), indices: tensor([0, 2, 2, 0, 1, 2], dtype=torch.int32), data: tensor([1., 4., 5., 2., 3., 6.]), dtype=ht.float32, device=cpu:0, split=1)

    Create a :class:`~heat.sparse.DCSC_matrix` using data that is already distributed (with `is_split`)
    >>> indptrs = [torch.tensor([0, 2, 3]), torch.tensor([0, 3])]
    >>> indices = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 2])]
    >>> data = [torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([4, 5, 6], dtype=torch.float)]
    >>> rank = ht.MPI_WORLD.rank
    >>> local_indptr = indptrs[rank]
    >>> local_indices = indices[rank]
    >>> local_data = data[rank]
    >>> local_torch_sparse_csr = torch.sparse_csr_tensor(local_indptr, local_indices, local_data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(local_torch_sparse_csr, is_split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=1)

    Create a :class:`~heat.sparse.DCSC_matrix` from List
    >>> ht.sparse.sparse_csc_matrix([[0, 0, 1], [1, 0, 2], [0, 0, 3]])
    (indptr: tensor([0, 1, 1, 4]), indices: tensor([1, 0, 1, 2]), data: tensor([1, 1, 2, 3]), dtype=ht.int64, device=cpu:0, split=None)
    """
    return __sparse_matrix(obj, dtype, split, is_split, device, comm, orientation="col")


def __sparse_matrix(
    obj: Iterable,
    dtype: Optional[Type[datatype]] = None,
    split: Optional[int] = None,
    is_split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    orientation: str = "row",
) -> __DCSX_matrix:
    """
    Create a :class:`~heat.sparse.__DCSX_matrix`.
    This is a common method for converting a distributed array to a sparse matrix representation.

    Raises
    ------
    ValueError
        If the orientation is not ``'row'`` or ``'col'``.
    ValueError
        If the number of dimensions of the input array is not 2.
    ValueError
        If the split or is_split axis is not supported for the type.
    """
    # version check
    if int(torch.__version__.split(".")[0]) <= 1 and int(torch.__version__.split(".")[1]) < 10:
        raise RuntimeError(f"ht.sparse requires torch >= 1.10. Found version {torch.__version__}.")

    # sanitize the data type
    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    # sanitize device
    if device is not None:
        device = devices.sanitize_device(device)

    if orientation not in ["row", "col"]:
        raise ValueError(f"Invalid orientation: '{orientation}'. Options: 'row' or 'col'")

    torch_class = torch.sparse_csr_tensor if orientation == "row" else torch.sparse_csc_tensor
    # Convert input into torch.Tensor (layout ==> torch.sparse_csr)
    # TODO: Check if conversion works across types
    if isinstance(obj, (scipy_csr_matrix, scipy_csc_matrix)):
        obj = torch_class(
            obj.indptr,
            obj.indices,
            obj.data,
            device=device.torch_device if device is not None else devices.get_device().torch_device,
            size=obj.shape,
        )

    if not isinstance(obj, torch.Tensor):
        try:
            obj = torch.tensor(
                obj,
                device=(
                    device.torch_device if device is not None else devices.get_device().torch_device
                ),
            )
        except RuntimeError:
            raise TypeError(f"Invalid data of type {type(obj)}")

    if obj.ndim != 2:
        raise ValueError(f"The number of dimensions must be 2, found {str(obj.ndim)}")

    torch_layout = torch.sparse_csr if orientation == "row" else torch.sparse_csc
    if obj.layout != torch_layout:
        if torch_layout == torch.sparse_csr:
            obj = obj.to_sparse_csr()
        else:
            obj = obj.to_sparse_csc()

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
            f"Array 'obj' is not on device '{device}'. It will be moved to it.",
            UserWarning,
        )
        obj = obj.to(device.torch_device)

    comm = sanitize_comm(comm)
    gshape = tuple(obj.shape)
    lshape = gshape
    gnnz = obj.values().shape[0]

    compressed_indices = obj.crow_indices() if orientation == "row" else obj.ccol_indices()
    element_indices = obj.col_indices() if orientation == "row" else obj.row_indices()
    cls = DCSR_matrix if orientation == "row" else DCSC_matrix

    if (split == 0 and orientation == "row") or (split == 1 and orientation == "col"):
        start, end = comm.chunk(gshape, split, sparse=True)

        # Find the starting and ending indices for
        # element_indices and values tensors for this process
        indices_start = compressed_indices[start]
        indices_end = compressed_indices[end]

        # Slice the data belonging to this process
        data = obj.values()[indices_start:indices_end]
        # start:(end + 1) because indptr is of size (n + 1) for array with n rows
        indptr = compressed_indices[start : end + 1]

        indices = element_indices[indices_start:indices_end]

        indptr = indptr - indptr[0]

        lshape = list(lshape)
        lshape[split] = end - start
        lshape = tuple(lshape)

    elif split is not None:
        raise ValueError(f"Split axis {split} not supported for class {cls.__name__}")

    elif (is_split == 0 and orientation == "row") or (is_split == 1 and orientation == "col"):
        # Check whether the distributed data matches in
        # all dimensions other than axis 0
        neighbour_shape = np.array(gshape)
        lshape = np.array(lshape)

        if comm.rank < comm.size - 1:
            comm.Isend(lshape, dest=comm.rank + 1)
        if comm.rank != 0:
            # Dont have to check whether the number of dimensions are same since
            # both torch.sparse_csr_tensor and scipy.sparse.csr_matrix are 2D only

            # check whether the individual shape elements match
            comm.Recv(neighbour_shape, source=comm.rank - 1)
            for i in range(len(lshape)):
                if i == is_split:
                    continue
                elif lshape[i] != neighbour_shape[i]:
                    neighbour_shape[is_split] = np.iinfo(neighbour_shape.dtype).min

        lshape = tuple(lshape)

        # sum up the elements along the split dimension
        reduction_buffer = np.array(neighbour_shape[is_split])
        # To check if any process has found that its neighbour
        # does not match with itself in shape
        comm.Allreduce(MPI.IN_PLACE, reduction_buffer, MPI.MIN)
        if reduction_buffer < 0:
            raise ValueError(
                f"Unable to construct {cls.__name__}. Local data slices have inconsistent shapes or dimensions."
            )

        data = obj.values()
        indptr = compressed_indices
        indices = element_indices

        # Calculate gshape
        gshape_split = torch.tensor(gshape[is_split])
        comm.Allreduce(MPI.IN_PLACE, gshape_split, MPI.SUM)
        gshape = list(gshape)
        gshape[is_split] = gshape_split.item()
        gshape = tuple(gshape)

        # Calculate gnnz
        lnnz = data.shape[0]
        gnnz_buffer = torch.tensor(lnnz)
        comm.Allreduce(MPI.IN_PLACE, gnnz_buffer, MPI.SUM)
        gnnz = gnnz_buffer.item()

        split = is_split

    elif is_split is not None:
        raise ValueError(f"Split axis {is_split} not supported for class {cls.__name__}")

    else:  # split is None and is_split is None
        data = obj.values()
        indptr = compressed_indices
        indices = element_indices

    sparse_array = torch_class(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data,
        size=lshape,
        dtype=dtype.torch_type(),
        device=device.torch_device,
    )

    return cls(
        array=sparse_array,
        gnnz=gnnz,
        gshape=gshape,
        dtype=dtype,
        split=split,
        device=device,
        comm=comm,
        balanced=True,
    )
