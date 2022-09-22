"""Provides high-level Dcsr_matrix initialization functions"""

import torch
import numpy as np
from scipy.sparse import csr_matrix as scipy_csr_matrix

from typing import Optional, Type, Union
import warnings

from ..core.communication import MPI, sanitize_comm, Communication
from ..core.devices import Device
from ..core.types import datatype

from .dcsr_matrix import Dcsr_matrix

from ..core import devices
from ..core import types

__all__ = [
    "sparse_csr_matrix",
]


def sparse_csr_matrix(
    obj: Union[torch.Tensor, scipy_csr_matrix],
    dtype: Optional[Type[datatype]] = None,
    copy: bool = True,
    ndmin: int = 0,
    order: str = "C",
    split: Optional[int] = None,
    is_split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> Dcsr_matrix:
    """
    Create a :class:`~heat.sparse.Dcsr_matrix`.

    Parameters
    ----------
    obj : :class:`torch.Tensor` (layout ==> torch.sparse_csr) or :class:`scipy.sparse.csr_matrix`
        Sparse tensor that needs to be distributed
    dtype : datatype, optional
        The desired data-type for the sparse matrix. If not given, then the type will be determined as the minimum type required
        to hold the objects in the sequence. This argument can only be used to ‘upcast’ the array. For downcasting, use
        the :func:`~heat.sparse.dcsr_matrix.astype` method.
    split : int or None, optional
        The axis along which the passed array content ``obj`` is split and distributed in memory. Mutually exclusive
        with ``is_split``. Dcsr_matrix only supports distribution along axis 0.
    copy : bool, optional
        TODO
        If ``True`` (default), then the object is copied. Otherwise, a copy will only be made if obj is a nested
        sequence or if a copy is needed to satisfy any of the other requirements, e.g. ``dtype``.
    ndmin : int, optional
        TODO
        Specifies the minimum number of dimensions that the resulting array should have. Ones will, if needed, be
        attached to the shape if ``ndim > 0`` and prefaced in case of ``ndim < 0`` to meet the requirement.
    order: str, optional
        TODO
        Options: ``'C'`` or ``'F'``. Specifies the memory layout of the newly created array. Default is ``order='C'``,
        meaning the array will be stored in row-major order (C-like). If ``order=‘F’``, the array will be stored in
        column-major order (Fortran-like).
    is_split : int or None, optional
        Specifies the axis along which the local data portions, passed in obj, are split across all machines. Useful for
        interfacing with other distributed-memory code. The shape of the global array is automatically inferred.
        Mutually exclusive with ``split``. Dcsr_matrix only supports distribution along axis 0.
    device : str or Device, optional
        Specifies the :class:`~heat.core.devices.Device` the array shall be allocated on (i.e. globally set default
        device).
    comm : Communication, optional
        Handle to the nodes holding distributed array chunks.

    Raises
    ------
    NotImplementedError
        If split parameter is not one of 0 or None.
    NotImplementedError
        If is_split parameter is not one of 0 or None.

    Examples
    --------
    Create a :class:`~heat.sparse.Dcsr_matrix` from :class:`torch.Tensor` (layout ==> torch.sparse_csr)
    >>> indptr = torch.tensor([0, 2, 3, 6])
    >>> indices = torch.tensor([0, 2, 2, 0, 1, 2])
    >>> data = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
    >>> torch_sparse_csr = torch.sparse_csr_tensor(indptr, indices, data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(torch_sparse_csr, split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6]), indices: tensor([0, 2, 2, 0, 1, 2]), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.Dcsr_matrix` from :class:`scipy.sparse.csr_matrix`
    >>> scipy_sparse_csr = scipy.sparse.csr_matrix((data, indices, indptr))
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(scipy_sparse_csr, split=0)
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3, 6], dtype=torch.int32), indices: tensor([0, 2, 2, 0, 1, 2], dtype=torch.int32), data: tensor([1., 2., 3., 4., 5., 6.]), dtype=ht.float32, device=cpu:0, split=0)

    Create a :class:`~heat.sparse.Dcsr_matrix` using data that is already distributed (with `is_split`)
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
    """
    # sanitize the data type
    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    # sanitize device
    if device is not None:
        device = devices.sanitize_device(device)

    # Convert input into torch.Tensor (layout ==> torch.sparse_csr)
    if isinstance(obj, scipy_csr_matrix):
        obj = torch.sparse_csr_tensor(
            obj.indptr,
            obj.indices,
            obj.data,
            device=device.torch_device if device is not None else devices.get_device().torch_device,
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
            "Array 'obj' is not on device '{}'. It will be moved to it.".format(device), UserWarning
        )
        obj = obj.to(device.torch_device)

    # For now, assuming the obj is a torch.Tensor (layout ==> torch.sparse_csr)
    comm = sanitize_comm(comm)
    gshape = tuple(obj.shape)
    lshape = gshape
    gnnz = obj.values().shape[0]

    if split == 0:
        start, end = comm.chunk(gshape, split, sparse=True)

        # Find the starting and ending indices for
        # col_indices and values tensors for this process
        indicesStart = obj.crow_indices()[start]
        indicesEnd = obj.crow_indices()[end]

        # Slice the data belonging to this process
        data = obj.values()[indicesStart:indicesEnd]
        # start:(end + 1) because indptr is of size (n + 1) for array with n rows
        indptr = obj.crow_indices()[start : end + 1]
        indices = obj.col_indices()[indicesStart:indicesEnd]

        indptr = indptr - indptr[0]

        lshape = list(lshape)
        lshape[split] = end - start
        lshape = tuple(lshape)

    elif split is not None:
        raise NotImplementedError("Not implemented for other splitting-axes")

    elif is_split == 0:
        # Check whether the distributed data matches in
        # all dimensions other than axis 0
        neighbour_shape = np.array(gshape)
        lshape = np.array(lshape)

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
                neighbour_shape[is_split] = np.iinfo(neighbour_shape.dtype).min
            else:
                # check whether the individual shape elements match
                comm.Recv(neighbour_shape, source=comm.rank - 1)
                for i in range(length):
                    if i == is_split:
                        continue
                    elif lshape[i] != neighbour_shape[i] and lshape[i] - 1 != neighbour_shape[i]:
                        neighbour_shape[is_split] = np.iinfo(neighbour_shape.dtype).min

        lshape = tuple(lshape)

        # sum up the elements along the split dimension
        reduction_buffer = np.array(neighbour_shape[is_split])
        comm.Allreduce(MPI.IN_PLACE, reduction_buffer, MPI.SUM)
        if reduction_buffer < 0:
            raise ValueError("unable to construct tensor, shape of local data chunk does not match")

        data = obj.values()
        indptr = obj.crow_indices()
        indices = obj.col_indices()

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
        raise NotImplementedError("Not implemented for other splitting-axes")

    else:  # split is None and is_split is None
        data = obj.values()
        indptr = obj.crow_indices()
        indices = obj.col_indices()

    sparse_array = torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data,
        size=lshape,
        dtype=dtype.torch_type(),
        device=device.torch_device if device is not None else devices.get_device().torch_device,
    )

    return Dcsr_matrix(
        array=sparse_array,
        gnnz=gnnz,
        gshape=gshape,
        dtype=dtype,
        split=split,
        device=device,
        comm=comm,
        balanced=True,
    )
