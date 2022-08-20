"""Provides high-level Dcsr_matrix initialization functions"""

import torch
from scipy.sparse import csr_matrix as scipy_csr_matrix

from typing import Optional, Type

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
    obj: torch.sparse_csr_tensor,
    dtype: Optional[Type[datatype]] = None,
    copy: bool = True,
    ndmin: int = 0,
    order: str = "C",
    split: Optional[int] = None,
    is_split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
) -> Dcsr_matrix:

    # TODO:
    # Things I have not really paid attention to:
    #   1. copy
    #   2. ndim
    #   3. order
    #   4. balanced
    #   5. lshape

    # sanitize the data type
    if dtype is not None:
        dtype = types.canonical_heat_type(dtype)

    # sanitize device
    if device is not None:
        device = devices.sanitize_device(device)

    # Convert input into torch.sparse_csr_tensor
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

    # For now, assuming the obj is torch.sparse_csr_tensor
    comm = sanitize_comm(comm)
    gshape = tuple(obj.shape)
    lshape = gshape
    gnnz = obj.values().shape[0]

    if split == 0:
        start, end = comm.chunk(gshape, split, type="sparse")

        # Find the starting and ending indices for
        # col_indices and values tensors for this process
        indicesStart = obj.crow_indices()[start]
        indicesEnd = obj.crow_indices()[end]

        # Slice the data belonging to this process
        data = obj.values()[indicesStart:indicesEnd]
        # start:(end + 1) because indptr is of size (n + 1) for array with n rows
        indptr = obj.crow_indices()[start : end + 1]
        indices = obj.col_indices()[indicesStart:indicesEnd]

        lnnz = data.shape[0]
        indptr = indptr - indptr[0]

        lshape = list(lshape)
        lshape[split] = end - start
        lshape = tuple(lshape)

    elif split is not None:
        raise NotImplementedError("Not implemented for other splitting-axes")

    elif is_split == 0:
        data = obj.values()
        indptr = obj.crow_indices()
        indices = obj.col_indices()
        lnnz = data.shape[0]

        # Calculate gshape
        gshape_split = torch.tensor(gshape[is_split])
        comm.Allreduce(MPI.IN_PLACE, gshape_split, MPI.SUM)
        gshape = list(gshape)
        gshape[is_split] = gshape_split
        gshape = tuple(gshape)

        split = is_split

    elif is_split is not None:
        raise NotImplementedError("Not implemented for other splitting-axes")

    else:  # split is None and is_split is None
        data = obj.values()
        indptr = obj.crow_indices()
        indices = obj.col_indices()
        lnnz = gnnz

    sparse_array = torch.sparse_csr_tensor(
        indptr,
        indices,
        data,
        size=lshape,
        dtype=dtype.torch_type(),
        device=device.torch_device if device is not None else devices.get_device().torch_device,
    )

    return Dcsr_matrix(
        array=sparse_array,
        gnnz=gnnz,
        lnnz=lnnz,
        gshape=gshape,
        lshape=lshape,
        dtype=dtype,
        split=split,
        device=device,
        comm=comm,
        balanced=True,
    )
