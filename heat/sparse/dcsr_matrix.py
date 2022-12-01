"""Provides DCSR_matrix, a distributed compressed sparse row matrix"""
from __future__ import annotations

import torch
from mpi4py import MPI
from typing import Union, Tuple, TypeVar

from ..core.devices import Device
from ..core.dndarray import DNDarray
from ..core.factories import array
from ..core.types import datatype, canonical_heat_type

__all__ = ["DCSR_matrix"]

Communication = TypeVar("Communication")


class DCSR_matrix:
    """
    Distributed Compressed Sparse Row Matrix. It is composed of
    PyTorch sparse_csr_tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor (layout ==> torch.sparse_csr)
        Local sparse array
    gnnz: int
        Total number of non-zero elements across all processes
    gshape : Tuple[int,...]
        The global shape of the array
    dtype : datatype
        The datatype of the array
    split : int or None
        If split is not None, it denotes the axis on which the array is divided between processes.
        DCSR_matrix only supports distribution along axis 0.
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    balanced: bool or None
        Describes whether the data are evenly distributed across processes.
    """

    def __init__(
        self,
        array: torch.Tensor,
        gnnz: int,
        gshape: Tuple[int, ...],
        dtype: datatype,
        split: Union[int, None],
        device: Device,
        comm: Communication,
        balanced: bool,
    ):
        self.__array = array
        self.__gnnz = gnnz
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm
        self.__balanced = balanced

    def global_indptr(self) -> DNDarray:
        """
        Global indptr of the ``DCSR_matrix`` as a ``DNDarray``
        """
        if self.split is None:
            raise ValueError("This method works only for distributed matrices")

        # Need to know the number of non-zero elements
        # in the processes with lesser rank
        all_nnz = torch.zeros(self.comm.size + 1, device=self.device.torch_device)

        # Each process must drop their nnz in index = rank + 1
        all_nnz[self.comm.rank + 1] = self.lnnz
        self.comm.Allreduce(MPI.IN_PLACE, all_nnz, MPI.SUM)

        # Build prefix array out of all the nnz
        all_nnz = torch.cumsum(all_nnz, dim=0)

        global_indptr = self.lindptr + int(all_nnz[self.comm.rank])

        # Remove the (n+1) the element from all the processes except last
        if self.comm.rank != self.comm.size - 1:
            global_indptr = global_indptr[:-1]

        # NOTE: indptr might be unbalanced in distribution but should not be self balanced
        return array(
            global_indptr,
            dtype=self.lindptr.dtype,
            device=self.device,
            comm=self.comm,
            is_split=self.split,
        )

    @property
    def balanced(self) -> bool:
        """
        Boolean value indicating if the DCSR_matrix is balanced between the MPI processes
        """
        return self.__balanced

    @property
    def comm(self) -> Communication:
        """
        The :class:`~heat.core.communication.Communication` of the ``DCSR_matrix``
        """
        return self.__comm

    @property
    def device(self) -> Device:
        """
        The :class:`~heat.core.devices.Device` of the ``DCSR_matrix``
        """
        return self.__device

    @property
    def larray(self) -> torch.Tensor:
        """
        Local data of the ``DCSR_matrix``
        """
        return self.__array

    @property
    def data(self) -> torch.Tensor:
        """
        Global data of the ``DCSR_matrix``
        """
        if self.split is None:
            return self.ldata

        data_buffer = torch.zeros(
            size=(self.gnnz,), dtype=self.dtype.torch_type(), device=self.device.torch_device
        )
        counts, displs = self.counts_displs_nnz()
        self.comm.Allgatherv(self.ldata, (data_buffer, counts, displs))
        return data_buffer

    @property
    def gdata(self) -> torch.Tensor:
        """
        Global data of the ``DCSR_matrix``
        """
        return self.data

    @property
    def ldata(self) -> torch.Tensor:
        """
        Local data of the ``DCSR_matrix``
        """
        return self.__array.values()

    @property
    def indptr(self) -> torch.Tensor:
        """
        Global indptr of the ``DCSR_matrix``
        """
        if self.split is None:
            return self.lindptr

        return self.global_indptr().resplit(axis=None).larray

    @property
    def gindptr(self) -> torch.Tensor:
        """
        Global indptr of the ``DCSR_matrix``
        """
        return self.indptr

    @property
    def lindptr(self) -> torch.Tensor:
        """
        Local indptr of the ``DCSR_matrix``
        """
        return self.__array.crow_indices()

    @property
    def indices(self) -> torch.Tensor:
        """
        Global indices of the ``DCSR_matrix``
        """
        if self.split is None:
            return self.lindices

        indices_buffer = torch.zeros(
            size=(self.gnnz,), dtype=self.lindices.dtype, device=self.device.torch_device
        )
        counts, displs = self.counts_displs_nnz()
        self.comm.Allgatherv(self.lindices, (indices_buffer, counts, displs))
        return indices_buffer

    @property
    def gindices(self) -> torch.Tensor:
        """
        Global indices of the ``DCSR_matrix``
        """
        return self.indices

    @property
    def lindices(self) -> torch.Tensor:
        """
        Local indices of the ``DCSR_matrix``
        """
        return self.__array.col_indices()

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``DCSR_matrix``
        """
        return len(self.__gshape)

    @property
    def nnz(self) -> int:
        """
        Total number of non-zero elements of the ``DCSR_matrix``
        """
        return self.__gnnz

    @property
    def gnnz(self) -> int:
        """
        Total number of non-zero elements of the ``DCSR_matrix``
        """
        return self.nnz

    @property
    def lnnz(self) -> int:
        """
        Number of non-zero elements on the local process of the ``DCSR_matrix``
        """
        return self.__array._nnz()

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Global shape of the ``DCSR_matrix``
        """
        return self.__gshape

    @property
    def gshape(self) -> Tuple[int, ...]:
        """
        Global shape of the ``DCSR_matrix``
        """
        return self.shape

    @property
    def lshape(self) -> Tuple[int, ...]:
        """
        Local shape of the ``DCSR_matrix``
        """
        return tuple(self.__array.size())

    @property
    def dtype(self):
        """
        The :class:`~heat.core.types.datatype` of the ``DCSR_matrix``
        """
        return self.__dtype

    @property
    def split(self) -> int:
        """
        Returns the axis on which the ``DCSR_matrix`` is split
        """
        return self.__split

    def is_distributed(self) -> bool:
        """
        Determines whether the data of this ``DCSR_matrix`` is distributed across multiple processes.
        """
        return self.split is not None and self.comm.is_distributed()

    def counts_displs_nnz(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Returns actual counts (number of non-zero items per process) and displacements (offsets) of the DCSR_matrix.
        Does not assume load balance.
        """
        if self.split is not None:
            counts = torch.zeros(self.comm.size)
            counts[self.comm.rank] = self.lnnz
            self.comm.Allreduce(MPI.IN_PLACE, counts, MPI.SUM)
            displs = [0] + torch.cumsum(counts, dim=0)[:-1].tolist()
            return tuple(counts.tolist()), tuple(displs)
        else:
            raise ValueError(
                "Non-distributed DCSR_matrix. Cannot calculate counts and displacements."
            )

    def astype(self, dtype, copy=True) -> DCSR_matrix:
        """
        Returns a casted version of this matrix.
        Casted matrix is a new matrix of the same shape but with given type of this matrix. If copy is ``True``, the
        same matrix is returned instead.

        Parameters
        ----------
        dtype : datatype
            HeAT type to which the matrix is cast
        copy : bool, optional
            By default the operation returns a copy of this matrix. If copy is set to ``False`` the cast is performed
            in-place and this matrix is returned
        """
        dtype = canonical_heat_type(dtype)
        casted_matrix = self.__array.type(dtype.torch_type())
        if copy:
            return DCSR_matrix(
                casted_matrix,
                self.gnnz,
                self.gshape,
                dtype,
                self.split,
                self.device,
                self.comm,
                self.balanced,
            )

        self.__array = casted_matrix
        self.__dtype = dtype

        return self

    def __repr__(self) -> str:
        """
        Computes a printable representation of the passed DCSR_matrix.
        """
        print_string = (
            f"(indptr: {self.indptr}, indices: {self.indices}, data: {self.data}, "
            f"dtype=ht.{self.dtype.__name__}, device={self.device}, split={self.split})"
        )

        # Check has to happen after generating string because
        # generation of string invokes functions that require
        # participation from all processes
        if self.comm.rank != 0:
            return ""
        return print_string
