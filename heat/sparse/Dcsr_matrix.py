from __future__ import annotations

import torch

from mpi4py import MPI
from typing import Union, Tuple, TypeVar

from heat.core.dndarray import DNDarray
from heat.core.factories import array

__all__ = ["Dcsr_matrix"]

Communication = TypeVar("Communication")


class Dcsr_matrix:
    def __init__(
        self,
        array: torch.sparse_csr_tensor,
        gnnz: int,
        lnnz: int,
        gshape: Tuple[int, ...],
        lshape: Tuple[int, ...],
        dtype: datatype,
        split: Union[int, None],
        device: Device,
        comm: Communication,
        balanced: bool,
    ):
        # TODO: Proper getters and setters for local and global members
        self.__array = array
        self.__gnnz = gnnz
        self.__lnnz = lnnz
        self.__gshape = gshape
        self.__lshape = lshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm
        self.__balanced = balanced

    def global_indptr(self) -> DNDarray:
        # Need to know the number of non-zero elements
        # in the processes with lesser rank
        all_nnz = torch.zeros(self.comm.size + 1)

        # Each process must drop their nnz in index = rank + 1
        all_nnz[self.comm.rank + 1] = self.lnnz
        self.comm.Allreduce(MPI.IN_PLACE, all_nnz, MPI.SUM)

        # Build prefix array out of all the nnz
        all_nnz = torch.cumsum(all_nnz, dim=0)

        global_indptr = self.lindptr + int(all_nnz[self.comm.rank])

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
        Boolean value indicating if the Dcsr_matrix is balanced between the MPI processes
        """
        return self.__balanced

    @property
    def comm(self) -> Communication:
        """
        The :class:`~heat.core.communication.Communication` of the ``Dcsr_matrix``
        """
        return self.__comm

    @property
    def device(self) -> Device:
        """
        The :class:`~heat.core.devices.Device` of the ``Dcsr_matrix``
        """
        return self.__device

    @property
    def larray(self) -> torch.sparse_csr_tensor:
        """
        Local data of the ``Dcsr_matrix``
        """
        return self.__array

    @property
    def data(self) -> torch.Tensor:
        """
        Global data of the ``Dcsr_matrix``
        """
        if self.split is None:
            return self.ldata

        data_buffer = torch.zeros(size=(self.gnnz,), dtype=self.dtype.torch_type())
        counts, displs = self.counts_displs_nnz()
        self.comm.Allgatherv(self.ldata, (data_buffer, counts, displs))
        return data_buffer

    @property
    def ldata(self) -> torch.Tensor:
        """
        Local data of the ``Dcsr_matrix``
        """
        return self.__array.values()

    @property
    def indptr(self) -> torch.Tensor:
        """
        Global indptr of the ``Dcsr_matrix``
        """
        if self.split is None:
            return self.lindptr

        indptr_buffer = torch.zeros(size=(self.shape[0],), dtype=self.lindptr.dtype)
        local_gindptr = self.global_indptr().larray[:-1]  # Remove the (n+1)th element
        last_element = torch.tensor([self.gnnz])

        counts = torch.zeros(self.comm.size)
        counts[self.comm.rank] = self.lshape[0]
        self.comm.Allreduce(MPI.IN_PLACE, counts, MPI.SUM)
        displs = [0] + torch.cumsum(counts, dim=0)[:-1].tolist()
        counts = counts.tolist()

        self.comm.Allgatherv(local_gindptr, (indptr_buffer, counts, displs))

        indptr_buffer = torch.cat(
            (indptr_buffer, last_element)
        )  # Add the (n+1)th element to the final ind_ptr
        return indptr_buffer

    @property
    def lindptr(self) -> torch.Tensor:
        """
        Local indptr of the ``Dcsr_matrix``
        """
        return self.__array.crow_indices()

    @property
    def indices(self) -> torch.Tensor:
        """
        Global indices of the ``Dcsr_matrix``
        """
        if self.split is None:
            return self.lindices

        indices_buffer = torch.zeros(size=(self.gnnz,), dtype=self.lindices.dtype)
        counts, displs = self.counts_displs_nnz()
        self.comm.Allgatherv(self.lindices, (indices_buffer, counts, displs))
        return indices_buffer

    @property
    def lindices(self) -> torch.Tensor:
        """
        Local indices of the ``Dcsr_matrix``
        """
        return self.__array.col_indices()

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``Dcsr_matrix``
        """
        return len(self.__gshape)

    @property
    def gnnz(self) -> int:
        """
        Total number of non-zero elements of the ``Dcsr_matrix``
        """
        return self.__gnnz

    @property
    def lnnz(self) -> int:
        """
        Number of non-zero elements on the local process of the ``Dcsr_matrix``
        """
        return self.__lnnz

    @property
    def shape(self) -> int:
        """
        Global shape of the ``Dcsr_matrix``
        """
        return self.__gshape

    @property
    def lshape(self) -> int:
        """
        Local shape of the ``Dcsr_matrix``
        """
        return self.__lshape

    @property
    def dtype(self):
        """
        The :class:`~heat.core.types.datatype` of the ``Dcsr_matrix``
        """
        return self.__dtype

    @property
    def split(self) -> int:
        """
        Returns the axis on which the ``Dcsr_matrix`` is split
        """
        return self.__split

    def counts_displs_nnz(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Returns actual counts (number of non-zero items per process) and displacements (offsets) of the Dcsr_matrix.
        Does not assume load balance.
        """
        if self.split is not None:
            counts = torch.zeros(self.comm.size)
            counts[self.comm.rank] = self.lnnz
            self.comm.Allreduce(MPI.IN_PLACE, counts, MPI.SUM)
            displs = [0] + torch.cumsum(counts, dim=0)[:-1].tolist()
            return tuple(counts.tolist()), tuple(displs)
        else:
            raise ValueError("Non-distributed DNDarray. Cannot calculate counts and displacements.")


# HeAT imports at the end to break cyclic dependencies
from ..core import complex_math
from ..core import devices
from ..core import factories
from ..core import indexing
from ..core import linalg
from ..core import manipulations
from ..core import printing
from ..core import rounding
from ..core import sanitation
from ..core import statistics
from ..core import stride_tricks
from ..core import tiling

from ..core.devices import Device
from ..core.stride_tricks import sanitize_axis
from ..core.types import datatype, canonical_heat_type
