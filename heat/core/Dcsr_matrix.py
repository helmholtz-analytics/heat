from __future__ import annotations

import math
import numpy as np
import torch
import warnings

from inspect import stack
from mpi4py import MPI
from pathlib import Path
from typing import List, Union, Tuple, TypeVar, Optional

from heat.core.dndarray import DNDarray

__all__ = ["Dcsr_matrix"]

Communication = TypeVar("Communication")


class Dcsr_matrix:
    def __init__(
        self,
        data: DNDarray,
        indptr: DNDarray,
        indices: DNDarray,
        gnnz: int,
        lnnz: int,
        gshape: Tuple[int, ...],
        dtype: datatype,
        split: Union[int, None],
        device: Device,
        comm: Communication,
        balanced: bool,
    ):
        # TODO: Proper getters and setters for local and global members
        self.__data = data
        self.__indptr = indptr
        self.__indices = indices
        self.__gnnz = gnnz
        self.__lnnz = lnnz
        self.__gshape = gshape
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
        torch.cumsum(all_nnz, dim=0)

        global_indptr = self.indptr + int(all_nnz[self.comm.rank])

        return global_indptr

    @property
    def balanced(self) -> bool:
        """
        Boolean value indicating if the coo_array is balanced between the MPI processes
        """
        return self.__balanced

    @property
    def comm(self) -> Communication:
        """
        The :class:`~heat.core.communication.Communication` of the ``coo_array``
        """
        return self.__comm

    @property
    def device(self) -> Device:
        """
        The :class:`~heat.core.devices.Device` of the ``coo_array``
        """
        return self.__device

    @property
    def data(self) -> Tuple:
        """
        Global data of the ``coo_array``
        """
        return self.__data

    @property
    def ldata(self) -> Tuple:
        """
        Local data of the ``coo_array``
        """
        return self.__ldata

    @property
    def indptr(self) -> Tuple:
        """
        Global indptr of the ``coo_array``
        """
        return self.__indptr

    @property
    def indices(self) -> Tuple:
        """
        Global indices of the ``coo_array``
        """
        return self.__indices

    @property
    def lindptr(self) -> Tuple:
        """
        Local indptr of the ``coo_array``
        """
        return self.__lindptr

    @property
    def lindices(self) -> Tuple:
        """
        Local indices of the ``coo_array``
        """
        return self.__lindices

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``coo_array``
        """
        return len(self.__gshape)

    @property
    def gnnz(self) -> int:
        """
        Number of global non-zero elemnents of the ``coo_array``
        """
        return self.__gnnz

    @property
    def lnnz(self) -> int:
        """
        Number of non-zero elemnent on the local process of the ``coo_array``
        """
        return self.__lnnz

    @property
    def shape(self) -> int:
        """
        Global shape of the coo array ``coo_array``
        """
        return self.__gshape

    @property
    def dtype(self):
        """
        The :class:`~heat.core.types.datatype` of the ``coo_array``
        """
        return self.__dtype

    @property
    def split(self) -> int:
        """
        Returns the axis on which the ``coo_array`` is split
        """
        return self.__split


# HeAT imports at the end to break cyclic dependencies
from . import complex_math
from . import devices
from . import factories
from . import indexing
from . import linalg
from . import manipulations
from . import printing
from . import rounding
from . import sanitation
from . import statistics
from . import stride_tricks
from . import tiling

from .devices import Device
from .stride_tricks import sanitize_axis
from .types import datatype, canonical_heat_type
