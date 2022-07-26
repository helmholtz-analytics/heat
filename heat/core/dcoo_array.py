"""Provides HeAT's core data structure, the DNDcoo_array, a distributed n-dimensional coo sparse array"""
from __future__ import annotations

import math
import numpy as np
import torch
import warnings
import heat as ht

from inspect import stack
from mpi4py import MPI
from pathlib import Path
from typing import List, Union, Tuple, TypeVar, Optional

# from .coo_matrix import DNDcoo_array

__all__ = ["Dcoo_array"]

Communication = TypeVar("Communication")


def indices(gshape, obj, split, comm):
    """
    Returns the indices of the ob if dense
    """
    start, _, _ = comm.chunk(gshape, split=split)
    t_indices = obj.coalesce().indices()
    t_indices[split] += start
    global_indices = ht.array(t_indices, is_split=1)
    return global_indices


class Dcoo_array:
    """
    Distributed N-Dimentional coo_sparse tensor. It follows scipy
    coo_array attributes.

    Parameters
    ----------
    array : torch.Tensor
        Local array elements
    gshape : Tuple[int,...]
        The global shape of the sparse array
    dtype : datatype
        The datatype of the sparse array
    split : int or None
        The axis on which the sparse array is divided between processes
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    balanced: bool or None
        Describes whether the data are evenly distributed across processes.
        If this information is not available (``self.balanced is None``), it
        can be gathered via the :func:`is_balanced()` method (requires communication).
    gnnz: int or None
        Number of non-zero elements of the sparse array
    """

    def __init__(
        self,
        array: torch.sparse_coo_tensor,
        gshape: Tuple[int, ...],
        dtype: datatype,
        split: Union[int, None],
        device: Device,
        comm: Communication,
        balanced: bool,
        gnnz: int,
    ):
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm
        self.__balanced = balanced
        self.__lnnz = array._nnz()
        self.__gnnz = gnnz
        self.__indices = indices(gshape, array, split, comm)
        self.__lindices = array.coalesce().indices()
        # TODO: indices need to include explicit zeros
        # create
        # .coalesce()
        # self.__indices = array.nonzero(as_tuple=True)
        # print(self.__indices)
        # print("here")
        # self.__data = array[self.__indices]
        # print(self.__data)
        self.has_canonical_format = True

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
    def indices(self) -> Tuple:
        """
        Global indices of the ``coo_array``
        """
        return self.__indices

    @property
    def lindices(self) -> Tuple:
        """
        Global indices of the ``coo_array``
        """
        return self.__lindices

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``coo_array``
        """
        return len(self.__gshape)

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
    def gnnz(self) -> int:
        """
        Number of global non-zero elemnents of the ``coo_array``
        """
        return self.__gnnz

    @property
    def dtype(self):
        """
        The :class:`~heat.core.types.datatype` of the ``coo_array``
        """
        return self.__dtype

    @property
    def row(self):
        """
        COO format row index array of the array
        """
        return self.__indices[0]

    @property
    def col(self):
        """
        COO format col index array of the array
        """
        return self.__indices[1]

    @property
    def split(self) -> int:
        """
        Returns the axis on which the ``coo_array`` is split
        """
        return self.__split


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

# from . import coo_array

from .devices import Device
from .stride_tricks import sanitize_axis
from .types import datatype, canonical_heat_type
