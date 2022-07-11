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
from .dndarray import DNDarray

__all__ = ["coo_matrix"]

Communication = TypeVar("Communication")

def indices(gshape, obj, split, comm) -> coo_matrix:
    start, _, _ = comm.chunk(gshape, split=split)
    t_indices = obj.coalesce().indices()
    t_indices[split] += start
    global_indices = ht.array(t_indices, is_split=1)
    return global_indices   

class coo_matrix():
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
        self.__indices = indices(gshape,array,split, comm)
        #TODO: indices need to include explicit zeros       
        # create 
        # .coalesce() 
        # self.__indices = array.nonzero(as_tuple=True)
        # print(self.__indices)
        # print("here")
        # self.__data = array[self.__indices]
        # print(self.__data)
        self.has_canonical_format = True 

    @property
    def indices(self) -> coo_matrix:
        return self.__indices

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``coo_array``
        """
        return len(self.__gshape)


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
from . import dndarray

from .devices import Device
from .stride_tricks import sanitize_axis
from .types import datatype, canonical_heat_type
