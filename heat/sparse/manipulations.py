"""
Manipulation operations for (potentially distributed) `Dcsr_matrix`.
"""
from __future__ import annotations

import torch

from heat.sparse.dcsr_matrix import Dcsr_matrix

from ..core.communication import MPI
from ..core.dndarray import DNDarray
from ..core.factories import empty

__all__ = [
    "todense",
]


def todense(sparse_matrix: Dcsr_matrix, order=None, out: DNDarray = None):
    # TODO: Things that have been ignored and to be implemented
    #   1. order

    if out is not None:
        if out.shape != sparse_matrix.shape:
            raise ValueError("Shape of output buffer does not match")

        if out.split != sparse_matrix.split:
            raise ValueError("Split axis of output buffer does not match")

    if out is None:
        out = empty(
            shape=sparse_matrix.shape,
            split=sparse_matrix.split,
            dtype=sparse_matrix.ldata.dtype,  # TODO: Change after fixing dtype in factory function
            device=sparse_matrix.device,
            comm=sparse_matrix.comm,
        )

    out.larray = sparse_matrix.larray.to_dense()
    return out


Dcsr_matrix.todense = lambda self, order=None, out=None: todense(self, order, out)
