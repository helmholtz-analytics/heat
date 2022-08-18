from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Union, Tuple

from .dcsr_matrix import Dcsr_matrix

from ..core.factories import array
from ..core.dndarray import DNDarray

__all__ = [
    "sparse_mul",
]

def sparse_mul(t1: Union[Dcsr_matrix, float], t2: Union[Dcsr_matrix, float]) -> Dcsr_matrix:
    # Works for split = 0, 0 or None, None
    # For split = None, 0 or 0, None, should we just split the other array along 0
    numRows = t1.indptr.shape[0] - 1
    colFinal = []
    dataFinal = []
    for row in range(numRows):
        start1, end1 = t1.indptr.larray[row], t1.indptr.larray[row + 1]
        start2, end2 = t2.indptr.larray[row], t2.indptr.larray[row + 1]

        col1 = t1.indices.larray[start1: end1]
        col2 = t2.indices.larray[start2: end2]
        
        common, col1_ind, col2_ind = np.intersect1d(col1, col2, return_indices=True)

        colFinal.append(common)
        dataFinal.append([t1.data.larray[start1 + x] * t2.data.larray[start2 + y] for (x, y) in zip(col1_ind, col2_ind)])

    indptr = [0]
    for row in range(numRows):
        nnz_row = indptr[row] + len(colFinal[row])
        indptr.append(nnz_row)
    indptr = array(indptr)

    data = [torch.tensor(x) for x in dataFinal]
    data = array(torch.concat(data))

    col = [torch.tensor(x) for x in colFinal]
    col = array(torch.concat(col))

    gnnz = len(col)
    lnnz = len(col)

    return Dcsr_matrix(
        data=data,
        indptr=indptr,
        indices=col,
        gnnz=gnnz,
        lnnz=lnnz,
        gshape=t1.shape,
        dtype=t1.dtype,
        split=t1.split,
        device=t1.device,
        comm=t1.comm,
        balanced=True,
    )

Dcsr_matrix.__mul__ = lambda self, other: sparse_mul(self, other)