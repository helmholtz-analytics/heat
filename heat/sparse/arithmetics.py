from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Union, Tuple

from .dcsr_matrix import Dcsr_matrix

from ..core.factories import array
from ..core.dndarray import DNDarray
from . import _operations

__all__ = [
    "sparse_add",
    "sparse_mul",
]


def sparse_add(t1: Union[Dcsr_matrix, float], t2: Union[Dcsr_matrix, float]) -> Dcsr_matrix:
    return _operations.__binary_op_sparse(torch.add, t1, t2)


Dcsr_matrix.__add__ = lambda self, other: sparse_add(self, other)


def sparse_mul(t1: Union[Dcsr_matrix, float], t2: Union[Dcsr_matrix, float]) -> Dcsr_matrix:
    return _operations.__binary_op_sparse(torch.mul, t1, t2)


Dcsr_matrix.__mul__ = lambda self, other: sparse_mul(self, other)

# def sparse_sub(t1: Union[Dcsr_matrix, float], t2: Union[Dcsr_matrix, float]) -> Dcsr_matrix:
#     return _operations.__binary_op_sparse(torch.sub, t1, t2)
# Dcsr_matrix.__sub__ = lambda self, other: sparse_sub(self, other)
