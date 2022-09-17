"""
Arithmetic functions for Dcsr_matrices
"""
from __future__ import annotations

import torch

from .dcsr_matrix import Dcsr_matrix

from . import _operations

__all__ = [
    "sparse_add",
    "sparse_mul",
]


def sparse_add(t1: Dcsr_matrix, t2: Dcsr_matrix) -> Dcsr_matrix:
    """
    Element-wise addition of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.sparse.Dcsr_matrix`) whose elements are to be added
    as argument and returns a ``Dcsr_matrix`` containing the results of element-wise addition of ``t1`` and ``t2``.

    Parameters
    ----------
    t1: Dcsr_matrix
        The first operand involved in the addition
    t2: Dcsr_matrix
        The second operand involved in the addition

    Examples
    --------
    >>> import heat as ht
    >>> indptrs = [torch.tensor([0, 2, 3], dtype=torch.int),
                   torch.tensor([0, 3], dtype=torch.int)]
    >>> indices = [torch.tensor([0, 2, 2], dtype=torch.int),
                   torch.tensor([0, 1, 2], dtype=torch.int)]
    >>> data = [torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([4, 5, 6], dtype=torch.float)]
    >>> rank = ht.MPI_WORLD.rank
    >>> local_indptr = indptrs[rank]
    >>> local_indices = indices[rank]
    >>> local_data = data[rank]
    >>> local_torch_sparse_csr = torch.sparse_csr_tensor(local_indptr, local_indices, local_data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(local_torch_sparse_csr, is_split=0)
    >>> sum_sparse = heat_sparse_csr + heat_sparse_csr
        (or)
    >>> sum_sparse = ht.sparse.sparse_add(heat_sparse_csr, heat_sparse_csr)
    >>> sum_sparse
    (indptr: tensor([0, 2, 3, 6], dtype=torch.int32),
     indices: tensor([0, 2, 2, 0, 1, 2], dtype=torch.int32),
     data: tensor([ 2.,  4.,  6.,  8., 10., 12.]), dtype=ht.float32, device=cpu:0, split=0)
    >>> sum_sparse.todense()
    DNDarray([[ 2.,  0.,  4.],
              [ 0.,  0.,  6.],
              [ 8., 10., 12.]], dtype=ht.float32, device=cpu:0, split=0)
    """
    return _operations.__binary_op_sparse_csr(torch.add, t1, t2)


Dcsr_matrix.__add__ = lambda self, other: sparse_add(self, other)


def sparse_mul(t1: Dcsr_matrix, t2: Dcsr_matrix) -> Dcsr_matrix:
    """
    Element-wise multiplication (NOT matrix multiplication) of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.sparse.Dcsr_matrix`) whose elements are to be
    multiplied as argument.

    Parameters
    ----------
    t1: Dcsr_matrix
        The first operand involved in the multiplication
    t2: Dcsr_matrix
        The second operand involved in the multiplication

    Examples
    --------
    >>> import heat as ht
    >>> indptrs = [torch.tensor([0, 2, 3], dtype=torch.int),
                   torch.tensor([0, 3], dtype=torch.int)]
    >>> indices = [torch.tensor([0, 2, 2], dtype=torch.int),
                   torch.tensor([0, 1, 2], dtype=torch.int)]
    >>> data = [torch.tensor([1, 2, 3], dtype=torch.float),
                torch.tensor([4, 5, 6], dtype=torch.float)]
    >>> rank = ht.MPI_WORLD.rank
    >>> local_indptr = indptrs[rank]
    >>> local_indices = indices[rank]
    >>> local_data = data[rank]
    >>> local_torch_sparse_csr = torch.sparse_csr_tensor(local_indptr, local_indices, local_data)
    >>> heat_sparse_csr = ht.sparse.sparse_csr_matrix(local_torch_sparse_csr, is_split=0)
    >>> pdt_sparse = heat_sparse_csr * heat_sparse_csr
        (or)
    >>> pdt_sparse = ht.sparse.sparse_mul(heat_sparse_csr, heat_sparse_csr)
    >>> pdt_sparse
    (indptr: tensor([0, 2, 3, 6]),
     indices: tensor([0, 2, 2, 0, 1, 2]),
     data: tensor([ 1.,  4.,  9., 16., 25., 36.]), dtype=ht.float32, device=cpu:0, split=0)
    >>> pdt_sparse.todense()
    DNDarray([[ 1.,  0.,  4.],
              [ 0.,  0.,  9.],
              [16., 25., 36.]], dtype=ht.float32, device=cpu:0, split=0)
    """
    return _operations.__binary_op_sparse_csr(torch.mul, t1, t2)


Dcsr_matrix.__mul__ = lambda self, other: sparse_mul(self, other)
