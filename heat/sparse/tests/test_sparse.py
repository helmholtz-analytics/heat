import heat as ht
import torch
import numpy as np
from scipy.sparse import csr_matrix
from mpi4py import MPI

from heat.core.tests.test_suites.basic_test import TestCase

# Ignore warning from torch
import warnings

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class TestLasso(TestCase):
    def test_split():
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        t = torch.sparse_csr_tensor(
            torch.tensor(indptr, dtype=torch.int64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(data),
        )
        t_sparse = ht.sparse.sparse_csr_matrix(t, split=0)

        # Test with 2 processes
        sparse_data = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        sparse_indptr = [torch.Tensor([0, 2, 3]), torch.Tensor([0, 3])]
        sparse_indices = torch.Tensor([[0, 2, 2], [0, 1, 2]])
        sparse_lnnz = torch.Tensor([3, 3])

        assert t_sparse.data.larray.eq(sparse_data[rank]).all()
        assert t_sparse.indptr.larray.eq(sparse_indptr[rank]).all()
        assert t_sparse.indices.larray.eq(sparse_indices[rank]).all()
        assert t_sparse.lnnz == sparse_lnnz[rank]
        assert t_sparse.gnnz == len(data)

        print(f"Rank: {rank} Passed tests: split = 0")

    def test_conversion():
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        t = torch.sparse_csr_tensor(
            torch.tensor(indptr, dtype=torch.int64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(data),
        )
        s = csr_matrix((data, indices, indptr), shape=(3, 3))
        assert (ht.sparse.sparse_csr_matrix(t).data == ht.sparse.sparse_csr_matrix(s).data).all()
        assert (
            ht.sparse.sparse_csr_matrix(t).indptr == ht.sparse.sparse_csr_matrix(s).indptr
        ).all()
        assert (
            ht.sparse.sparse_csr_matrix(t).indices == ht.sparse.sparse_csr_matrix(s).indices
        ).all()

        print(f"Rank: {rank} Passed tests: conversion of scipy.sparse to torch.sparse")

    def test_is_split():
        sparse_data = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        sparse_indptr = [torch.Tensor([0, 2, 3]), torch.Tensor([0, 3])]
        sparse_indices = torch.Tensor([[0, 2, 2], [0, 1, 2]])
        sparse_global_indptr = [torch.Tensor([0, 2, 3]), torch.Tensor([3, 6])]

        t = torch.sparse_csr_tensor(
            torch.tensor(sparse_indptr[rank], dtype=torch.int64),
            torch.tensor(sparse_indices[rank], dtype=torch.int64),
            torch.tensor(sparse_data[rank]),
        )
        t_sparse = ht.sparse.sparse_csr_matrix(t, is_split=0)

        assert isinstance(t_sparse, ht.sparse.Dcsr_matrix)

        global_indptr = t_sparse.global_indptr()
        assert global_indptr.larray.eq(sparse_global_indptr[rank]).all()

        # print(f'''Rank: {rank}
        #         data: {t_sparse.data.larray}
        #         indptr: {t_sparse.indptr.larray}
        #         global_indptr: {t_sparse.global_indptr().larray}
        #         indices: {t_sparse.indices.larray}''')
        print(f"Rank: {rank} Passed test: is_split = 0")
