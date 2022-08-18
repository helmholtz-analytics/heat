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


# Test with 2 processes
class TestSparse(TestCase):
    def test_split(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        t = torch.sparse_csr_tensor(
            torch.tensor(indptr, dtype=torch.int64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(data),
        )
        t_sparse = ht.sparse.sparse_csr_matrix(t, split=0)

        sparse_data = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        sparse_indptr = [torch.Tensor([0, 2, 3]), torch.Tensor([0, 3])]
        sparse_indices = torch.Tensor([[0, 2, 2], [0, 1, 2]])
        sparse_lnnz = torch.Tensor([3, 3])

        assert t_sparse.ldata.eq(sparse_data[rank]).all()
        assert t_sparse.lindptr.eq(sparse_indptr[rank]).all()
        assert t_sparse.lindices.eq(sparse_indices[rank]).all()
        assert t_sparse.lnnz == sparse_lnnz[rank]
        assert t_sparse.gnnz == len(data)

        print(f"Rank: {rank} Passed tests: split = 0")

    def test_conversion(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        t = torch.sparse_csr_tensor(
            torch.tensor(indptr, dtype=torch.int64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(data),
        )
        s = csr_matrix((data, indices, indptr), shape=(3, 3))
        assert (ht.sparse.sparse_csr_matrix(t).ldata == ht.sparse.sparse_csr_matrix(s).ldata).all()
        assert (
            ht.sparse.sparse_csr_matrix(t).lindptr == ht.sparse.sparse_csr_matrix(s).lindptr
        ).all()
        assert (
            ht.sparse.sparse_csr_matrix(t).lindices == ht.sparse.sparse_csr_matrix(s).lindices
        ).all()

        print(f"Rank: {rank} Passed tests: conversion of scipy.sparse to torch.sparse")

    def test_is_split(self):
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
        assert global_indptr.eq(sparse_global_indptr[rank]).all()

        # print(f'''Rank: {rank}
        #         data: {t_sparse.ldata}
        #         indptr: {t_sparse.lindptr}
        #         global_indptr: {t_sparse.global_indptr()}
        #         indices: {t_sparse.lindices}''')
        print(f"Rank: {rank} Passed test: is_split = 0")
