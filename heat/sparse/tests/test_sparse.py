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

        self.assertTrue(t_sparse.ldata.eq(sparse_data[rank]).all())
        self.assertTrue(t_sparse.lindptr.eq(sparse_indptr[rank]).all())
        self.assertTrue(t_sparse.lindices.eq(sparse_indices[rank]).all())
        self.assertTrue(t_sparse.lnnz == sparse_lnnz[rank])
        self.assertTrue(t_sparse.gnnz == len(data))

        # print(f"Rank: {rank} Passed tests: split = 0")

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
        self.assertTrue(
            (ht.sparse.sparse_csr_matrix(t).ldata == ht.sparse.sparse_csr_matrix(s).ldata).all()
        )
        self.assertTrue(
            (ht.sparse.sparse_csr_matrix(t).lindptr == ht.sparse.sparse_csr_matrix(s).lindptr).all()
        )
        self.assertTrue(
            (
                ht.sparse.sparse_csr_matrix(t).lindices == ht.sparse.sparse_csr_matrix(s).lindices
            ).all()
        )

        # print(f"Rank: {rank} Passed tests: conversion of scipy.sparse to torch.sparse")

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

        self.assertTrue(isinstance(t_sparse, ht.sparse.Dcsr_matrix))

        global_indptr = t_sparse.global_indptr()
        self.assertTrue(global_indptr.larray.eq(sparse_global_indptr[rank]).all())

        # print(f'''Rank: {rank}
        #         data: {t_sparse.ldata}
        #         indptr: {t_sparse.lindptr}
        #         global_indptr: {t_sparse.global_indptr()}
        #         indices: {t_sparse.lindices}''')
        # print(f"Rank: {rank} Passed test: is_split = 0")

    def test_to_dense(self):
        indptr = np.array([0, 1, 3, 6])
        indices = np.array([1, 0, 1, 0, 1, 2])
        data = np.array([2, 1, 2, 1, 2, 3])

        dense_expected = ht.array([[0, 2, 0], [1, 2, 0], [1, 2, 3]], split=0)

        # Test dense function output w/ split
        t = torch.sparse_csr_tensor(
            torch.tensor(indptr, dtype=torch.int64),
            torch.tensor(indices, dtype=torch.int64),
            torch.tensor(data),
        )
        t_sparse = ht.sparse.sparse_csr_matrix(t, split=0)
        dense = t_sparse.todense()
        self.assertTrue((dense_expected == dense).all())

        # Test w/o split
        t_sparse = ht.sparse.sparse_csr_matrix(t)
        dense = t_sparse.todense()
        self.assertTrue((dense_expected == dense).all())

        # Test with output buffer w/ split
        t_sparse = ht.sparse.sparse_csr_matrix(t, split=0)
        out = ht.empty(shape=t_sparse.shape, split=0, dtype=t_sparse.dtype)
        t_sparse.todense(out=out)
        self.assertTrue((dense_expected == out).all())

        # Test with output buffer w/o split
        t_sparse = ht.sparse.sparse_csr_matrix(t)
        out = ht.empty(shape=t_sparse.shape, dtype=t_sparse.dtype)
        t_sparse.todense(out=out)
        self.assertTrue((dense_expected == out).all())

        # Test gaurds for output shape mismatch
        with self.assertRaises(ValueError) as context:
            t_sparse = ht.sparse.sparse_csr_matrix(t, split=0)
            out = ht.empty(shape=(1, 1), split=0, dtype=t_sparse.dtype)
            t_sparse.todense(out=out)
        self.assertTrue("Shape of output buffer does not match" in str(context.exception))

        # Test gaurds for output split axis mismatch w/split = 0
        with self.assertRaises(ValueError) as context:
            t_sparse = ht.sparse.sparse_csr_matrix(t, split=0)
            out = ht.empty(shape=t_sparse.shape, dtype=t_sparse.dtype)
            t_sparse.todense(out=out)
        self.assertTrue("Split axis of output buffer does not match" in str(context.exception))

        # Test gaurds for output split axis mismatch w/split = None
        with self.assertRaises(ValueError) as context:
            t_sparse = ht.sparse.sparse_csr_matrix(t, split=None)
            out = ht.empty(shape=t_sparse.shape, split=0, dtype=t_sparse.dtype)
            t_sparse.todense(out=out)
        self.assertTrue("Split axis of output buffer does not match" in str(context.exception))

    def test_arithmetics(self):
        # TODO: Write tests for arithmetic operations
        pass
