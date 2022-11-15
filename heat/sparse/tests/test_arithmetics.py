import heat as ht
import torch
import numpy as np

import random

from heat.core.tests.test_suites.basic_test import TestCase


class TestArithmetics(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestArithmetics, cls).setUpClass()
        """
        A = [[0, 0, 1, 0, 2]
             [0, 0, 0, 0, 0]
             [0, 3, 0, 0, 0]
             [4, 0, 0, 5, 0]
             [0, 0, 0, 0, 6]]
        """
        cls.ref_indptr_A = torch.tensor([0, 2, 2, 3, 5, 6], dtype=torch.int)
        cls.ref_indices_A = torch.tensor([2, 4, 1, 0, 3, 4], dtype=torch.int)
        cls.ref_data_A = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
        cls.ref_torch_sparse_csr_A = torch.sparse_csr_tensor(
            cls.ref_indptr_A, cls.ref_indices_A, cls.ref_data_A
        )

        """
        B = [[2, 0, 0, 0, 3]
             [0, 0, 4, 0, 0]
             [0, 1, 0, 1, 0]
             [0, 0, 0, 0, 0]
             [0, 3, 0, 4, 0]]
        """
        cls.ref_indptr_B = torch.tensor([0, 2, 3, 5, 5, 7], dtype=torch.int)
        cls.ref_indices_B = torch.tensor([0, 4, 2, 1, 3, 1, 3], dtype=torch.int)
        cls.ref_data_B = torch.tensor([2, 3, 4, 1, 1, 3, 4], dtype=torch.float)
        cls.ref_torch_sparse_csr_B = torch.sparse_csr_tensor(
            cls.ref_indptr_B, cls.ref_indices_B, cls.ref_data_B
        )

        cls.world_size = ht.communication.MPI_WORLD.size
        cls.rank = ht.communication.MPI_WORLD.rank

        cls.scalar = np.array(random.randint(1, 100))
        if cls.world_size > 0:
            ht.communication.MPI_WORLD.Bcast(cls.scalar, root=0)
        cls.scalar = cls.scalar.item()

    def test_add(self):
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B)

        """
        C = [[2, 0, 1, 0, 5]
             [0, 0, 4, 0, 0]
             [0, 4, 0, 1, 0]
             [4, 0, 0, 5, 0]
             [0, 3, 0, 4, 6]]
        """
        indptr_C = torch.tensor([0, 3, 4, 6, 8, 11], dtype=torch.int)
        indices_C = torch.tensor([0, 2, 4, 2, 1, 3, 0, 3, 1, 3, 4], dtype=torch.int)
        data_C = torch.tensor([2, 1, 5, 4, 4, 1, 4, 5, 3, 4, 6], dtype=torch.float)

        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
        self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
        self.assertTrue((heat_sparse_csr_C.data == data_C).all())
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # Distributed case
        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [torch.tensor([0, 3, 4, 6]), torch.tensor([0, 2, 5])]
            indices_C_dist = [torch.tensor([0, 2, 4, 2, 1, 3]), torch.tensor([0, 3, 1, 3, 4])]
            data_C_dist = [torch.tensor([2, 1, 5, 4, 4, 1]), torch.tensor([4, 5, 3, 4, 6])]

            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [torch.tensor([0, 3, 4]), torch.tensor([0, 2, 4]), torch.tensor([0, 3])]
            indices_C_dist = [
                torch.tensor([0, 2, 4, 2]),
                torch.tensor([1, 3, 0, 3]),
                torch.tensor([1, 3, 4]),
            ]
            data_C_dist = [
                torch.tensor([2, 1, 5, 4]),
                torch.tensor([4, 1, 4, 5]),
                torch.tensor([3, 4, 6]),
            ]

            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # scalar
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A + self.scalar
        heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
        self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
        self.assertTrue((heat_sparse_csr_C.data == data_C).all())
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
            indices_C_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
            data_C_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
            data_C_dist = [data + self.scalar for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 3]), torch.tensor([0, 1])]
            indices_C_dist = [torch.tensor([2, 4]), torch.tensor([1, 0, 3]), torch.tensor([4])]
            data_C_dist = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]
            data_C_dist = [data + self.scalar for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        """
        [[1, 0]
         [0, 1]]
        """
        torch_sparse_csr_2x2 = torch.sparse_csr_tensor([0, 1, 2], [0, 1], [1, 1])
        heat_sparse_csr_2x2 = ht.sparse.sparse_csr_matrix(torch_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_2x2

        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.add(2, 3)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.add(heat_sparse_csr_2x2, torch_sparse_csr_2x2)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.add(torch_sparse_csr_2x2, heat_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = ht.sparse.add(heat_sparse_csr_2x2, heat_sparse_csr_A)

    def test_mul(self):
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B)

        """
        C = [[0, 0, 0, 0, 6]
             [0, 0, 0, 0, 0]
             [0, 3, 0, 0, 0]
             [0, 0, 0, 0, 0]
             [0, 0, 0, 0, 0]]
        """
        indptr_C = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int)
        indices_C = torch.tensor([4, 1], dtype=torch.int)
        data_C = torch.tensor([6, 3], dtype=torch.float)

        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
        self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
        self.assertTrue((heat_sparse_csr_C.data == data_C).all())
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # Distributed case
        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [torch.tensor([0, 1, 1, 2]), torch.tensor([0, 0, 0])]
            indices_C_dist = [torch.tensor([4, 1]), torch.tensor([])]
            data_C_dist = [torch.tensor([6, 3]), torch.tensor([])]

            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [torch.tensor([0, 1, 1]), torch.tensor([0, 1, 1]), torch.tensor([0, 0])]
            indices_C_dist = [torch.tensor([4]), torch.tensor([1]), torch.tensor([])]
            data_C_dist = [torch.tensor([6]), torch.tensor([3]), torch.tensor([])]

            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # scalar
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A * self.scalar
        heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
        self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
        self.assertTrue((heat_sparse_csr_C.data == data_C).all())
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
            indices_C_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
            data_C_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
            data_C_dist = [data * self.scalar for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 3]), torch.tensor([0, 1])]
            indices_C_dist = [torch.tensor([2, 4]), torch.tensor([1, 0, 3]), torch.tensor([4])]
            data_C_dist = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]
            data_C_dist = [data * self.scalar for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue((heat_sparse_csr_C.indptr == indptr_C).all())
            self.assertTrue((heat_sparse_csr_C.lindptr == indptr_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.indices == indices_C).all())
            self.assertTrue((heat_sparse_csr_C.lindices == indices_C_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr_C.data == data_C).all())
            self.assertTrue((heat_sparse_csr_C.ldata == data_C_dist[self.rank]).all())
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        """
        [[1, 0]
         [0, 1]]
        """
        torch_sparse_csr_2x2 = torch.sparse_csr_tensor([0, 1, 2], [0, 1], [1, 1])
        heat_sparse_csr_2x2 = ht.sparse.sparse_csr_matrix(torch_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_2x2

        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.mul(2, 3)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.mul(heat_sparse_csr_2x2, torch_sparse_csr_2x2)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.mul(torch_sparse_csr_2x2, heat_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = ht.sparse.mul(heat_sparse_csr_2x2, heat_sparse_csr_A)
