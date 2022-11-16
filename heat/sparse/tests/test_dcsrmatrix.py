import heat as ht
import torch

from heat.core.tests.test_suites.basic_test import TestCase

from typing import Tuple


class TestDCSR_matrix(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestDCSR_matrix, cls).setUpClass()
        """
        A = [[0, 0, 1, 0, 2]
             [0, 0, 0, 0, 0]
             [0, 3, 0, 0, 0]
             [4, 0, 0, 5, 0]
             [0, 0, 0, 0, 6]]
        """
        cls.ref_indptr = torch.tensor([0, 2, 2, 3, 5, 6], dtype=torch.int)
        cls.ref_indices = torch.tensor([2, 4, 1, 0, 3, 4], dtype=torch.int)
        cls.ref_data = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
        cls.ref_torch_sparse_csr = torch.sparse_csr_tensor(
            cls.ref_indptr, cls.ref_indices, cls.ref_data
        )

        cls.world_size = ht.communication.MPI_WORLD.size
        cls.rank = ht.communication.MPI_WORLD.rank

    def test_larray(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertIsInstance(heat_sparse_csr.larray, torch.Tensor)
        self.assertEqual(heat_sparse_csr.larray.layout, torch.sparse_csr)
        self.assertEqual(tuple(heat_sparse_csr.larray.shape), heat_sparse_csr.lshape)
        self.assertEqual(tuple(heat_sparse_csr.larray.shape), heat_sparse_csr.gshape)

        # Distributed case
        if self.world_size > 1:
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)

            self.assertIsInstance(heat_sparse_csr.larray, torch.Tensor)
            self.assertEqual(heat_sparse_csr.larray.layout, torch.sparse_csr)
            self.assertEqual(tuple(heat_sparse_csr.larray.shape), heat_sparse_csr.lshape)
            self.assertNotEqual(tuple(heat_sparse_csr.larray.shape), heat_sparse_csr.gshape)

    def test_nnz(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertIsInstance(heat_sparse_csr.nnz, int)
        self.assertIsInstance(heat_sparse_csr.gnnz, int)
        self.assertIsInstance(heat_sparse_csr.lnnz, int)

        self.assertEqual(heat_sparse_csr.nnz, self.ref_torch_sparse_csr._nnz())
        self.assertEqual(heat_sparse_csr.nnz, heat_sparse_csr.gnnz)
        self.assertEqual(heat_sparse_csr.nnz, heat_sparse_csr.lnnz)

        # Distributed case
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)

        if self.world_size == 2:
            nnz_dist = [3, 3]
            self.assertEqual(heat_sparse_csr.nnz, self.ref_torch_sparse_csr._nnz())
            self.assertEqual(heat_sparse_csr.lnnz, nnz_dist[self.rank])

        if self.world_size == 3:
            nnz_dist = [2, 3, 1]
            self.assertEqual(heat_sparse_csr.nnz, self.ref_torch_sparse_csr._nnz())
            self.assertEqual(heat_sparse_csr.lnnz, nnz_dist[self.rank])

        # Number of processes > Number of rows
        if self.world_size == 6:
            nnz_dist = [2, 0, 1, 2, 1, 0]
            self.assertEqual(heat_sparse_csr.nnz, self.ref_torch_sparse_csr._nnz())
            self.assertEqual(heat_sparse_csr.lnnz, nnz_dist[self.rank])

    def test_shape(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertIsInstance(heat_sparse_csr.shape, Tuple)
        self.assertIsInstance(heat_sparse_csr.gshape, Tuple)
        self.assertIsInstance(heat_sparse_csr.lshape, Tuple)

        self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.shape, heat_sparse_csr.gshape)
        self.assertEqual(heat_sparse_csr.shape, heat_sparse_csr.lshape)

        # Distributed case
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)

        if self.world_size == 2:
            lshape_dist = [(3, 5), (2, 5)]

            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])

        if self.world_size == 3:
            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])

        # Number of processes > Number of rows
        if self.world_size == 6:
            lshape_dist = [(1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (0, 5)]

            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])

    def test_dtype(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)

    def test_data(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
        self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.gdata).all())
        self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.ldata).all())

        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
        if self.world_size == 2:
            data_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.gdata).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        if self.world_size == 3:
            data_dist = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]

            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.gdata).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        # Number of processes > Number of rows
        if self.world_size == 6:
            data_dist = [
                torch.tensor([1, 2]),
                torch.tensor([]),
                torch.tensor([3]),
                torch.tensor([4, 5]),
                torch.tensor([6]),
                torch.tensor([]),
            ]

            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.gdata).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

    def test_indices(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
        self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
        self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.lindices).all())

        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
        if self.world_size == 2:
            indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]

            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())

        if self.world_size == 3:
            indices_dist = [torch.tensor([2, 4]), torch.tensor([1, 0, 3]), torch.tensor([4])]

            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())

        # Number of processes > Number of rows
        if self.world_size == 6:
            indices_dist = [
                torch.tensor([2, 4]),
                torch.tensor([]),
                torch.tensor([1]),
                torch.tensor([0, 3]),
                torch.tensor([4]),
                torch.tensor([]),
            ]

            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())

    def test_indptr(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
        self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.lindptr).all())

        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
        if self.world_size == 2:
            indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]

            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())

        if self.world_size == 3:
            indptr_dist = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 3]), torch.tensor([0, 1])]

            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_dist = [
                torch.tensor([0, 2]),
                torch.tensor([0, 0]),
                torch.tensor([0, 1]),
                torch.tensor([0, 2]),
                torch.tensor([0, 1]),
                torch.tensor([0]),
            ]

            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())

    def test_astype(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        # check starting invariant
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)

        # check the copy case for uint8
        as_uint8 = heat_sparse_csr.astype(ht.uint8)
        self.assertIsInstance(as_uint8, ht.sparse.DCSR_matrix)
        self.assertEqual(as_uint8.dtype, ht.uint8)
        self.assertEqual(as_uint8.larray.dtype, torch.uint8)
        self.assertIsNot(as_uint8, heat_sparse_csr)

        # check the copy case for uint8
        as_float64 = heat_sparse_csr.astype(ht.float64, copy=False)
        self.assertIsInstance(as_float64, ht.sparse.DCSR_matrix)
        self.assertEqual(as_float64.dtype, ht.float64)
        self.assertEqual(as_float64.larray.dtype, torch.float64)
        self.assertIs(as_float64, heat_sparse_csr)
