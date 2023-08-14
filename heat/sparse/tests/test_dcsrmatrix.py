import unittest
from typing import Tuple

import torch

import heat as ht
from heat.core.tests.test_suites.basic_test import TestCase


@unittest.skipIf(
    int(torch.__version__.split(".")[0]) <= 1 and int(torch.__version__.split(".")[1]) < 12,
    f"ht.sparse requires torch >= 1.12. Found version {torch.__version__}.",
)
class TestDCSR_matrix(TestCase):
    @classmethod
    def setUpClass(self):
        super(TestDCSR_matrix, self).setUpClass()
        """
        A = [[0, 0, 1, 0, 2]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 0, 0]
            [4, 0, 0, 5, 0]
            [0, 0, 0, 0, 6]]
        """
        self.ref_indptr = torch.tensor(
            [0, 2, 2, 3, 5, 6], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_indices = torch.tensor(
            [2, 4, 1, 0, 3, 4], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_data = torch.tensor(
            [1, 2, 3, 4, 5, 6], dtype=torch.float, device=self.device.torch_device
        )
        self.ref_torch_sparse_csr = torch.sparse_csr_tensor(
            self.ref_indptr, self.ref_indices, self.ref_data, device=self.device.torch_device
        )

        self.world_size = ht.communication.MPI_WORLD.size
        self.rank = ht.communication.MPI_WORLD.rank

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
            data_dist = [[1, 2, 3], [4, 5, 6]]

            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.gdata).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            data_dist = [[1, 2], [3, 4, 5], [6]]

            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.gdata).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            data_dist = [[1, 2], [], [3], [4, 5], [6], []]

            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.data == heat_sparse_csr.gdata).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

    def test_indices(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
        self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
        self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.lindices).all())

        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
        if self.world_size == 2:
            indices_dist = [[2, 4, 1], [0, 3, 4]]

            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indices_dist = [[2, 4], [1, 0, 3], [4]]

            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indices_dist = [[2, 4], [], [1], [0, 3], [4], []]

            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.indices == heat_sparse_csr.gindices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

    def test_indptr(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
        self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.lindptr).all())

        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
        if self.world_size == 2:
            indptr_dist = [[0, 2, 2, 3], [0, 2, 3]]

            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]

            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_dist = [[0, 2], [0, 0], [0, 1], [0, 2], [0, 1], [0]]

            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.indptr == heat_sparse_csr.gindptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

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
