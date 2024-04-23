import unittest
import heat as ht
import torch

from heat.core.tests.test_suites.basic_test import TestCase

from typing import Tuple


@unittest.skipIf(
    int(torch.__version__.split(".")[0]) <= 1 and int(torch.__version__.split(".")[1]) < 12,
    f"ht.sparse requires torch >= 1.12. Found version {torch.__version__}.",
)
class TestDCSC_matrix(TestCase):
    @classmethod
    def setUpClass(self):
        super(TestDCSC_matrix, self).setUpClass()
        """
        A = [[0, 0, 1, 0, 2]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 0, 0]
            [4, 0, 0, 5, 0]
            [0, 0, 0, 0, 6]]
        """
        self.ref_indptr = torch.tensor(
            [0, 1, 2, 3, 4, 6], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_indices = torch.tensor(
            [3, 2, 0, 3, 0, 4], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_data = torch.tensor(
            [4, 3, 1, 5, 2, 6], dtype=torch.float, device=self.device.torch_device
        )
        self.ref_torch_sparse_csc = torch.sparse_csc_tensor(
            self.ref_indptr, self.ref_indices, self.ref_data, device=self.device.torch_device
        )

        self.world_size = ht.communication.MPI_WORLD.size
        self.rank = ht.communication.MPI_WORLD.rank

    def test_larray(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)

        self.assertIsInstance(heat_sparse_csc.larray, torch.Tensor)
        self.assertEqual(heat_sparse_csc.larray.layout, torch.sparse_csc)
        self.assertEqual(tuple(heat_sparse_csc.larray.shape), heat_sparse_csc.lshape)
        self.assertEqual(tuple(heat_sparse_csc.larray.shape), heat_sparse_csc.gshape)

        # Distributed case
        if self.world_size > 1:
            heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc, split=1)

            self.assertIsInstance(heat_sparse_csc.larray, torch.Tensor)
            self.assertEqual(heat_sparse_csc.larray.layout, torch.sparse_csc)
            self.assertEqual(tuple(heat_sparse_csc.larray.shape), heat_sparse_csc.lshape)
            self.assertNotEqual(tuple(heat_sparse_csc.larray.shape), heat_sparse_csc.gshape)

    def test_nnz(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)

        self.assertIsInstance(heat_sparse_csc.nnz, int)
        self.assertIsInstance(heat_sparse_csc.gnnz, int)
        self.assertIsInstance(heat_sparse_csc.lnnz, int)

        self.assertEqual(heat_sparse_csc.nnz, self.ref_torch_sparse_csc._nnz())
        self.assertEqual(heat_sparse_csc.nnz, heat_sparse_csc.gnnz)
        self.assertEqual(heat_sparse_csc.nnz, heat_sparse_csc.lnnz)

        # Distributed case
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc, split=1)

        if self.world_size == 2:
            nnz_dist = [3, 3]
            self.assertEqual(heat_sparse_csc.nnz, self.ref_torch_sparse_csc._nnz())
            self.assertEqual(heat_sparse_csc.lnnz, nnz_dist[self.rank])

        if self.world_size == 3:
            nnz_dist = [2, 2, 2]
            self.assertEqual(heat_sparse_csc.nnz, self.ref_torch_sparse_csc._nnz())
            self.assertEqual(heat_sparse_csc.lnnz, nnz_dist[self.rank])

        # Number of processes > Number of rows
        if self.world_size == 6:
            nnz_dist = [1, 1, 1, 1, 2, 0]
            self.assertEqual(heat_sparse_csc.nnz, self.ref_torch_sparse_csc._nnz())
            self.assertEqual(heat_sparse_csc.lnnz, nnz_dist[self.rank])

    def test_shape(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)

        self.assertIsInstance(heat_sparse_csc.shape, Tuple)
        self.assertIsInstance(heat_sparse_csc.gshape, Tuple)
        self.assertIsInstance(heat_sparse_csc.lshape, Tuple)

        self.assertEqual(heat_sparse_csc.shape, self.ref_torch_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.shape, heat_sparse_csc.gshape)
        self.assertEqual(heat_sparse_csc.shape, heat_sparse_csc.lshape)

        # Distributed case
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc, split=1)

        if self.world_size == 2:
            lshape_dist = [(5, 3), (5, 2)]

            self.assertEqual(heat_sparse_csc.shape, self.ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])

        if self.world_size == 3:
            lshape_dist = [(5, 2), (5, 2), (5, 1)]

            self.assertEqual(heat_sparse_csc.shape, self.ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])

        # Number of processes > Number of rows
        if self.world_size == 6:
            lshape_dist = [(5, 1), (5, 1), (5, 1), (5, 1), (5, 1), (5, 0)]

            self.assertEqual(heat_sparse_csc.shape, self.ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])

    def test_dtype(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)
        self.assertEqual(heat_sparse_csc.dtype, ht.float32)

    def test_data(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)

        self.assertTrue((heat_sparse_csc.data == self.ref_data).all())
        self.assertTrue((heat_sparse_csc.data == heat_sparse_csc.gdata).all())
        self.assertTrue((heat_sparse_csc.data == heat_sparse_csc.ldata).all())

        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc, split=1)
        if self.world_size == 2:
            data_dist = [[4, 3, 1], [5, 2, 6]]

            self.assertTrue((heat_sparse_csc.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csc.data == heat_sparse_csc.gdata).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            data_dist = [[4, 3], [1, 5], [2, 6]]

            self.assertTrue((heat_sparse_csc.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csc.data == heat_sparse_csc.gdata).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            data_dist = [[4], [3], [1], [5], [2, 6], []]

            self.assertTrue((heat_sparse_csc.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csc.data == heat_sparse_csc.gdata).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

    def test_indices(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)

        self.assertTrue((heat_sparse_csc.indices == self.ref_indices).all())
        self.assertTrue((heat_sparse_csc.indices == heat_sparse_csc.gindices).all())
        self.assertTrue((heat_sparse_csc.indices == heat_sparse_csc.lindices).all())

        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc, split=1)
        if self.world_size == 2:
            indices_dist = [[3, 2, 0], [3, 0, 4]]

            self.assertTrue((heat_sparse_csc.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csc.indices == heat_sparse_csc.gindices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indices_dist = [[3, 2], [0, 3], [0, 4]]

            self.assertTrue((heat_sparse_csc.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csc.indices == heat_sparse_csc.gindices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indices_dist = [[3], [2], [0], [3], [0, 4], []]

            self.assertTrue((heat_sparse_csc.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csc.indices == heat_sparse_csc.gindices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

    def test_indptr(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)

        self.assertTrue((heat_sparse_csc.indptr == self.ref_indptr).all())
        self.assertTrue((heat_sparse_csc.indptr == heat_sparse_csc.gindptr).all())
        self.assertTrue((heat_sparse_csc.indptr == heat_sparse_csc.lindptr).all())
        """
        A = [[0, 0, 1, 0, 2]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 0, 0]
            [4, 0, 0, 5, 0]
            [0, 0, 0, 0, 6]]
        """
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc, split=1)
        if self.world_size == 2:
            indptr_dist = [[0, 1, 2, 3], [0, 1, 3]]

            self.assertTrue((heat_sparse_csc.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csc.indptr == heat_sparse_csc.gindptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 1, 2], [0, 1, 2], [0, 2]]

            self.assertTrue((heat_sparse_csc.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csc.indptr == heat_sparse_csc.gindptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_dist = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 2], [0]]

            self.assertTrue((heat_sparse_csc.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csc.indptr == heat_sparse_csc.gindptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

    def test_astype(self):
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_torch_sparse_csc)

        # check starting invariant
        self.assertEqual(heat_sparse_csc.dtype, ht.float32)

        # check the copy case for uint8
        as_uint8 = heat_sparse_csc.astype(ht.uint8)
        self.assertIsInstance(as_uint8, ht.sparse.DCSC_matrix)
        self.assertEqual(as_uint8.dtype, ht.uint8)
        self.assertEqual(as_uint8.larray.dtype, torch.uint8)
        self.assertIsNot(as_uint8, heat_sparse_csc)

        # check the copy case for uint8
        as_float64 = heat_sparse_csc.astype(ht.float64, copy=False)
        self.assertIsInstance(as_float64, ht.sparse.DCSC_matrix)
        self.assertEqual(as_float64.dtype, ht.float64)
        self.assertEqual(as_float64.larray.dtype, torch.float64)
        self.assertIs(as_float64, heat_sparse_csc)
