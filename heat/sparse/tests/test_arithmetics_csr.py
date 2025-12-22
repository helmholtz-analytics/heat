import unittest
import warnings
import numpy as np
import torch
import heat as ht

import os
import platform
import random

from heat.core.tests.test_suites.basic_test import TestCase


default_device = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
is_mps = default_device == "gpu" and platform.system() == "Darwin"


@unittest.skipIf(
    is_mps,
    "sparse_csr_tensor not supported on MPS (PyTorch 2.3)",
)
class TestArithmeticsCSR(TestCase):
    @classmethod
    def setUpClass(self):
        super(TestArithmeticsCSR, self).setUpClass()

        """
        A = [[0, 0, 1, 0, 2]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 0, 0]
            [4, 0, 0, 5, 0]
            [0, 0, 0, 0, 6]]
        """
        self.ref_indptr_A = torch.tensor(
            [0, 2, 2, 3, 5, 6], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_indices_A = torch.tensor(
            [2, 4, 1, 0, 3, 4], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_data_A = torch.tensor(
            [1, 2, 3, 4, 5, 6], dtype=torch.float, device=self.device.torch_device
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.ref_torch_sparse_csr_A: torch.Tensor = torch.sparse_csr_tensor(
                self.ref_indptr_A,
                self.ref_indices_A,
                self.ref_data_A,
                device=self.device.torch_device,
            )

        """
        B = [[2, 0, 0, 0, 3]
            [0, 0, 4, 0, 0]
            [0, 1, 0, 1, 0]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 4, 0]]
        """
        self.ref_indptr_B = torch.tensor(
            [0, 2, 3, 5, 5, 7], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_indices_B = torch.tensor(
            [0, 4, 2, 1, 3, 1, 3], dtype=torch.int, device=self.device.torch_device
        )
        self.ref_data_B = torch.tensor(
            [2, 3, 4, 1, 1, 3, 4], dtype=torch.float, device=self.device.torch_device
        )
        self.ref_torch_sparse_csr_B = torch.sparse_csr_tensor(
            self.ref_indptr_B,
            self.ref_indices_B,
            self.ref_data_B,
            device=self.device.torch_device,
        )

        self.world_size = ht.communication.MPI_WORLD.size
        self.rank = ht.communication.MPI_WORLD.rank

        self.scalar = np.array(random.randint(1, 100))
        if self.world_size > 0:
            ht.communication.MPI_WORLD.Bcast(self.scalar, root=0)
        self.scalar = self.scalar.item()

    # ==================== Helper Methods ====================

    def _assert_csr_equal(self, result, indptr, indices, data, split=None, lindptr=None, lindices=None, ldata=None):
        """Helper to assert CSR matrix equality."""
        # Filter warnings for copy construct of a tensor
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            self.assertIsInstance(result, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (result.indptr == torch.tensor(indptr, device=self.device.torch_device)).all()
            )
            self.assertTrue(
                (result.indices == torch.tensor(indices, device=self.device.torch_device)).all()
            )
            self.assertTrue(
                (result.data == torch.tensor(data, device=self.device.torch_device)).all()
            )
            self.assertEqual(result.nnz, len(data))
            self.assertEqual(result.split, split)

            if lindptr is not None:
                self.assertTrue(
                    (result.lindptr == torch.tensor(lindptr, device=self.device.torch_device)).all()
                )
            if lindices is not None:
                self.assertTrue(
                    (result.lindices == torch.tensor(lindices, device=self.device.torch_device)).all()
                )
            if ldata is not None:
                self.assertTrue(
                    (result.ldata == torch.tensor(ldata, device=self.device.torch_device)).all()
                )
                self.assertEqual(result.lnnz, len(ldata))

    def _to_device_tensor(self, data):
        """Convert data to tensor on device."""
        if isinstance(data, list):
            return torch.tensor(data, device=self.device.torch_device)
        return data.detach().clone().to(self.device.torch_device)

    # ==================== ADD Tests ====================

    def test_add_non_distributed(self):
        """Test ADD operation without distribution."""
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B)

        indptr_C = [0, 3, 4, 6, 8, 11]
        indices_C = [0, 2, 4, 2, 1, 3, 0, 3, 1, 3, 4]
        data_C = [2, 1, 5, 4, 4, 1, 4, 5, 3, 4, 6]

        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(heat_sparse_csr_C, indptr_C, indices_C, data_C, split=None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

    @unittest.skipIf(default_device != "cpu", "only testable on CPU")
    def test_add_distributed_world_size_2(self):
        """Test ADD operation with world_size=2."""
        if ht.communication.MPI_WORLD.size != 2:
            self.skipTest("Requires world_size=2")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

        indptr_C = [0, 3, 4, 6, 8, 11]
        indices_C = [0, 2, 4, 2, 1, 3, 0, 3, 1, 3, 4]
        data_C = [2, 1, 5, 4, 4, 1, 4, 5, 3, 4, 6]
        indptr_C_dist = [[0, 3, 4, 6], [0, 2, 5]]
        indices_C_dist = [[0, 2, 4, 2, 1, 3], [0, 3, 1, 3, 4]]
        data_C_dist = [[2, 1, 5, 4, 4, 1], [4, 5, 3, 4, 6]]

        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

    @unittest.skipIf(default_device != "cpu", "only testable on CPU")
    def test_add_distributed_world_size_2_different_splits(self):
        """Test ADD operation with world_size=2 and different splits."""
        if ht.communication.MPI_WORLD.size != 2:
            self.skipTest("Requires world_size=2")

        indptr_C = [0, 3, 4, 6, 8, 11]
        indices_C = [0, 2, 4, 2, 1, 3, 0, 3, 1, 3, 4]
        data_C = [2, 1, 5, 4, 4, 1, 4, 5, 3, 4, 6]
        indptr_C_dist = [[0, 3, 4, 6], [0, 2, 5]]
        indices_C_dist = [[0, 2, 4, 2, 1, 3], [0, 3, 1, 3, 4]]
        data_C_dist = [[2, 1, 5, 4, 4, 1], [4, 5, 3, 4, 6]]

        # A split, B non-split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)

        # A non-split, B split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
        self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)

    @unittest.skipIf(default_device != "cpu", "only testable on CPU")
    def test_add_distributed_world_size_3(self):
        """Test ADD operation with world_size=3."""
        if ht.communication.MPI_WORLD.size != 3:
            self.skipTest("Requires world_size=3")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

        indptr_C = [0, 3, 4, 6, 8, 11]
        indices_C = [0, 2, 4, 2, 1, 3, 0, 3, 1, 3, 4]
        data_C = [2, 1, 5, 4, 4, 1, 4, 5, 3, 4, 6]
        indptr_C_dist = [[0, 3, 4], [0, 2, 4], [0, 3]]
        indices_C_dist = [[0, 2, 4, 2], [1, 3, 0, 3], [1, 3, 4]]
        data_C_dist = [[2, 1, 5, 4], [4, 1, 4, 5], [3, 4, 6]]

        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)

    @unittest.skipIf(default_device != "cpu", "only testable on CPU")
    def test_add_distributed_world_size_3_different_splits(self):
        """Test ADD operation with world_size=3 and different splits."""
        if ht.communication.MPI_WORLD.size != 3:
            self.skipTest("Requires world_size=3")

        indptr_C = [0, 3, 4, 6, 8, 11]
        indices_C = [0, 2, 4, 2, 1, 3, 0, 3, 1, 3, 4]
        data_C = [2, 1, 5, 4, 4, 1, 4, 5, 3, 4, 6]
        indptr_C_dist = [[0, 3, 4], [0, 2, 4], [0, 3]]
        indices_C_dist = [[0, 2, 4, 2], [1, 3, 0, 3], [1, 3, 4]]
        data_C_dist = [[2, 1, 5, 4], [4, 1, 4, 5], [3, 4, 6]]

        # A split, B non-split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

        # A non-split, B split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)

    @unittest.skipIf(default_device != "cpu", "only testable on CPU")
    def test_add_distributed_world_size_6(self):
        """Test ADD operation with world_size=6 (more processes than rows)."""
        if ht.communication.MPI_WORLD.size != 6:
            self.skipTest("Requires world_size=6")

        indptr_A = torch.tensor([0, 2, 2, 2, 2, 2], dtype=torch.int, device=self.device.torch_device)
        indices_A = torch.tensor([2, 4], dtype=torch.int, device=self.device.torch_device)
        data_A = torch.tensor([1, 2], dtype=torch.float, device=self.device.torch_device)
        torch_sparse_csr_A = torch.sparse_csr_tensor(indptr_A, indices_A, data_A, device=self.device.torch_device)
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=0)

        indptr_B = torch.tensor([0, 2, 3, 5, 5, 5], dtype=torch.int, device=self.device.torch_device)
        indices_B = torch.tensor([0, 4, 2, 1, 3], dtype=torch.int, device=self.device.torch_device)
        data_B = torch.tensor([2, 3, 4, 1, 1], dtype=torch.float, device=self.device.torch_device)
        torch_sparse_csr_B = torch.sparse_csr_tensor(indptr_B, indices_B, data_B, device=self.device.torch_device)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=0)

        indptr_C = [0, 3, 4, 6, 6, 6]
        indices_C = [0, 2, 4, 2, 1, 3]
        data_C = [2, 1, 5, 4, 1, 1]
        indptr_C_dist = [[0, 3], [0, 1], [0, 2], [0, 0], [0, 0], [0]]
        indices_C_dist = [[0, 2, 4], [2], [1, 3], [], [], []]
        data_C_dist = [[2, 1, 5], [4], [1, 1], [], [], []]

        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    def test_add_scalar(self):
        """Test ADD operation with scalar."""
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A + self.scalar
        heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

        self._assert_csr_equal(heat_sparse_csr_C, indptr_C, indices_C, data_C, split=None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

    @unittest.skipIf(default_device != "cpu", "only testable on CPU")
    def test_add_scalar_distributed_world_size_2(self):
        """Test ADD with scalar, world_size=2."""
        if ht.communication.MPI_WORLD.size != 2:
            self.skipTest("Requires world_size=2")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A + self.scalar
        indptr_C_dist = [[0, 2, 2, 3], [0, 2, 3]]
        indices_C_dist = [[2, 4, 1], [0, 3, 4]]
        data_C_dist = [[1, 2, 3], [4, 5, 6]]
        data_C_dist = [[x + self.scalar for x in data] for data in data_C_dist]
        heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    @unittest.skipIf(default_device != "cpu", "only testable on CPU")
    def test_add_scalar_distributed_world_size_3(self):
        """Test ADD with scalar, world_size=3."""
        if ht.communication.MPI_WORLD.size != 3:
            self.skipTest("Requires world_size=3")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A + self.scalar
        indptr_C_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
        indices_C_dist = [[2, 4], [1, 0, 3], [4]]
        data_C_dist = [[1, 2], [3, 4, 5], [6]]
        data_C_dist = [[x + self.scalar for x in data] for data in data_C_dist]
        heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    def test_add_errors(self):
        """Test ADD operation error handling."""
        torch_sparse_csr_2x2 = torch.sparse_csr_tensor(
            [0, 1, 2], [0, 1], [1, 1], device=self.device.torch_device
        )
        heat_sparse_csr_2x2 = ht.sparse.sparse_csr_matrix(torch_sparse_csr_2x2)
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        with self.assertRaises(ValueError):
            heat_sparse_csr_A + heat_sparse_csr_2x2

        with self.assertRaises(TypeError):
            ht.sparse.add(2, 3)
        with self.assertRaises(TypeError):
            ht.sparse.add(heat_sparse_csr_2x2, torch_sparse_csr_2x2)
        with self.assertRaises(TypeError):
            ht.sparse.add(torch_sparse_csr_2x2, heat_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            ht.sparse.add(heat_sparse_csr_2x2, heat_sparse_csr_A)

    # ==================== MUL Tests ====================

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_non_distributed(self):
        """Test MUL operation without distribution."""
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B)

        indptr_C = [0, 1, 1, 2, 2, 2]
        indices_C = [4, 1]
        data_C = [6, 3]

        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(heat_sparse_csr_C, indptr_C, indices_C, data_C, split=None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_distributed_world_size_2(self):
        """Test MUL operation with world_size=2."""
        if ht.communication.MPI_WORLD.size != 2:
            self.skipTest("Requires world_size=2")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

        indptr_C = [0, 1, 1, 2, 2, 2]
        indices_C = [4, 1]
        data_C = [6, 3]
        indptr_C_dist = [[0, 1, 1, 2], [0, 0, 0]]
        indices_C_dist = [[4, 1], []]
        data_C_dist = [[6, 3], []]

        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_distributed_world_size_2_different_splits(self):
        """Test MUL operation with world_size=2 and different splits."""
        if ht.communication.MPI_WORLD.size != 2:
            self.skipTest("Requires world_size=2")

        indptr_C = [0, 1, 1, 2, 2, 2]
        indices_C = [4, 1]
        data_C = [6, 3]
        indptr_C_dist = [[0, 1, 1, 2], [0, 0, 0]]
        indices_C_dist = [[4, 1], []]
        data_C_dist = [[6, 3], []]

        # A split, B non-split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )

        # A non-split, B split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_distributed_world_size_3(self):
        """Test MUL operation with world_size=3."""
        if ht.communication.MPI_WORLD.size != 3:
            self.skipTest("Requires world_size=3")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

        indptr_C = [0, 1, 1, 2, 2, 2]
        indices_C = [4, 1]
        data_C = [6, 3]
        indptr_C_dist = [[0, 1, 1], [0, 1, 1], [0, 0]]
        indices_C_dist = [[4], [1], []]
        data_C_dist = [[6], [3], []]

        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_distributed_world_size_3_different_splits(self):
        """Test MUL operation with world_size=3 and different splits."""
        if ht.communication.MPI_WORLD.size != 3:
            self.skipTest("Requires world_size=3")

        indptr_C = [0, 1, 1, 2, 2, 2]
        indices_C = [4, 1]
        data_C = [6, 3]
        indptr_C_dist = [[0, 1, 1], [0, 1, 1], [0, 0]]
        indices_C_dist = [[4], [1], []]
        data_C_dist = [[6], [3], []]

        # A split, B non-split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )

        # A non-split, B split
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_distributed_world_size_6(self):
        """Test MUL operation with world_size=6 (more processes than rows)."""
        if ht.communication.MPI_WORLD.size != 6:
            self.skipTest("Requires world_size=6")

        indptr_A = torch.tensor([0, 2, 2, 2, 2, 2], dtype=torch.int, device=self.device.torch_device)
        indices_A = torch.tensor([2, 4], dtype=torch.int, device=self.device.torch_device)
        data_A = torch.tensor([1, 2], dtype=torch.float, device=self.device.torch_device)
        torch_sparse_csr_A = torch.sparse_csr_tensor(indptr_A, indices_A, data_A, device=self.device.torch_device)
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=0)

        indptr_B = torch.tensor([0, 2, 3, 5, 5, 5], dtype=torch.int, device=self.device.torch_device)
        indices_B = torch.tensor([0, 4, 2, 1, 3], dtype=torch.int, device=self.device.torch_device)
        data_B = torch.tensor([2, 3, 4, 1, 1], dtype=torch.float, device=self.device.torch_device)
        torch_sparse_csr_B = torch.sparse_csr_tensor(indptr_B, indices_B, data_B, device=self.device.torch_device)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=0)

        indptr_C = [0, 1, 1, 1, 1, 1]
        indices_C = [4]
        data_C = [6]
        indptr_C_dist = [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0]]
        indices_C_dist = [[4], [], [], [], [], []]
        data_C_dist = [[6], [], [], [], [], []]

        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_scalar(self):
        """Test MUL operation with scalar."""
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A * self.scalar
        heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

        self._assert_csr_equal(heat_sparse_csr_C, indptr_C, indices_C, data_C, split=None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_scalar_distributed_world_size_2(self):
        """Test MUL with scalar, world_size=2."""
        if ht.communication.MPI_WORLD.size != 2:
            self.skipTest("Requires world_size=2")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A * self.scalar
        indptr_C_dist = [[0, 2, 2, 3], [0, 2, 3]]
        indices_C_dist = [[2, 4, 1], [0, 3, 4]]
        data_C_dist = [[1, 2, 3], [4, 5, 6]]
        data_C_dist = [[x * self.scalar for x in data] for data in data_C_dist]
        heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_scalar_distributed_world_size_3(self):
        """Test MUL with scalar, world_size=3."""
        if ht.communication.MPI_WORLD.size != 3:
            self.skipTest("Requires world_size=3")

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A * self.scalar
        indptr_C_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
        indices_C_dist = [[2, 4], [1, 0, 3], [4]]
        data_C_dist = [[1, 2], [3, 4, 5], [6]]
        data_C_dist = [[x * self.scalar for x in data] for data in data_C_dist]
        heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

        self._assert_csr_equal(
            heat_sparse_csr_C, indptr_C, indices_C, data_C, split=0,
            lindptr=indptr_C_dist[self.rank],
            lindices=indices_C_dist[self.rank],
            ldata=data_C_dist[self.rank]
        )
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)

    @unittest.skipUnless(default_device == "cpu", "only testable on CPU")
    def test_mul_errors(self):
        """Test MUL operation error handling."""
        torch_sparse_csr_2x2 = torch.sparse_csr_tensor(
            [0, 1, 2], [0, 1], [1, 1], device=self.device.torch_device
        )
        heat_sparse_csr_2x2 = ht.sparse.sparse_csr_matrix(torch_sparse_csr_2x2)
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        with self.assertRaises(ValueError):
            heat_sparse_csr_A * heat_sparse_csr_2x2

        with self.assertRaises(TypeError):
            ht.sparse.mul(2, 3)
        with self.assertRaises(TypeError):
            ht.sparse.mul(heat_sparse_csr_2x2, torch_sparse_csr_2x2)
        with self.assertRaises(TypeError):
            ht.sparse.mul(torch_sparse_csr_2x2, heat_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            ht.sparse.mul(heat_sparse_csr_2x2, heat_sparse_csr_A)
