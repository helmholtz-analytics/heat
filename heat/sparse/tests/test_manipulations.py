import unittest
import os
import platform
import heat as ht
import torch

from heat.core.tests.test_suites.basic_test import TestCase

envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
is_mps = envar == "gpu" and platform.system() == "Darwin"


@unittest.skipIf(
    is_mps,
    "sparse_csr_tensor not supported on MPS (PyTorch 2.3)",
)
class TestManipulations(TestCase):
    @classmethod
    def setUpClass(self):
        super(TestManipulations, self).setUpClass()
        """
        A = [[0, 0, 1, 0, 2]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 0, 0]
            [4, 0, 0, 5, 0]
            [0, 0, 0, 0, 6]]
        """
        self.arr = [
            [0, 0, 1, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [4, 0, 0, 5, 0],
            [0, 0, 0, 0, 6],
        ]

    def test_to_sparse_csr(self):
        A = ht.array(self.arr, split=0)
        B = A.to_sparse_csr()

        indptr_B = [0, 2, 2, 3, 5, 6]
        indices_B = [2, 4, 1, 0, 3, 4]
        data_B = [1, 2, 3, 4, 5, 6]

        self.assertIsInstance(B, ht.sparse.DCSR_matrix)
        self.assertTrue((B.indptr == torch.tensor(indptr_B, device=self.device.torch_device)).all())
        self.assertTrue(
            (B.indices == torch.tensor(indices_B, device=self.device.torch_device)).all()
        )
        self.assertTrue((B.data == torch.tensor(data_B, device=self.device.torch_device)).all())
        self.assertEqual(B.nnz, len(data_B))
        self.assertEqual(B.split, 0)
        self.assertEqual(B.shape, A.shape)
        self.assertEqual(B.dtype, A.dtype)

    def test_to_sparse_csc(self):
        A = ht.array(self.arr, split=1)
        B = A.to_sparse_csc()

        indptr_B = [0, 1, 2, 3, 4, 6]
        indices_B = [3, 2, 0, 3, 0, 4]
        data_B = [4, 3, 1, 5, 2, 6]

        self.assertIsInstance(B, ht.sparse.DCSC_matrix)
        self.assertTrue((B.indptr == torch.tensor(indptr_B, device=self.device.torch_device)).all())
        self.assertTrue(
            (B.indices == torch.tensor(indices_B, device=self.device.torch_device)).all()
        )
        self.assertTrue((B.data == torch.tensor(data_B, device=self.device.torch_device)).all())
        self.assertEqual(B.nnz, len(data_B))
        self.assertEqual(B.split, 1)
        self.assertEqual(B.shape, A.shape)
        self.assertEqual(B.dtype, A.dtype)

    def test_to_dense_csr(self):
        ref_indptr = torch.tensor(
            [0, 2, 2, 3, 5, 6], dtype=torch.int, device=self.device.torch_device
        )
        ref_indices = torch.tensor(
            [2, 4, 1, 0, 3, 4], dtype=torch.int, device=self.device.torch_device
        )
        ref_data = torch.tensor(
            [1, 2, 3, 4, 5, 6], dtype=torch.float, device=self.device.torch_device
        )
        ref_torch_sparse_csr = torch.sparse_csr_tensor(
            ref_indptr, ref_indices, ref_data, device=self.device.torch_device
        )
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(ref_torch_sparse_csr)

        ref_dense_array = ht.array(self.arr)

        dense_array = heat_sparse_csr.todense()

        self.assertTrue(ht.equal(ref_dense_array, dense_array))
        self.assertEqual(dense_array.split, None)
        self.assertEqual(dense_array.dtype, heat_sparse_csr.dtype)
        self.assertEqual(dense_array.shape, heat_sparse_csr.shape)

        # with output buffer
        out_buffer = ht.empty(shape=[5, 5])
        heat_sparse_csr.todense(out=out_buffer)

        self.assertTrue(ht.equal(ref_dense_array, out_buffer))
        self.assertEqual(out_buffer.split, None)
        self.assertEqual(out_buffer.dtype, heat_sparse_csr.dtype)
        self.assertEqual(out_buffer.shape, heat_sparse_csr.shape)

        # Distributed case
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(ref_torch_sparse_csr, split=0)

        dense_array = heat_sparse_csr.todense()
        ref_dense_array = ht.array(ref_dense_array, split=0)

        self.assertTrue(ht.equal(ref_dense_array, dense_array))
        self.assertEqual(dense_array.split, 0)
        self.assertEqual(dense_array.dtype, heat_sparse_csr.dtype)
        self.assertEqual(dense_array.shape, heat_sparse_csr.shape)

        # with output buffer
        out_buffer = ht.empty(shape=[5, 5], split=0)
        heat_sparse_csr.todense(out=out_buffer)

        self.assertTrue(ht.equal(ref_dense_array, out_buffer))
        self.assertEqual(out_buffer.split, 0)
        self.assertEqual(out_buffer.dtype, heat_sparse_csr.dtype)
        self.assertEqual(out_buffer.shape, heat_sparse_csr.shape)

        with self.assertRaises(ValueError):
            out_buffer = ht.empty(shape=[3, 3], split=0)
            heat_sparse_csr.todense(out=out_buffer)

        with self.assertRaises(ValueError):
            out_buffer = ht.empty(shape=[5, 5], split=None)
            heat_sparse_csr.todense(out=out_buffer)

    def test_to_dense_csc(self):
        ref_indptr = torch.tensor(
            [0, 1, 2, 3, 4, 6], dtype=torch.int, device=self.device.torch_device
        )
        ref_indices = torch.tensor(
            [3, 2, 0, 3, 0, 4], dtype=torch.int, device=self.device.torch_device
        )
        ref_data = torch.tensor(
            [4, 3, 1, 5, 2, 6], dtype=torch.float, device=self.device.torch_device
        )
        ref_torch_sparse_csc = torch.sparse_csc_tensor(
            ref_indptr, ref_indices, ref_data, device=self.device.torch_device
        )
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(ref_torch_sparse_csc)

        ref_dense_array = ht.array(self.arr)

        dense_array = heat_sparse_csc.todense()

        self.assertTrue(ht.equal(ref_dense_array, dense_array))
        self.assertEqual(dense_array.split, None)
        self.assertEqual(dense_array.dtype, heat_sparse_csc.dtype)
        self.assertEqual(dense_array.shape, heat_sparse_csc.shape)

        # with output buffer
        out_buffer = ht.empty(shape=[5, 5])
        heat_sparse_csc.todense(out=out_buffer)

        self.assertTrue(ht.equal(ref_dense_array, out_buffer))
        self.assertEqual(out_buffer.split, None)
        self.assertEqual(out_buffer.dtype, heat_sparse_csc.dtype)
        self.assertEqual(out_buffer.shape, heat_sparse_csc.shape)

        # Distributed case
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(ref_torch_sparse_csc, split=1)

        dense_array = heat_sparse_csc.todense()
        ref_dense_array = ht.array(ref_dense_array, split=1)

        self.assertTrue(ht.equal(ref_dense_array, dense_array))
        self.assertEqual(dense_array.split, 1)
        self.assertEqual(dense_array.dtype, heat_sparse_csc.dtype)
        self.assertEqual(dense_array.shape, heat_sparse_csc.shape)

        # with output buffer
        out_buffer = ht.empty(shape=[5, 5], split=1)
        heat_sparse_csc.todense(out=out_buffer)

        self.assertTrue(ht.equal(ref_dense_array, out_buffer))
        self.assertEqual(out_buffer.split, 1)
        self.assertEqual(out_buffer.dtype, heat_sparse_csc.dtype)
        self.assertEqual(out_buffer.shape, heat_sparse_csc.shape)

        with self.assertRaises(ValueError):
            out_buffer = ht.empty(shape=[3, 3], split=1)
            heat_sparse_csc.todense(out=out_buffer)

        with self.assertRaises(ValueError):
            out_buffer = ht.empty(shape=[5, 5], split=None)
            heat_sparse_csc.todense(out=out_buffer)
