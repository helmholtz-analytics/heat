import unittest
import heat as ht
import torch

from heat.core.tests.test_suites.basic_test import TestCase


@unittest.skipIf(
    int(torch.__version__.split(".")[0]) <= 1 and int(torch.__version__.split(".")[1]) < 10,
    f"ht.sparse requires torch >= 1.10. Found version {torch.__version__}.",
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

    def test_todense(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        ref_dense_array = ht.array(
            [
                [0, 0, 1, 0, 2],
                [0, 0, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [4, 0, 0, 5, 0],
                [0, 0, 0, 0, 6],
            ]
        )

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
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)

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
