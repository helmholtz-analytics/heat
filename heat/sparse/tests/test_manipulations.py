from weakref import ref
import heat as ht
import torch

from heat.core.tests.test_suites.basic_test import TestCase


class TestManipulations(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestManipulations, cls).setUpClass()
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

    def test_todense(self):
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        dense_array = heat_sparse_csr.todense()

        ref_dense_array = ht.array(
            [[0, 0, 1, 0, 2], [0, 0, 0, 0, 0], [0, 3, 0, 0, 0], [4, 0, 0, 5, 0], [0, 0, 0, 0, 6]]
        )

        self.assertTrue(ht.equal(ref_dense_array, dense_array))
        self.assertEqual(dense_array.split, None)
        self.assertEqual(dense_array.dtype, heat_sparse_csr.dtype)
        self.assertEqual(dense_array.shape, heat_sparse_csr.shape)

        # Distributed case
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)

        dense_array = heat_sparse_csr.todense()
        ref_dense_array = ht.array(ref_dense_array, split=0)

        self.assertTrue(ht.equal(ref_dense_array, dense_array))
        self.assertEqual(dense_array.split, 0)
        self.assertEqual(dense_array.dtype, heat_sparse_csr.dtype)
        self.assertEqual(dense_array.shape, heat_sparse_csr.shape)
