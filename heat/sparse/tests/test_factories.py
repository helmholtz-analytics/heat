import heat as ht
import torch
import scipy

from heat.core.tests.test_suites.basic_test import TestCase


class TestFactories(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestFactories, cls).setUpClass()
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
        cls.ref_scipy_sparse_csr = scipy.sparse.csr_matrix(
            (cls.ref_data, cls.ref_indices, cls.ref_indptr)
        )

        cls.world_size = ht.communication.MPI_WORLD.size
        cls.rank = ht.communication.MPI_WORLD.rank

    def test_sparse_csr_matrix(self):

        """
        Input sparse: torch.Tensor
        """
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.Dcsr_matrix)
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)
        self.assertEqual(heat_sparse_csr.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.lshape, self.ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.split, None)
        self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
        self.assertTrue((heat_sparse_csr.lindptr == self.ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
        self.assertTrue((heat_sparse_csr.lindices == self.ref_indices).all())
        self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
        self.assertTrue((heat_sparse_csr.ldata == self.ref_data).all())

        heat_sparse_csr = ht.sparse.sparse_csr_matrix(
            self.ref_torch_sparse_csr, dtype=ht.float32, device=self.device
        )
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)
        self.assertEqual(heat_sparse_csr.device, self.device)

        # Distributed case (split)
        if self.world_size == 2:
            indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
            indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
            data_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

            lshape_dist = [(3, 5), (2, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        if self.world_size == 3:
            indptr_dist = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 3]), torch.tensor([0, 1])]
            indices_dist = [torch.tensor([2, 4]), torch.tensor([1, 0, 3]), torch.tensor([4])]
            data_dist = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        # Distributed case (is_split)
        if self.world_size == 2:
            indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
            indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
            data_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

            lshape_dist = [(3, 5), (2, 5)]

            dist_torch_sparse_csr = torch.sparse_csr_tensor(
                indptr_dist[self.rank],
                indices_dist[self.rank],
                data_dist[self.rank],
                size=lshape_dist[self.rank],
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        if self.world_size == 3:
            indptr_dist = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 3]), torch.tensor([0, 1])]
            indices_dist = [torch.tensor([2, 4]), torch.tensor([1, 0, 3]), torch.tensor([4])]
            data_dist = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            dist_torch_sparse_csr = torch.sparse_csr_tensor(
                indptr_dist[self.rank],
                indices_dist[self.rank],
                data_dist[self.rank],
                size=lshape_dist[self.rank],
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        """
        Input sparse: scipy.sparse.csr_matrix
        """
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.Dcsr_matrix)
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)
        self.assertEqual(heat_sparse_csr.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.shape, self.ref_scipy_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.lshape, self.ref_scipy_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.split, None)
        self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
        self.assertTrue((heat_sparse_csr.lindptr == self.ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
        self.assertTrue((heat_sparse_csr.lindices == self.ref_indices).all())
        self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
        self.assertTrue((heat_sparse_csr.ldata == self.ref_data).all())

        # Distributed case (split)
        if self.world_size == 2:
            indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
            indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
            data_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

            lshape_dist = [(3, 5), (2, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_scipy_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        if self.world_size == 3:
            indptr_dist = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 3]), torch.tensor([0, 1])]
            indices_dist = [torch.tensor([2, 4]), torch.tensor([1, 0, 3]), torch.tensor([4])]
            data_dist = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_scipy_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        # Distributed case (is_split)
        if self.world_size == 2:
            indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
            indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
            data_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

            lshape_dist = [(3, 5), (2, 5)]

            dist_scipy_sparse_csr = scipy.sparse.csr_matrix(
                (data_dist[self.rank], indices_dist[self.rank], indptr_dist[self.rank]),
                shape=lshape_dist[self.rank],
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_scipy_sparse_csr, is_split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        if self.world_size == 3:
            indptr_dist = [torch.tensor([0, 2, 2]), torch.tensor([0, 1, 3]), torch.tensor([0, 1])]
            indices_dist = [torch.tensor([2, 4]), torch.tensor([1, 0, 3]), torch.tensor([4])]
            data_dist = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            dist_scipy_sparse_csr = scipy.sparse.csr_matrix(
                (data_dist[self.rank], indices_dist[self.rank], indptr_dist[self.rank]),
                shape=lshape_dist[self.rank],
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue((heat_sparse_csr.lindptr == indptr_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue((heat_sparse_csr.lindices == indices_dist[self.rank]).all())
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue((heat_sparse_csr.ldata == data_dist[self.rank]).all())

        # Errors (torch.Tensor)
        with self.assertRaises(NotImplementedError):
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=1)

        if self.world_size == 2:
            with self.assertRaises(NotImplementedError):
                indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
                indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
                data_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

                lshape_dist = [(3, 5), (2, 5)]

                dist_torch_sparse_csr = torch.sparse_csr_tensor(
                    indptr_dist[self.rank],
                    indices_dist[self.rank],
                    data_dist[self.rank],
                    size=lshape_dist[self.rank],
                )

                heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=1)

        # Errors (scipy.sparse.csr_matrix)
        with self.assertRaises(NotImplementedError):
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr, split=1)

        if self.world_size == 2:
            with self.assertRaises(NotImplementedError):
                indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 2, 3])]
                indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([0, 3, 4])]
                data_dist = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

                lshape_dist = [(3, 5), (2, 5)]

                dist_scipy_sparse_csr = scipy.sparse.csr_matrix(
                    (data_dist[self.rank], indices_dist[self.rank], indptr_dist[self.rank]),
                    shape=lshape_dist[self.rank],
                )

                heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=1)

        # Invalid distribution for is_split
        if self.world_size == 2:
            with self.assertRaises(ValueError):
                indptr_dist = [torch.tensor([0, 2, 2, 3]), torch.tensor([0, 1, 2])]
                indices_dist = [torch.tensor([2, 4, 1]), torch.tensor([1, 2])]
                data_dist = [torch.tensor([1, 2, 3]), torch.tensor([5, 6])]

                lshape_dist = [(3, 5), (2, 3)]

                dist_torch_sparse_csr = torch.sparse_csr_tensor(
                    indptr_dist[self.rank],
                    indices_dist[self.rank],
                    data_dist[self.rank],
                    size=lshape_dist[self.rank],
                )

                heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=0)
