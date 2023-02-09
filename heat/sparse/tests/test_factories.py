import unittest
import heat as ht
import torch
import scipy

from heat.core.tests.test_suites.basic_test import TestCase


@unittest.skipIf(
    int(torch.__version__.split(".")[0]) <= 1 and int(torch.__version__.split(".")[1]) < 10,
    f"ht.sparse requires torch >= 1.10. Found version {torch.__version__}.",
)
class TestFactories(TestCase):
    @classmethod
    def setUpClass(self):
        super(TestFactories, self).setUpClass()

        """
        A = [[0, 0, 1, 0, 2]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 0, 0]
            [4, 0, 0, 5, 0]
            [0, 0, 0, 0, 6]]
        """
        self.matrix_list = [
            [0, 0, 1, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [4, 0, 0, 5, 0],
            [0, 0, 0, 0, 6],
        ]
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

        self.ref_scipy_sparse_csr = scipy.sparse.csr_matrix(
            (
                torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float, device="cpu"),
                torch.tensor([2, 4, 1, 0, 3, 4], dtype=torch.int, device="cpu"),
                torch.tensor([0, 2, 2, 3, 5, 6], dtype=torch.int, device="cpu"),
            )
        )

        self.world_size = ht.communication.MPI_WORLD.size
        self.rank = ht.communication.MPI_WORLD.rank

    def test_sparse_csr_matrix(self):
        """
        Input sparse: torch.Tensor
        """
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.DCSR_matrix)
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
            indptr_dist = [[0, 2, 2, 3], [0, 2, 3]]
            indices_dist = [[2, 4, 1], [0, 3, 4]]
            data_dist = [[1, 2, 3], [4, 5, 6]]

            lshape_dist = [(3, 5), (2, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
            indices_dist = [[2, 4], [1, 0, 3], [4]]
            data_dist = [[1, 2], [3, 4, 5], [6]]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_dist = [[0, 2], [0, 0], [0, 1], [0, 2], [0, 1], [0]]
            indices_dist = [[2, 4], [], [1], [0, 3], [4], []]
            data_dist = [[1, 2], [], [3], [4, 5], [6], []]

            lshape_dist = [(1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (0, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Distributed case (is_split)
        if self.world_size == 2:
            indptr_dist = [[0, 2, 2, 3], [0, 2, 3]]
            indices_dist = [[2, 4, 1], [0, 3, 4]]
            data_dist = [[1, 2, 3], [4, 5, 6]]

            lshape_dist = [(3, 5), (2, 5)]

            dist_torch_sparse_csr = torch.sparse_csr_tensor(
                torch.tensor(indptr_dist[self.rank], device=self.device.torch_device),
                torch.tensor(indices_dist[self.rank], device=self.device.torch_device),
                torch.tensor(data_dist[self.rank], device=self.device.torch_device),
                size=lshape_dist[self.rank],
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
            indices_dist = [[2, 4], [1, 0, 3], [4]]
            data_dist = [[1, 2], [3, 4, 5], [6]]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            dist_torch_sparse_csr = torch.sparse_csr_tensor(
                torch.tensor(indptr_dist[self.rank], device=self.device.torch_device),
                torch.tensor(indices_dist[self.rank], device=self.device.torch_device),
                torch.tensor(data_dist[self.rank], device=self.device.torch_device),
                size=lshape_dist[self.rank],
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        """
        Input sparse: scipy.sparse.csr_matrix
        """
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.DCSR_matrix)
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
            indptr_dist = [[0, 2, 2, 3], [0, 2, 3]]
            indices_dist = [[2, 4, 1], [0, 3, 4]]
            data_dist = [[1, 2, 3], [4, 5, 6]]

            lshape_dist = [(3, 5), (2, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_scipy_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
            indices_dist = [[2, 4], [1, 0, 3], [4]]
            data_dist = [[1, 2], [3, 4, 5], [6]]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_scipy_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_dist = [[0, 2], [0, 0], [0, 1], [0, 2], [0, 1], [0]]
            indices_dist = [[2, 4], [], [1], [0, 3], [4], []]
            data_dist = [[1, 2], [], [3], [4, 5], [6], []]

            lshape_dist = [(1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (0, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_scipy_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Distributed case (is_split)
        if self.world_size == 2:
            indptr_dist = [[0, 2, 2, 3], [0, 2, 3]]
            indices_dist = [[2, 4, 1], [0, 3, 4]]
            data_dist = [[1, 2, 3], [4, 5, 6]]

            lshape_dist = [(3, 5), (2, 5)]

            dist_scipy_sparse_csr = scipy.sparse.csr_matrix(
                (
                    torch.tensor(data_dist[self.rank], device="cpu"),
                    torch.tensor(indices_dist[self.rank], device="cpu"),
                    torch.tensor(indptr_dist[self.rank], device="cpu"),
                ),
                shape=lshape_dist[self.rank],
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_scipy_sparse_csr, is_split=0)
            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
            indices_dist = [[2, 4], [1, 0, 3], [4]]
            data_dist = [[1, 2], [3, 4, 5], [6]]

            lshape_dist = [(2, 5), (2, 5), (1, 5)]

            dist_scipy_sparse_csr = scipy.sparse.csr_matrix(
                (
                    torch.tensor(data_dist[self.rank], device="cpu"),
                    torch.tensor(indices_dist[self.rank], device="cpu"),
                    torch.tensor(indptr_dist[self.rank], device="cpu"),
                ),
                shape=lshape_dist[self.rank],
            )

            self.assertEqual(heat_sparse_csr.shape, self.ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == self.ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == self.ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == self.ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        """
        Input: torch.Tensor
        """
        torch_tensor = torch.tensor(
            self.matrix_list, dtype=torch.float, device=self.device.torch_device
        )
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(torch_tensor)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.DCSR_matrix)
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

        """
        Input: List[int]
        """
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.matrix_list)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.DCSR_matrix)
        self.assertEqual(heat_sparse_csr.dtype, ht.int64)
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

        with self.assertRaises(TypeError):
            # Passing an object which cant be converted into a torch.Tensor
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self)

        # Errors (torch.Tensor)
        with self.assertRaises(ValueError):
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr, split=1)

        with self.assertRaises(ValueError):
            dist_torch_sparse_csr = torch.sparse_csr_tensor(
                torch.tensor([0, 0, 0], device=self.device.torch_device),  # indptr
                torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # indices
                torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # data
                size=(2, 2),
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=1)

        # Errors (scipy.sparse.csr_matrix)
        with self.assertRaises(ValueError):
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.ref_scipy_sparse_csr, split=1)

        with self.assertRaises(ValueError):
            dist_scipy_sparse_csr = scipy.sparse.csr_matrix(
                (
                    torch.tensor([], dtype=torch.int64, device="cpu"),  # data
                    torch.tensor([], dtype=torch.int64, device="cpu"),  # indices
                    torch.tensor([0, 0, 0], device="cpu"),  # indptr
                ),
                shape=(2, 2),
            )

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=1)

        # Invalid distribution for is_split
        if self.world_size > 1:
            with self.assertRaises(ValueError):
                dist_torch_sparse_csr = torch.sparse_csr_tensor(
                    torch.tensor(
                        [0] * ((self.rank + 1) + 1), device=self.device.torch_device
                    ),  # indptr
                    torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # indices
                    torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # data
                    size=(self.rank + 1, self.rank + 1),
                )

                heat_sparse_csr = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csr, is_split=0)

        self.assertRaises(ValueError, ht.sparse.sparse_csr_matrix, torch.tensor([0, 1]))
        self.assertRaises(ValueError, ht.sparse.sparse_csr_matrix, torch.tensor([[[1, 0, 3]]]))
