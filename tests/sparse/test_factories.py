import unittest
import os
import platform
import heat as ht
import torch
import scipy

from tests.test_suites.basic_test import TestCase

envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
is_mps = envar == "gpu" and platform.system() == "Darwin"


@unittest.skipIf(
    is_mps,
    "sparse_csr_tensor not supported on MPS (PyTorch 2.3)",
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
        self.arr = [
            [0, 0, 1, 0, 2],
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [4, 0, 0, 5, 0],
            [0, 0, 0, 0, 6],
        ]

        self.world_size = ht.communication.MPI_WORLD.size
        self.rank = ht.communication.MPI_WORLD.rank

    def test_sparse_csr_matrix(self):
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
        """
        Input sparse: torch.Tensor
        """
        self.ref_scipy_sparse_csr = scipy.sparse.csr_matrix(
            (
                torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float, device="cpu"),
                torch.tensor([2, 4, 1, 0, 3, 4], dtype=torch.int, device="cpu"),
                torch.tensor([0, 2, 2, 3, 5, 6], dtype=torch.int, device="cpu"),
            )
        )
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(ref_torch_sparse_csr)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.DCSR_matrix)
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)
        self.assertEqual(heat_sparse_csr.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.lshape, ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.split, None)
        self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.data == ref_data).all())
        self.assertTrue((heat_sparse_csr.ldata == ref_data).all())

        heat_sparse_csr = ht.sparse.sparse_csr_matrix(
            ref_torch_sparse_csr, dtype=ht.float32, device=self.device
        )
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)
        self.assertEqual(heat_sparse_csr.device, self.device)

        # Distributed case (split)
        if self.world_size == 2:
            indptr_dist = [[0, 2, 2, 3], [0, 2, 3]]
            indices_dist = [[2, 4, 1], [0, 3, 4]]
            data_dist = [[1, 2, 3], [4, 5, 6]]

            lshape_dist = [(3, 5), (2, 5)]

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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

            heat_sparse_csr = ht.sparse.sparse_csr_matrix(ref_torch_sparse_csr, split=0)
            self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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
            self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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
            self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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
        self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.data == ref_data).all())
        self.assertTrue((heat_sparse_csr.ldata == ref_data).all())

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
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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
            self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
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

            self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
            self.assertEqual(heat_sparse_csr.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csr.split, 0)
            self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csr.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csr.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csr.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        """
        Input: torch.Tensor
        """
        torch_tensor = torch.tensor(self.arr, dtype=torch.float, device=self.device.torch_device)
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(torch_tensor)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.DCSR_matrix)
        self.assertEqual(heat_sparse_csr.dtype, ht.float32)
        self.assertEqual(heat_sparse_csr.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.lshape, ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.split, None)
        self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.data == ref_data).all())
        self.assertTrue((heat_sparse_csr.ldata == ref_data).all())

        """
        Input: List[int]
        """
        heat_sparse_csr = ht.sparse.sparse_csr_matrix(self.arr)

        self.assertIsInstance(heat_sparse_csr, ht.sparse.DCSR_matrix)
        self.assertEqual(heat_sparse_csr.dtype, ht.int64)
        self.assertEqual(heat_sparse_csr.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csr.shape, ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.lshape, ref_torch_sparse_csr.shape)
        self.assertEqual(heat_sparse_csr.split, None)
        self.assertTrue((heat_sparse_csr.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csr.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csr.data == ref_data).all())
        self.assertTrue((heat_sparse_csr.ldata == ref_data).all())

        with self.assertRaises(TypeError):
            # Passing an object which cant be converted into a torch.Tensor
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(self)

        # Errors (torch.Tensor)
        with self.assertRaises(ValueError):
            heat_sparse_csr = ht.sparse.sparse_csr_matrix(ref_torch_sparse_csr, split=1)

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

    def test_sparse_csc_matrix(self):
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
        """
        Input sparse: torch.Tensor
        """
        self.ref_scipy_sparse_csc = scipy.sparse.csc_matrix(
            (
                torch.tensor([4, 3, 1, 5, 2, 6], dtype=torch.float, device="cpu"),
                torch.tensor([3, 2, 0, 3, 0, 4], dtype=torch.int, device="cpu"),
                torch.tensor([0, 1, 2, 3, 4, 6], dtype=torch.int, device="cpu"),
            )
        )
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(ref_torch_sparse_csc)

        self.assertIsInstance(heat_sparse_csc, ht.sparse.DCSC_matrix)
        self.assertEqual(heat_sparse_csc.dtype, ht.float32)
        self.assertEqual(heat_sparse_csc.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.lshape, ref_torch_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.split, None)
        self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.data == ref_data).all())
        self.assertTrue((heat_sparse_csc.ldata == ref_data).all())

        heat_sparse_csc = ht.sparse.sparse_csc_matrix(
            ref_torch_sparse_csc, dtype=ht.float32, device=self.device
        )
        self.assertEqual(heat_sparse_csc.dtype, ht.float32)
        self.assertEqual(heat_sparse_csc.device, self.device)

        # Distributed case (split)
        if self.world_size == 2:
            indptr_dist = [[0, 1, 2, 3], [0, 1, 3]]
            indices_dist = [[3, 2, 0], [3, 0, 4]]
            data_dist = [[4, 3, 1], [5, 2, 6]]

            lshape_dist = [(5, 3), (5, 2)]

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(ref_torch_sparse_csc, split=1)
            self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 1, 2], [0, 1, 2], [0, 2]]
            indices_dist = [[3, 2], [0, 3], [0, 4]]
            data_dist = [[4, 3], [1, 5], [2, 6]]

            lshape_dist = [(5, 2), (5, 2), (5, 1)]

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(ref_torch_sparse_csc, split=1)
            self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_dist = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 2], [0]]
            indices_dist = [[3], [2], [0], [3], [0, 4], []]
            data_dist = [[4], [3], [1], [5], [2, 6], []]

            lshape_dist = [(5, 1), (5, 1), (5, 1), (5, 1), (5, 1), (5, 0)]

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(ref_torch_sparse_csc, split=1)
            self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Distributed case (is_split)
        if self.world_size == 2:
            indptr_dist = [[0, 1, 2, 3], [0, 1, 3]]
            indices_dist = [[3, 2, 0], [3, 0, 4]]
            data_dist = [[4, 3, 1], [5, 2, 6]]

            lshape_dist = [(5, 3), (5, 2)]

            dist_torch_sparse_csc = torch.sparse_csc_tensor(
                torch.tensor(indptr_dist[self.rank], device=self.device.torch_device),
                torch.tensor(indices_dist[self.rank], device=self.device.torch_device),
                torch.tensor(data_dist[self.rank], device=self.device.torch_device),
                size=lshape_dist[self.rank],
            )

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(dist_torch_sparse_csc, is_split=1)
            self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 1, 2], [0, 1, 2], [0, 2]]
            indices_dist = [[3, 2], [0, 3], [0, 4]]
            data_dist = [[4, 3], [1, 5], [2, 6]]

            lshape_dist = [(5, 2), (5, 2), (5, 1)]

            dist_torch_sparse_csc = torch.sparse_csc_tensor(
                torch.tensor(indptr_dist[self.rank], device=self.device.torch_device),
                torch.tensor(indices_dist[self.rank], device=self.device.torch_device),
                torch.tensor(data_dist[self.rank], device=self.device.torch_device),
                size=lshape_dist[self.rank],
            )

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(dist_torch_sparse_csc, is_split=1)
            self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        """
        Input sparse: scipy.sparse.csc_matrix
        """
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_scipy_sparse_csc)

        self.assertIsInstance(heat_sparse_csc, ht.sparse.DCSC_matrix)
        self.assertEqual(heat_sparse_csc.dtype, ht.float32)
        self.assertEqual(heat_sparse_csc.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.shape, self.ref_scipy_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.lshape, self.ref_scipy_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.split, None)
        self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.data == ref_data).all())
        self.assertTrue((heat_sparse_csc.ldata == ref_data).all())

        # Distributed case (split)
        if self.world_size == 2:
            indptr_dist = [[0, 1, 2, 3], [0, 1, 3]]
            indices_dist = [[3, 2, 0], [3, 0, 4]]
            data_dist = [[4, 3, 1], [5, 2, 6]]

            lshape_dist = [(5, 3), (5, 2)]

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_scipy_sparse_csc, split=1)
            self.assertEqual(heat_sparse_csc.shape, self.ref_scipy_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 1, 2], [0, 1, 2], [0, 2]]
            indices_dist = [[3, 2], [0, 3], [0, 4]]
            data_dist = [[4, 3], [1, 5], [2, 6]]

            lshape_dist = [(5, 2), (5, 2), (5, 1)]

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_scipy_sparse_csc, split=1)
            self.assertEqual(heat_sparse_csc.shape, self.ref_scipy_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_dist = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 2], [0]]
            indices_dist = [[3], [2], [0], [3], [0, 4], []]
            data_dist = [[4], [3], [1], [5], [2, 6], []]

            lshape_dist = [(5, 1), (5, 1), (5, 1), (5, 1), (5, 1), (5, 0)]

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_scipy_sparse_csc, split=1)
            self.assertEqual(heat_sparse_csc.shape, self.ref_scipy_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        # Distributed case (is_split)
        if self.world_size == 2:
            indptr_dist = [[0, 1, 2, 3], [0, 1, 3]]
            indices_dist = [[3, 2, 0], [3, 0, 4]]
            data_dist = [[4, 3, 1], [5, 2, 6]]

            lshape_dist = [(5, 3), (5, 2)]

            dist_scipy_sparse_csc = scipy.sparse.csc_matrix(
                (
                    torch.tensor(data_dist[self.rank], device="cpu"),
                    torch.tensor(indices_dist[self.rank], device="cpu"),
                    torch.tensor(indptr_dist[self.rank], device="cpu"),
                ),
                shape=lshape_dist[self.rank],
            )

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(dist_scipy_sparse_csc, is_split=1)
            self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        if self.world_size == 3:
            indptr_dist = [[0, 1, 2], [0, 1, 2], [0, 2]]
            indices_dist = [[3, 2], [0, 3], [0, 4]]
            data_dist = [[4, 3], [1, 5], [2, 6]]

            lshape_dist = [(5, 2), (5, 2), (5, 1)]

            dist_scipy_sparse_csc = scipy.sparse.csc_matrix(
                (
                    torch.tensor(data_dist[self.rank], device="cpu"),
                    torch.tensor(indices_dist[self.rank], device="cpu"),
                    torch.tensor(indptr_dist[self.rank], device="cpu"),
                ),
                shape=lshape_dist[self.rank],
            )

            self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
            self.assertEqual(heat_sparse_csc.lshape, lshape_dist[self.rank])
            self.assertEqual(heat_sparse_csc.split, 1)
            self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindptr
                    == torch.tensor(indptr_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
            self.assertTrue(
                (
                    heat_sparse_csc.lindices
                    == torch.tensor(indices_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue((heat_sparse_csc.data == ref_data).all())
            self.assertTrue(
                (
                    heat_sparse_csc.ldata
                    == torch.tensor(data_dist[self.rank], device=self.device.torch_device)
                ).all()
            )

        """
        Input: torch.Tensor
        """
        torch_tensor = torch.tensor(self.arr, dtype=torch.float, device=self.device.torch_device)
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(torch_tensor)

        self.assertIsInstance(heat_sparse_csc, ht.sparse.DCSC_matrix)
        self.assertEqual(heat_sparse_csc.dtype, ht.float32)
        self.assertEqual(heat_sparse_csc.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.lshape, ref_torch_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.split, None)
        self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.data == ref_data).all())
        self.assertTrue((heat_sparse_csc.ldata == ref_data).all())

        """
        Input: List[int]
        """
        heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.arr)

        self.assertIsInstance(heat_sparse_csc, ht.sparse.DCSC_matrix)
        self.assertEqual(heat_sparse_csc.dtype, ht.int64)
        self.assertEqual(heat_sparse_csc.indptr.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.indices.dtype, torch.int64)
        self.assertEqual(heat_sparse_csc.shape, ref_torch_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.lshape, ref_torch_sparse_csc.shape)
        self.assertEqual(heat_sparse_csc.split, None)
        self.assertTrue((heat_sparse_csc.indptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.lindptr == ref_indptr).all())
        self.assertTrue((heat_sparse_csc.indices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.lindices == ref_indices).all())
        self.assertTrue((heat_sparse_csc.data == ref_data).all())
        self.assertTrue((heat_sparse_csc.ldata == ref_data).all())

        with self.assertRaises(TypeError):
            # Passing an object which cant be converted into a torch.Tensor
            heat_sparse_csc = ht.sparse.sparse_csc_matrix(self)

        # Errors (torch.Tensor)
        with self.assertRaises(ValueError):
            heat_sparse_csc = ht.sparse.sparse_csc_matrix(ref_torch_sparse_csc, split=0)

        with self.assertRaises(ValueError):
            dist_torch_sparse_csc = torch.sparse_csc_tensor(
                torch.tensor([0, 0, 0], device=self.device.torch_device),  # indptr
                torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # indices
                torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # data
                size=(2, 2),
            )

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(dist_torch_sparse_csc, is_split=0)

        # Errors (scipy.sparse.csc_matrix)
        with self.assertRaises(ValueError):
            heat_sparse_csc = ht.sparse.sparse_csc_matrix(self.ref_scipy_sparse_csc, split=0)

        with self.assertRaises(ValueError):
            dist_scipy_sparse_csc = scipy.sparse.csc_matrix(
                (
                    torch.tensor([], dtype=torch.int64, device="cpu"),  # data
                    torch.tensor([], dtype=torch.int64, device="cpu"),  # indices
                    torch.tensor([0, 0, 0], device="cpu"),  # indptr
                ),
                shape=(2, 2),
            )

            heat_sparse_csc = ht.sparse.sparse_csc_matrix(dist_torch_sparse_csc, is_split=0)

        # Invalid distribution for is_split
        if self.world_size > 1:
            with self.assertRaises(ValueError):
                dist_torch_sparse_csc = torch.sparse_csc_tensor(
                    torch.tensor(
                        [0] * ((self.rank + 1) + 1), device=self.device.torch_device
                    ),  # indptr
                    torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # indices
                    torch.tensor([], dtype=torch.int64, device=self.device.torch_device),  # data
                    size=(self.rank + 1, self.rank + 1),
                )

                heat_sparse_csc = ht.sparse.sparse_csr_matrix(dist_torch_sparse_csc, is_split=1)

        self.assertRaises(ValueError, ht.sparse.sparse_csc_matrix, torch.tensor([0, 1]))
        self.assertRaises(ValueError, ht.sparse.sparse_csc_matrix, torch.tensor([[[1, 0, 3]]]))
