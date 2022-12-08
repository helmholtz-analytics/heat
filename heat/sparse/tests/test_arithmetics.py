import unittest
import heat as ht
import torch
import numpy as np

import random

from heat.core.tests.test_suites.basic_test import TestCase


@unittest.skipIf(
    int(torch.__version__.split(".")[1]) < 10,
    f"ht.sparse requires torch >= 1.10. Found version {torch.__version__}.",
)
class TestArithmetics(TestCase):
    @classmethod
    def setUpClass(self):
        super(TestArithmetics, self).setUpClass()

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
        self.ref_torch_sparse_csr_A = torch.sparse_csr_tensor(
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

    def test_add(self):

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B)

        """
        C = [[2, 0, 1, 0, 5]
            [0, 0, 4, 0, 0]
            [0, 4, 0, 1, 0]
            [4, 0, 0, 5, 0]
            [0, 3, 0, 4, 6]]
        """
        indptr_C = [0, 3, 4, 6, 8, 11]
        indices_C = [0, 2, 4, 2, 1, 3, 0, 3, 1, 3, 4]
        data_C = [2, 1, 5, 4, 4, 1, 4, 5, 3, 4, 6]

        heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue(
            (
                heat_sparse_csr_C.indptr == torch.tensor(indptr_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (
                heat_sparse_csr_C.indices
                == torch.tensor(indices_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)).all()
        )
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # Distributed case
        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [[0, 3, 4, 6], [0, 2, 5]]
            indices_C_dist = [[0, 2, 4, 2, 1, 3], [0, 3, 1, 3, 4]]
            data_C_dist = [[2, 1, 5, 4, 4, 1], [4, 5, 3, 4, 6]]

            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [[0, 3, 4], [0, 2, 4], [0, 3]]
            indices_C_dist = [[0, 2, 4, 2], [1, 3, 0, 3], [1, 3, 4]]
            data_C_dist = [[2, 1, 5, 4], [4, 1, 4, 5], [3, 4, 6]]

            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_A = torch.tensor(
                [0, 2, 2, 2, 2, 2], dtype=torch.int, device=self.device.torch_device
            )
            indices_A = torch.tensor([2, 4], dtype=torch.int, device=self.device.torch_device)
            data_A = torch.tensor([1, 2], dtype=torch.float, device=self.device.torch_device)
            torch_sparse_csr_A = torch.sparse_csr_tensor(
                indptr_A, indices_A, data_A, device=self.device.torch_device
            )
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=0)

            indptr_B = torch.tensor(
                [0, 2, 3, 5, 5, 5], dtype=torch.int, device=self.device.torch_device
            )
            indices_B = torch.tensor(
                [0, 4, 2, 1, 3], dtype=torch.int, device=self.device.torch_device
            )
            data_B = torch.tensor(
                [2, 3, 4, 1, 1], dtype=torch.float, device=self.device.torch_device
            )
            torch_sparse_csr_B = torch.sparse_csr_tensor(
                indptr_B, indices_B, data_B, device=self.device.torch_device
            )
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=0)

            indptr_C_dist = [[0, 3], [0, 1], [0, 2], [0, 0], [0, 0], [0]]
            indices_C_dist = [[0, 2, 4], [2], [1, 3], [], [], []]
            data_C_dist = [[2, 1, 5], [4], [1, 1], [], [], []]

            indptr_C = [0, 3, 4, 6, 6, 6]
            indices_C = [0, 2, 4, 2, 1, 3]
            data_C = [2, 1, 5, 4, 1, 1]

            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # scalar
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A + self.scalar
        heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue(
            (
                heat_sparse_csr_C.indptr == torch.tensor(indptr_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (
                heat_sparse_csr_C.indices
                == torch.tensor(indices_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)).all()
        )
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [[0, 2, 2, 3], [0, 2, 3]]
            indices_C_dist = [[2, 4, 1], [0, 3, 4]]
            data_C_dist = [[1, 2, 3], [4, 5, 6]]
            data_C_dist = [[x + self.scalar for x in data] for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
            indices_C_dist = [[2, 4], [1, 0, 3], [4]]
            data_C_dist = [[1, 2], [3, 4, 5], [6]]
            data_C_dist = [[x + self.scalar for x in data] for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A + self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        """
        [[1, 0]
        [0, 1]]
        """
        torch_sparse_csr_2x2 = torch.sparse_csr_tensor(
            [0, 1, 2], [0, 1], [1, 1], device=self.device.torch_device
        )
        heat_sparse_csr_2x2 = ht.sparse.sparse_csr_matrix(torch_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = heat_sparse_csr_A + heat_sparse_csr_2x2

        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.add(2, 3)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.add(heat_sparse_csr_2x2, torch_sparse_csr_2x2)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.add(torch_sparse_csr_2x2, heat_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = ht.sparse.add(heat_sparse_csr_2x2, heat_sparse_csr_A)

    def test_mul(self):

        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)
        heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B)

        """
        C = [[0, 0, 0, 0, 6]
            [0, 0, 0, 0, 0]
            [0, 3, 0, 0, 0]
            [0, 0, 0, 0, 0]
            [0, 0, 0, 0, 0]]
        """
        indptr_C = [0, 1, 1, 2, 2, 2]
        indices_C = [4, 1]
        data_C = [6, 3]

        heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue(
            (
                heat_sparse_csr_C.indptr == torch.tensor(indptr_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (
                heat_sparse_csr_C.indices
                == torch.tensor(indices_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)).all()
        )
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # Distributed case
        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [[0, 1, 1, 2], [0, 0, 0]]
            indices_C_dist = [[4, 1], []]
            data_C_dist = [[6, 3], []]

            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)

            indptr_C_dist = [[0, 1, 1], [0, 1, 1], [0, 0]]
            indices_C_dist = [[4], [1], []]
            data_C_dist = [[6], [3], []]

            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # Number of processes > Number of rows
        if self.world_size == 6:
            indptr_A = torch.tensor(
                [0, 2, 2, 2, 2, 2], dtype=torch.int, device=self.device.torch_device
            )
            indices_A = torch.tensor([2, 4], dtype=torch.int, device=self.device.torch_device)
            data_A = torch.tensor([1, 2], dtype=torch.float, device=self.device.torch_device)
            torch_sparse_csr_A = torch.sparse_csr_tensor(
                indptr_A, indices_A, data_A, device=self.device.torch_device
            )
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=0)

            indptr_B = torch.tensor(
                [0, 2, 3, 5, 5, 5], dtype=torch.int, device=self.device.torch_device
            )
            indices_B = torch.tensor(
                [0, 4, 2, 1, 3], dtype=torch.int, device=self.device.torch_device
            )
            data_B = torch.tensor(
                [2, 3, 4, 1, 1], dtype=torch.float, device=self.device.torch_device
            )
            torch_sparse_csr_B = torch.sparse_csr_tensor(
                indptr_B, indices_B, data_B, device=self.device.torch_device
            )
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=0)

            indptr_C_dist = [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0]]
            indices_C_dist = [[4], [], [], [], [], []]
            data_C_dist = [[6], [], [], [], [], []]

            indptr_C = [0, 1, 1, 1, 1, 1]
            indices_C = [4]
            data_C = [6]

            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            # Operands with different splits
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=0)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=None)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(torch_sparse_csr_A, split=None)
            heat_sparse_csr_B = ht.sparse.sparse_csr_matrix(torch_sparse_csr_B, split=0)
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_B

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_B.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_B.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        # scalar
        heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A)

        indptr_C = self.ref_indptr_A
        indices_C = self.ref_indices_A
        data_C = self.ref_data_A * self.scalar
        heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

        self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
        self.assertTrue(
            (
                heat_sparse_csr_C.indptr == torch.tensor(indptr_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (
                heat_sparse_csr_C.indices
                == torch.tensor(indices_C, device=self.device.torch_device)
            ).all()
        )
        self.assertTrue(
            (heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)).all()
        )
        self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
        self.assertEqual(heat_sparse_csr_C.split, None)
        self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
        self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 2:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [[0, 2, 2, 3], [0, 2, 3]]
            indices_C_dist = [[2, 4, 1], [0, 3, 4]]
            data_C_dist = [[1, 2, 3], [4, 5, 6]]
            data_C_dist = [[x * self.scalar for x in data] for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        if self.world_size == 3:
            heat_sparse_csr_A = ht.sparse.sparse_csr_matrix(self.ref_torch_sparse_csr_A, split=0)
            indptr_C_dist = [[0, 2, 2], [0, 1, 3], [0, 1]]
            indices_C_dist = [[2, 4], [1, 0, 3], [4]]
            data_C_dist = [[1, 2], [3, 4, 5], [6]]
            data_C_dist = [[x * self.scalar for x in data] for data in data_C_dist]
            heat_sparse_csr_C = heat_sparse_csr_A * self.scalar

            self.assertIsInstance(heat_sparse_csr_C, ht.sparse.DCSR_matrix)
            self.assertTrue(
                (
                    heat_sparse_csr_C.indptr
                    == torch.tensor(indptr_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindptr
                    == torch.tensor(indptr_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.indices
                    == torch.tensor(indices_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.lindices
                    == torch.tensor(indices_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.data == torch.tensor(data_C, device=self.device.torch_device)
                ).all()
            )
            self.assertTrue(
                (
                    heat_sparse_csr_C.ldata
                    == torch.tensor(data_C_dist[self.rank], device=self.device.torch_device)
                ).all()
            )
            self.assertEqual(heat_sparse_csr_C.nnz, len(data_C))
            self.assertEqual(heat_sparse_csr_C.lnnz, len(data_C_dist[self.rank]))
            self.assertEqual(heat_sparse_csr_C.split, 0)
            self.assertEqual(heat_sparse_csr_C.shape, heat_sparse_csr_A.shape)
            self.assertEqual(heat_sparse_csr_C.lshape, heat_sparse_csr_A.lshape)
            self.assertEqual(heat_sparse_csr_C.dtype, ht.float)

        """
        [[1, 0]
        [0, 1]]
        """
        torch_sparse_csr_2x2 = torch.sparse_csr_tensor(
            [0, 1, 2], [0, 1], [1, 1], device=self.device.torch_device
        )
        heat_sparse_csr_2x2 = ht.sparse.sparse_csr_matrix(torch_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = heat_sparse_csr_A * heat_sparse_csr_2x2

        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.mul(2, 3)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.mul(heat_sparse_csr_2x2, torch_sparse_csr_2x2)
        with self.assertRaises(TypeError):
            heat_sparse_csr_C = ht.sparse.mul(torch_sparse_csr_2x2, heat_sparse_csr_2x2)
        with self.assertRaises(ValueError):
            heat_sparse_csr_C = ht.sparse.mul(heat_sparse_csr_2x2, heat_sparse_csr_A)
