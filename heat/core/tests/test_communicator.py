import unittest
import torch

import heat as ht


class TestCommunicator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = torch.tensor([
            [3, 2, 1],
            [4, 5, 6]
        ], dtype=torch.float32)

    def test_none_communicator(self):
        comm = ht.core.communicator.NoneCommunicator()

        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=2)
        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=-3)

        offset, lshape, chunks = comm.chunk(self.data.shape, split=0)

        self.assertIsInstance(offset, int)
        self.assertEqual(offset, 0)

        self.assertIsInstance(lshape, tuple)
        self.assertEqual(len(lshape), len(self.data.shape))
        self.assertEqual(lshape, self.data.shape)

        self.assertIsInstance(chunks, tuple)
        self.assertEqual(len(chunks), len(self.data.shape))
        self.assertEqual(1, (self.data == self.data[chunks]).all().item())

    def test_mpi_communicator(self):
        comm = ht.core.communicator.MPICommunicator()
        self.assertLess(comm.rank, comm.size)

        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=2)
        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=-3)

        offset, lshape, chunks = comm.chunk(self.data.shape, split=0)
        self.assertIsInstance(offset, int)
        self.assertGreaterEqual(offset, 0)
        self.assertLess(offset, self.data.shape[0])

        self.assertIsInstance(lshape, tuple)
        self.assertEqual(len(lshape), len(self.data.shape))
        self.assertGreaterEqual(lshape[0], 0)
        self.assertLessEqual(lshape[0], self.data.shape[0])

        self.assertIsInstance(chunks, tuple)
        self.assertEqual(len(chunks), len(self.data.shape))
