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

    def test_mpi_communicator(self):
        comm = ht.core.communicator.MPICommunicator()
        self.assertLess(comm.rank, comm.size)

        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=2)
        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=-3)

        chunks = comm.chunk(self.data.shape, split=0)
        self.assertIsInstance(chunks, tuple)
        self.assertEqual(len(chunks), len(self.data.shape))

    def test_none_communicator(self):
        comm = ht.core.communicator.NoneCommunicator()

        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=2)
        with self.assertRaises(ValueError):
            comm.chunk(self.data.shape, split=-3)

        chunks = comm.chunk(self.data.shape, split=0)
        self.assertIsInstance(chunks, tuple)
        self.assertEqual(len(chunks), len(self.data.shape))
        self.assertEqual(1, (self.data == self.data[chunks]).all().item())
