import unittest

import heat as ht
import numpy as np


class TestTensor(unittest.TestCase):
    def test_rand(self):
        # int64 tests

        # Resetting seed works
        seed = 12345
        ht.random.seed(seed)
        a = ht.random.rand(2, 5, 7, 3, split=0, comm=ht.MPI_WORLD)
        b = ht.random.rand(2, 5, 7, 3, split=0, comm=ht.MPI_WORLD)
        self.assertFalse(ht.equal(a, b))
        ht.random.seed(seed)
        c = ht.random.rand(2, 5, 7, 3, split=0, comm=ht.MPI_WORLD)
        self.assertTrue(ht.equal(a, c))

        # Random numbers with overflow
        ht.random.set_state(('Threefry', seed, 0xfffffffffffffff0))
        a = ht.random.rand(2, 3, 4, 5, split=0, comm=ht.MPI_WORLD)
        ht.random.set_state(('Threefry', seed, 0x10000000000000000))
        b = ht.random.rand(2, 44, split=0, comm=ht.MPI_WORLD)
        a = a.numpy().flatten()
        b = b.numpy().flatten()
        self.assertTrue(np.array_equal(a[32:], b))

        # Check that random numbers don't repeat after first overflow
        seed = 12345
        ht.random.set_state(('Threefry', seed, 0x10000000000000000))
        a = ht.random.rand(2, 44)
        ht.random.seed(seed)
        b = ht.random.rand(2, 44)
        self.assertFalse(ht.equal(a, b))

        # Check that we start from beginning after 128 bit overflow
        ht.random.seed(seed)
        a = ht.random.rand(2, 34, split=0)
        ht.random.set_state(('Threefry', seed, 0xfffffffffffffffffffffffffffffff0))
        b = ht.random.rand(2, 50, split=0)
        a = a.numpy().flatten()
        b = b.numpy(). flatten()
        self.assertTrue(np.array_equal(a, b[32:]))

        # different split axis with resetting seed
        ht.random.seed(seed)
        a = ht.random.rand(3, 5, 2, 9, split=3, comm=ht.MPI_WORLD)
        ht.random.seed(seed)
        c = ht.random.rand(3, 5, 2, 9, split=3, comm=ht.MPI_WORLD)
        self.assertTrue(ht.equal(a, c))

        # Random values are in correct order
        ht.random.seed(seed)
        a = ht.random.rand(2, 50, split=0)
        ht.random.seed(seed)
        b = ht.random.rand(100, split=None)
        a = a.numpy().flatten()
        b = b._DNDarray__array.numpy()
        self.assertTrue(np.array_equal(a, b))

        # On different shape and split the same random values are used
        ht.random.seed(seed)
        a = ht.random.rand(3, 5, 2, 9, split=3, comm=ht.MPI_WORLD)
        ht.random.seed(seed)
        b = ht.random.rand(30, 9, split=1, comm=ht.MPI_WORLD)
        a = np.sort(a.numpy().flatten())
        b = np.sort(b.numpy().flatten())
        self.assertTrue(np.array_equal(a, b))

        # One large array does not have two similar values
        a = ht.random.rand(11, 15, 3, 7, split=2, comm=ht.MPI_WORLD)
        a = a.numpy()
        _, counts = np.unique(a, return_counts=True)
        self.assertTrue((counts == 1).all())    # Assert that no value appears more than once

        # Two large arrays that were created after each other don't share any values
        b = ht.random.rand(14, 7, 3, 12, 18, 42, split=5, comm=ht.MPI_WORLD)
        c = np.concatenate((a.flatten(), b.numpy().flatten()))
        _, counts = np.unique(c, return_counts=True)
        self.assertTrue((counts == 1).all())

    def test_randint(self):
        pass

    def test_randn(self):
        pass
