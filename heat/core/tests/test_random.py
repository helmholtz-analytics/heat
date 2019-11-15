import unittest
import os

import torch

import heat as ht
import numpy as np

ht.use_device(os.environ.get("DEVICE"))


class TestRandom(unittest.TestCase):
    def test_rand(self):
        # int64 tests

        # Resetting seed works
        seed = 12345
        ht.random.seed(seed)
        a = ht.random.rand(2, 5, 7, 3, split=0, comm=ht.MPI_WORLD)
        self.assertEqual(a.dtype, ht.float64)
        self.assertEqual(a._DNDarray__array.dtype, torch.float64)
        b = ht.random.rand(2, 5, 7, 3, split=0, comm=ht.MPI_WORLD)
        self.assertFalse(ht.equal(a, b))
        ht.random.seed(seed)
        c = ht.random.rand(2, 5, 7, 3, dtype=ht.float64, split=0, comm=ht.MPI_WORLD)
        self.assertTrue(ht.equal(a, c))

        # Random numbers with overflow
        ht.random.set_state(("Threefry", seed, 0xFFFFFFFFFFFFFFF0))
        a = ht.random.rand(2, 3, 4, 5, split=0, comm=ht.MPI_WORLD)
        ht.random.set_state(("Threefry", seed, 0x10000000000000000))
        b = ht.random.rand(2, 44, split=0, comm=ht.MPI_WORLD)
        a = a.numpy().flatten()
        b = b.numpy().flatten()
        self.assertEqual(a.dtype, np.float64)
        self.assertTrue(np.array_equal(a[32:], b))

        # Check that random numbers don't repeat after first overflow
        seed = 12345
        ht.random.set_state(("Threefry", seed, 0x10000000000000000))
        a = ht.random.rand(2, 44)
        ht.random.seed(seed)
        b = ht.random.rand(2, 44)
        self.assertFalse(ht.equal(a, b))

        # Check that we start from beginning after 128 bit overflow
        ht.random.seed(seed)
        a = ht.random.rand(2, 34, split=0)
        ht.random.set_state(("Threefry", seed, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0))
        b = ht.random.rand(2, 50, split=0)
        a = a.numpy().flatten()
        b = b.numpy().flatten()
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
        b = b._DNDarray__array.cpu().numpy()
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
        # Assert that no value appears more than once
        self.assertTrue((counts == 1).all())

        # Two large arrays that were created after each other don't share any values
        b = ht.random.rand(14, 7, 3, 12, 18, 42, split=5, comm=ht.MPI_WORLD)
        c = np.concatenate((a.flatten(), b.numpy().flatten()))
        _, counts = np.unique(c, return_counts=True)
        self.assertTrue((counts == 1).all())

        # Values should be spread evenly across the range [0, 1)
        mean = np.mean(c)
        median = np.median(c)
        std = np.std(c)
        self.assertTrue(0.49 < mean < 0.51)
        self.assertTrue(0.49 < median < 0.51)
        self.assertTrue(std < 0.3)
        self.assertTrue(((0 <= c) & (c < 1)).all())

        # No arguments work correctly
        ht.random.seed(seed)
        a = ht.random.rand()
        ht.random.seed(seed)
        b = ht.random.rand(1)
        self.assertTrue(ht.equal(a, b))

        # To big arrays cant be created
        with self.assertRaises(ValueError):
            ht.random.randn(0xFFFFFFFFFFFFFFFF * 2 + 1, comm=ht.MPI_WORLD)
        with self.assertRaises(ValueError):
            ht.random.rand(3, 2, -2, 5, split=1, comm=ht.MPI_WORLD)
        with self.assertRaises(ValueError):
            ht.random.randn(12, 43, dtype=ht.int32, split=0, comm=ht.MPI_WORLD)

        # 32 Bit tests
        ht.random.seed(9876)
        shape = (13, 43, 13, 23)
        a = ht.random.rand(*shape, dtype=ht.float32, split=0, comm=ht.MPI_WORLD)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a._DNDarray__array.dtype, torch.float32)

        ht.random.seed(9876)
        b = ht.random.rand(np.prod(shape), dtype=ht.float32, comm=ht.MPI_WORLD)
        a = a.numpy().flatten()
        b = b._DNDarray__array.cpu().numpy()
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.dtype, np.float32)

        a = ht.random.rand(21, 16, 17, 21, dtype=ht.float32, split=2, comm=ht.MPI_WORLD)
        b = ht.random.rand(15, 11, 19, 31, dtype=ht.float32, split=0, comm=ht.MPI_WORLD)
        a = a.numpy().flatten()
        b = b.numpy().flatten()
        c = np.concatenate((a, b))

        # Values should be spread evenly across the range [0, 1)
        mean = np.mean(c)
        median = np.median(c)
        std = np.std(c)
        self.assertTrue(0.49 < mean < 0.51)
        self.assertTrue(0.49 < median < 0.51)
        self.assertTrue(std < 0.3)
        self.assertTrue(((0 <= c) & (c < 1)).all())

        ht.random.seed(11111)
        a = ht.random.rand(12, 32, 44, split=1, dtype=ht.float32, comm=ht.MPI_WORLD).numpy()
        # Overflow reached
        ht.random.set_state(("Threefry", 11111, 0x10000000000000000))
        b = ht.random.rand(12, 32, 44, split=1, dtype=ht.float32, comm=ht.MPI_WORLD).numpy()
        self.assertTrue(np.array_equal(a, b))

        ht.random.set_state(("Threefry", 11111, 0x100000000))
        c = ht.random.rand(12, 32, 44, split=1, dtype=ht.float32, comm=ht.MPI_WORLD).numpy()
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(b, c))

    def test_randint(self):
        # Checked that the random values are in the correct range
        a = ht.random.randint(low=0, high=10, size=(10, 10))
        self.assertEqual(a.dtype, ht.int64)
        a = a.numpy()
        self.assertTrue(((0 <= a) & (a < 10)).all())

        a = ht.random.randint(low=100000, high=150000, size=(31, 25, 11), split=2)
        a = a.numpy()
        self.assertTrue(((100000 <= a) & (a < 150000)).all())

        # For the range [0, 1) only the value 0 is allowed
        a = ht.random.randint(1, size=(10,), split=0)
        b = ht.zeros((10,), dtype=ht.int64, split=0)
        self.assertTrue(ht.equal(a, b))

        # Two arrays with the same seed and same number of elements have the same random values
        ht.random.seed(13579)
        shape = (15, 13, 9, 21, 65)
        a = ht.random.randint(15, 100, size=shape, split=0)
        a = a.numpy().flatten()

        ht.random.seed(13579)
        elements = np.prod(shape)
        b = ht.random.randint(low=15, high=100, size=(elements,))
        b = b.numpy()
        self.assertTrue(np.array_equal(a, b))

        # Two arrays with the same seed and shape have identical values
        ht.random.seed(13579)
        a = ht.random.randint(10000, size=shape, split=2)
        a = a.numpy()

        ht.random.seed(13579)
        b = ht.random.randint(low=0, high=10000, size=shape, split=2)
        b = b.numpy()

        ht.random.seed(13579)
        c = ht.random.randint(low=0, high=10000)
        self.assertTrue(np.equal(b[0, 0, 0, 0, 0], c))

        self.assertTrue(np.array_equal(a, b))
        mean = np.mean(a)
        median = np.median(a)
        std = np.std(a)

        # Mean and median should be in the center while the std is very high due to an even distribution
        self.assertTrue(4900 < mean < 5100)
        self.assertTrue(4900 < median < 5100)
        self.assertTrue(std < 2900)

        with self.assertRaises(ValueError):
            ht.random.randint(5, 5, size=(10, 10), split=0)
        with self.assertRaises(ValueError):
            ht.random.randint(low=0, high=10, size=(3, -4))
        with self.assertRaises(ValueError):
            ht.random.randint(low=0, high=10, size=(15,), dtype=ht.float32)

        # int32 tests
        ht.random.seed(4545)
        a = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0, comm=ht.MPI_WORLD)
        ht.random.set_state(("Threefry", 4545, 0x10000000000000000))
        b = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0, comm=ht.MPI_WORLD)

        self.assertEqual(a.dtype, ht.int32)
        self.assertEqual(a._DNDarray__array.dtype, torch.int32)
        self.assertEqual(b.dtype, ht.int32)
        a = a.numpy()
        b = b.numpy()
        self.assertEqual(a.dtype, np.int32)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(((50 <= a) & (a < 1000)).all())
        self.assertTrue(((50 <= b) & (b < 1000)).all())

        c = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0, comm=ht.MPI_WORLD)
        c = c.numpy()
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(b, c))
        self.assertTrue(((50 <= c) & (c < 1000)).all())

        ht.random.seed(0xFFFFFFF)
        a = ht.random.randint(
            10000, size=(123, 42, 13, 21), split=3, dtype=ht.int32, comm=ht.MPI_WORLD
        )
        a = a.numpy()
        mean = np.mean(a)
        median = np.median(a)
        std = np.std(a)

        # Mean and median should be in the center while the std is very high due to an even distribution
        self.assertTrue(4900 < mean < 5100)
        self.assertTrue(4900 < median < 5100)
        self.assertTrue(std < 2900)

    def test_randn(self):
        # Test that the random values have the correct distribution
        ht.random.seed(54321)
        shape = (5, 10, 13, 23, 15, 20)
        a = ht.random.randn(*shape, split=0)
        self.assertEqual(a.dtype, ht.float64)
        a = a.numpy()
        mean = np.mean(a)
        median = np.median(a)
        std = np.std(a)
        self.assertTrue(-0.01 < mean < 0.01)
        self.assertTrue(-0.01 < median < 0.01)
        self.assertTrue(0.99 < std < 1.01)

        # Compare to a second array with a different shape but same number of elements and same seed
        ht.random.seed(54321)
        elements = np.prod(shape)
        b = ht.random.randn(elements, split=0)
        b = b.numpy()
        a = a.flatten()
        self.assertTrue(np.allclose(a, b))

        # Creating the same array two times without resetting seed results in different elements
        c = ht.random.randn(elements, split=0)
        c = c.numpy()
        self.assertEqual(c.shape, b.shape)
        self.assertFalse(np.allclose(b, c))

        # All the created values should be different
        d = np.concatenate((b, c))
        _, counts = np.unique(d, return_counts=True)
        self.assertTrue((counts == 1).all())

        # Two arrays are the same for same seed and split-axis != 0
        ht.random.seed(12345)
        a = ht.random.randn(*shape, split=5)
        ht.random.seed(12345)
        b = ht.random.randn(*shape, split=5)
        self.assertTrue(ht.equal(a, b))
        a = a.numpy()
        b = b.numpy()
        self.assertTrue(np.allclose(a, b))

        # Tests with float32
        ht.random.seed(54321)
        a = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2, comm=ht.MPI_WORLD)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a._DNDarray__array[0, 0, 0].dtype, torch.float32)
        a = a.numpy()
        self.assertEqual(a.dtype, np.float32)
        mean = np.mean(a)
        median = np.median(a)
        std = np.std(a)
        self.assertTrue(-0.01 < mean < 0.01)
        self.assertTrue(-0.01 < median < 0.01)
        self.assertTrue(0.99 < std < 1.01)

        ht.random.set_state(("Threefry", 54321, 0x10000000000000000))
        b = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2, comm=ht.MPI_WORLD).numpy()
        self.assertTrue(np.allclose(a, b))

        c = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2, comm=ht.MPI_WORLD).numpy()
        self.assertFalse(np.allclose(a, c))
        self.assertFalse(np.allclose(b, c))

    def test_set_state(self):
        ht.random.set_state(("Threefry", 12345, 0xFFF))
        self.assertEqual(ht.random.get_state(), ("Threefry", 12345, 0xFFF, 0, 0.0))

        ht.random.set_state(("Threefry", 55555, 0xFFFFFFFFFFFFFF, "for", "compatibility"))
        self.assertEqual(ht.random.get_state(), ("Threefry", 55555, 0xFFFFFFFFFFFFFF, 0, 0.0))

        with self.assertRaises(ValueError):
            ht.random.set_state(("Thrfry", 12, 0xF))
        with self.assertRaises(TypeError):
            ht.random.set_state(("Threefry", 12345))
