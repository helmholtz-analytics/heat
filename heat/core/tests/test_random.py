import os
import platform
import unittest

import numpy as np
import torch

import heat as ht
from .test_suites.basic_test import TestCase

envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
is_mps = envar == "gpu" and platform.system() == "Darwin"


class TestRandom_Batchparallel(TestCase):
    def test_default(self):
        # test the default
        state = ht.random.get_state()
        self.assertEqual(state[0], "Batchparallel")
        self.assertEqual(state[2], state[1] + ht.MPI_WORLD.rank)

    def test_normal(self):
        shape = (3, 4, 6)
        ht.random.seed(2)
        gnormal = ht.random.normal(shape=shape, split=2)
        ht.random.seed(2)
        snormal = ht.random.randn(*shape, split=2)

        self.assertEqual(gnormal.dtype, snormal.dtype)
        self.assertEqual(gnormal.shape, snormal.shape)
        self.assertEqual(gnormal.device, snormal.device)
        self.assertTrue(ht.equal(gnormal, snormal))

        shape = (2, 2)
        mu = ht.array([[-1, -0.5], [0, 5]])
        sigma = ht.array([[0, 0.5], [1, 2.5]])

        ht.random.seed(22)
        gnormal = ht.random.normal(mu, sigma, shape)
        ht.random.seed(22)
        snormal = ht.random.randn(*shape)

        compare = mu + sigma * snormal

        self.assertEqual(gnormal.dtype, compare.dtype)
        self.assertEqual(gnormal.shape, compare.shape)
        self.assertEqual(gnormal.device, compare.device)
        self.assertTrue(ht.equal(gnormal, compare))

        with self.assertRaises(TypeError):
            ht.random.normal([4, 5], 1, shape)
        with self.assertRaises(TypeError):
            ht.random.normal(0, "r", shape)
        with self.assertRaises(ValueError):
            ht.random.normal(0, -1, shape)

    def test_permutation(self):
        # Reset RNG
        ht.random.seed()
        if self.device.torch_device == "cpu":
            state = torch.random.get_rng_state()
        else:
            if self.is_mps:
                state = torch.mps.get_rng_state()
            else:
                state = torch.cuda.get_rng_state(self.device.torch_device)

        # results
        a = ht.random.permutation(10, device=self.device)

        b_arr = ht.arange(10, dtype=ht.float32)
        b = ht.random.permutation(ht.resplit(b_arr, 0))

        c_arr = ht.arange(16).reshape((4, 4))
        c = ht.random.permutation(c_arr)

        c0 = ht.random.permutation(ht.resplit(c_arr, 0))
        c1 = ht.random.permutation(ht.resplit(c_arr, 1))

        if self.device.torch_device == "cpu":
            torch.random.set_rng_state(state)
        else:
            if self.is_mps:
                torch.mps.set_rng_state(state)
            else:
                torch.cuda.set_rng_state(state, self.device.torch_device)

        # torch results to compare to
        a_cmp = torch.randperm(a.shape[0], device=self.device.torch_device)
        b_cmp = b_arr.larray[torch.randperm(b.shape[0], device=self.device.torch_device)]
        c_cmp = c_arr.larray[torch.randperm(c.shape[0], device=self.device.torch_device)]
        c0_cmp = c_arr.larray[torch.randperm(c.shape[0], device=self.device.torch_device)]
        c1_cmp = c_arr.larray[torch.randperm(c.shape[0], device=self.device.torch_device)]

        # compare
        self.assertEqual(a.dtype, ht.int64)
        self.assertEqual(b.dtype, ht.float32)

        if not self.is_mps:
            c0.resplit_(None)
            c1.resplit_(None)
            b.resplit_(None)

            # due to different states of the torch RNG on different processes and due to construction of the permutation
            # the values are only equal on process no 0 which has been used for generating the permutation
            if ht.MPI_WORLD.rank == 0:
                self.assertTrue((a.larray == a_cmp).all())
                self.assertTrue((b.larray == b_cmp).all())
                self.assertTrue((c.larray == c_cmp).all())
                self.assertTrue((c0.larray == c0_cmp).all())
                self.assertTrue((c1.larray == c1_cmp).all())

        with self.assertRaises(TypeError):
            ht.random.permutation("abc")

    def test_rand(self):
        # int64 tests

        # Resetting seed works
        seed = 123456
        ht.random.seed(seed)
        a = ht.random.rand(2, 5, 7, 3, split=0)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a.larray.dtype, torch.float32)
        b = ht.random.rand(2, 5, 7, 3, split=0)
        self.assertFalse(ht.equal(a, b))
        ht.random.seed(seed)
        c = ht.random.rand(2, 5, 7, 3, dtype=ht.float32, split=0)
        self.assertTrue(ht.equal(a, c))

        # One large array does not have too much similar values
        a = ht.random.rand(11, 15, 13, 17, split=2)
        a = a.numpy()
        _, counts = np.unique(a, return_counts=True)
        # Assert that no value appears more than once
        self.assertTrue((counts <= 2).all())

        # Two large arrays that were created after each other don't share too much values
        if not self.is_mps:
            # this condition is not met if b is float32, MPS does not support float64
            b = ht.random.rand(14, 7, 3, 12, 18, 42, split=5, comm=ht.MPI_WORLD, dtype=ht.float64)
            c = np.concatenate((a.flatten(), b.numpy().flatten()))
            _, counts = np.unique(c, return_counts=True)
            self.assertTrue((counts <= 2).all())

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
        self.assertTrue(isinstance(a, float))
        ht.random.seed(seed)
        b = ht.random.rand(1)
        self.assertTrue(ht.equal(a, b))

        # 32 Bit tests
        ht.random.seed(9876)
        shape = (13, 43, 13, 23)
        a = ht.random.rand(*shape, dtype=ht.float32, split=0)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a.larray.dtype, torch.float32)

        a = ht.random.rand(21, 16, 17, 21, dtype=ht.float32, split=2)
        b = ht.random.rand(15, 11, 19, 31, dtype=ht.float32, split=0)
        a = a.flatten()
        b = b.flatten()
        c = ht.concatenate((a, b))

        # Values should be spread evenly across the range [0, 1)
        mean = ht.mean(c)
        # median = np.median(c)
        std = ht.std(c)
        self.assertTrue(0.49 < mean < 0.51)
        # self.assertTrue(0.49 < median < 0.51)
        self.assertTrue(std < 0.3)
        self.assertTrue(((0 <= c) & (c < 1)).all())

    def test_randint(self):
        # Checked that the random values are in the correct range
        a = ht.random.randint(low=0, high=10, size=(10, 10), dtype=ht.int64)
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(((0 <= a) & (a < 10)).all())

        a = ht.random.randint(low=100000, high=150000, size=(31, 25, 11), dtype=ht.int64, split=2)
        self.assertTrue(((100000 <= a) & (a < 150000)).all())

        # For the range [0, 1) only the value 0 is allowed
        a = ht.random.randint(1, size=(10,), split=0, dtype=ht.int64)
        b = ht.zeros((10,), dtype=ht.int64, split=0)
        self.assertTrue(ht.equal(a, b))

        # size parameter allows int arguments
        a = ht.random.randint(1, size=10, split=0, dtype=ht.int64)
        self.assertTrue(ht.equal(a, b))

        # size is None
        a = ht.random.randint(0, 10)
        self.assertEqual(a.shape, ())

        # Two arrays with the same seed and shape have identical values
        shape = (15, 13, 9, 21, 65)
        ht.random.seed(13579)
        a = ht.random.randint(10000, size=shape, split=2, dtype=ht.int64)

        ht.random.seed(13579)
        b = ht.random.randint(low=0, high=10000, size=shape, split=2, dtype=ht.int64)

        if not self.is_mps:
            # assertion fails on more than 4 dimensions on MPS
            self.assertTrue(ht.equal(a, b))
        mean = ht.mean(a)
        # median = ht.median(a)
        std = ht.std(a)

        # Mean and median should be in the center while the std is very high due to an even distribution
        self.assertTrue(4900 < mean < 5100)
        # self.assertTrue(4900 < median < 5100)
        self.assertTrue(std < 2900)

        with self.assertRaises(ValueError):
            ht.random.randint(5, 5, size=(10, 10), split=0)
        with self.assertRaises(ValueError):
            ht.random.randint(low=0, high=10, size=(3, -4))
        with self.assertRaises(ValueError):
            ht.random.randint(low=0, high=10, size=(15,), dtype=ht.float32)

        # int32 tests
        ht.random.seed(4545)
        a = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0)
        ht.random.set_state(("Batchparallel", 4545, None))
        b = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0)

        self.assertEqual(a.dtype, ht.int32)
        self.assertEqual(a.larray.dtype, torch.int32)
        self.assertEqual(b.dtype, ht.int32)
        self.assertTrue(ht.equal(a, b))
        self.assertTrue(((50 <= a) & (a < 1000)).all())
        self.assertTrue(((50 <= b) & (b < 1000)).all())

        c = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0)
        self.assertFalse(ht.equal(a, c))
        self.assertFalse(ht.equal(b, c))
        self.assertTrue(((50 <= c) & (c < 1000)).all())

        ht.random.seed(0xFFFFFFF)
        a = ht.random.randint(
            10000, size=(123, 42, 13, 21), split=3, dtype=ht.int32, comm=ht.MPI_WORLD
        )
        mean = ht.mean(a)
        # median = np.median(a)
        std = ht.std(a)

        # Mean and median should be in the center while the std is very high due to an even distribution
        self.assertTrue(4900 < mean < 5100)
        # self.assertTrue(4900 < median < 5100)
        self.assertTrue(std < 2900)

        # test aliases
        ht.random.seed(234)
        a = ht.random.randint(10, 50)
        ht.random.seed(234)
        b = ht.random.random_integer(10, 50)
        self.assertTrue(ht.equal(a, b))

    def test_randn(self):
        float_dtype = ht.float32 if self.is_mps else ht.float64
        # Test that the random values have the correct distribution
        ht.random.seed(54321)
        shape = (5, 10, 13, 23)
        a = ht.random.randn(*shape, split=0, dtype=float_dtype)
        self.assertEqual(a.dtype, float_dtype)
        mean = ht.mean(a)
        median = ht.median(a)
        std = ht.std(a)
        self.assertTrue(-0.02 < mean < 0.02)
        self.assertTrue(-0.02 < median < 0.02)
        self.assertTrue(0.98 < std < 1.02)

        # Creating the same array two times without resetting seed results in different elements
        c = ht.random.randn(*shape, split=0, dtype=float_dtype)
        self.assertEqual(c.shape, a.shape)
        self.assertFalse(ht.allclose(a, c))

        if not self.is_mps:
            # If dtype is float64, all the created values should be different
            d = ht.concatenate((a, c))
            d.resplit_(None)
            d = d.numpy()
            _, counts = np.unique(d, return_counts=True)
            self.assertTrue((counts == 1).all())

        # Two arrays are the same for same seed and split-axis != 0
        ht.random.seed(12345)
        a = ht.random.randn(*shape, split=3, dtype=float_dtype)
        ht.random.seed(12345)
        b = ht.random.randn(*shape, split=3, dtype=float_dtype)
        self.assertTrue(ht.equal(a, b))

        # Tests with float32
        ht.random.seed(272)
        a = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a.larray[0, 0, 0].dtype, torch.float32)
        mean = ht.mean(a)
        # median = np.median(a)
        std = ht.std(a)
        self.assertTrue(-0.02 < mean < 0.02)
        # self.assertTrue(-0.02 < median < 0.02)
        self.assertTrue(0.99 < std < 1.01)

        ls = 272 + ht.MPI_WORLD.rank
        ht.random.set_state(("Batchparallel", None, ls))
        b = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2)
        self.assertTrue(ht.allclose(a, b))

        c = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2)
        self.assertFalse(ht.allclose(a, c))
        self.assertFalse(ht.allclose(b, c))

        # check wrong shapes
        with self.assertRaises(ValueError):
            ht.random.randn(2, -1, 2)

        # test generation of a single number
        x = ht.random.randn()
        self.assertTrue(isinstance(x, float))

    def test_randperm(self):
        # Reset RNG
        ht.random.seed()
        if self.device.torch_device == "cpu":
            state = torch.random.get_rng_state()
        else:
            if self.is_mps:
                state = torch.mps.get_rng_state()
            else:
                state = torch.cuda.get_rng_state(self.device.torch_device)

        # results
        a = ht.random.randperm(10, dtype=ht.int32)
        b = ht.random.randperm(4, dtype=ht.float32, split=0)
        c = ht.random.randperm(5, split=0)
        if not self.is_mps:
            d = ht.random.randperm(5, dtype=ht.float64)

        if self.device.torch_device == "cpu":
            torch.random.set_rng_state(state)
        else:
            if self.is_mps:
                torch.mps.set_rng_state(state)
            else:
                torch.cuda.set_rng_state(state, self.device.torch_device)

        # torch results to compare to
        a_cmp = torch.randperm(10, dtype=torch.int32, device=a.larray.device)
        b_cmp = torch.randperm(4, dtype=torch.float32, device=self.device.torch_device)
        c_cmp = torch.randperm(5, dtype=torch.int64, device=self.device.torch_device)
        if not self.is_mps:
            d_cmp = torch.randperm(5, dtype=torch.float64, device=self.device.torch_device)

        self.assertEqual(a.dtype, ht.int32)
        self.assertEqual(b.dtype, ht.float32)
        self.assertEqual(c.dtype, ht.int64)
        if not self.is_mps:
            self.assertEqual(d.dtype, ht.float64)
        brsp = ht.resplit(b)
        crsp = ht.resplit(c)

        # due to different states of the torch RNG on different processes and due to construction of the permutation
        # the values are only equal on process no 0 which has been used for generating the permutation
        if ht.MPI_WORLD.rank == 0:
            self.assertTrue((a.larray == a_cmp).all())
            self.assertTrue((brsp.larray == b_cmp).all())
            self.assertTrue((crsp.larray == c_cmp).all())
            if not self.is_mps:
                self.assertTrue((d.larray == d_cmp).all())

        with self.assertRaises(TypeError):
            ht.random.randperm("abc")

    def test_random_sample(self):
        # short test
        # compare random and aliases with rand
        ht.random.seed(534)
        a = ht.random.rand(6, 2, 3)
        ht.random.seed(534)
        b = ht.random.random((6, 2, 3))
        ht.random.seed(534)
        c = ht.random.random_sample((6, 2, 3))
        ht.random.seed(534)
        d = ht.random.ranf((6, 2, 3))
        ht.random.seed(534)
        e = ht.random.sample((6, 2, 3))

        self.assertTrue(ht.equal(a, b))
        self.assertTrue(ht.equal(a, c))
        self.assertTrue(ht.equal(a, d))
        self.assertTrue(ht.equal(a, e))

        # empty input
        a = ht.random.random_sample()
        self.assertEqual(a.shape, (1,))

    def test_standard_normal(self):
        # empty input
        stdn = ht.random.standard_normal()
        self.assertEqual(stdn.dtype, ht.float32)
        self.assertEqual(stdn.shape, (1,))

        # simple test
        shape = (3, 4, 6)
        ht.random.seed(11235)
        stdn = ht.random.standard_normal(shape, split=2)
        ht.random.seed(11235)
        rndn = ht.random.randn(*shape, split=2)

        self.assertEqual(stdn.shape, rndn.shape)
        self.assertEqual(stdn.dtype, rndn.dtype)
        self.assertEqual(stdn.device, rndn.device)
        self.assertTrue(ht.equal(stdn, rndn))

    def test_set_state(self):
        # test if setting the state ignores local seeds if global seed is provided
        ht.random.set_state(("Batchparallel", 321, 10))
        self.assertEqual(
            ht.random.get_state(), ("Batchparallel", 321, 321 + ht.MPI_WORLD.rank, 0, 0.0)
        )

        # test local seeds
        ht.random.set_state(("Batchparallel", None, ht.MPI_WORLD.rank))
        self.assertEqual(ht.random.get_state(), ("Batchparallel", None, ht.MPI_WORLD.rank, 0, 0.0))


"""
Tests for Threefry RNG
"""


@unittest.skipIf(is_mps, "Threefry not supported on Apple MPS")
class TestRandom_Threefry(TestCase):
    def test_setting_threefry(self):
        ht.random.set_state(("Threefry", 12345, 0xFFF))
        self.assertEqual(ht.random.get_state(), ("Threefry", 12345, 0xFFF, 0, 0.0))

        ht.random.set_state(("Threefry", 55555, 0xFFFFFFFFFFFFFF, "for", "compatibility"))
        self.assertEqual(ht.random.get_state(), ("Threefry", 55555, 0xFFFFFFFFFFFFFF, 0, 0.0))

        with self.assertRaises(ValueError):
            ht.random.set_state(("Thrfry", 12, 0xF))
        with self.assertRaises(TypeError):
            ht.random.set_state(("Threefry", 12345))

    def test_normal(self):
        ht.random.set_state(("Threefry", 0, 0))
        ht.random.seed()
        shape = (3, 4, 6)
        ht.random.seed(2)
        gnormal = ht.random.normal(shape=shape, split=2)
        ht.random.seed(2)
        snormal = ht.random.randn(*shape, split=2)

        self.assertEqual(gnormal.dtype, snormal.dtype)
        self.assertEqual(gnormal.shape, snormal.shape)
        self.assertEqual(gnormal.device, snormal.device)
        self.assertTrue(ht.equal(gnormal, snormal))

        shape = (2, 2)
        mu = ht.array([[-1, -0.5], [0, 5]])
        sigma = ht.array([[0, 0.5], [1, 2.5]])

        ht.random.seed(22)
        gnormal = ht.random.normal(mu, sigma, shape)
        ht.random.seed(22)
        snormal = ht.random.randn(*shape)

        compare = mu + sigma * snormal

        self.assertEqual(gnormal.dtype, compare.dtype)
        self.assertEqual(gnormal.shape, compare.shape)
        self.assertEqual(gnormal.device, compare.device)
        self.assertTrue(ht.equal(gnormal, compare))

        with self.assertRaises(TypeError):
            ht.random.normal([4, 5], 1, shape)
        with self.assertRaises(TypeError):
            ht.random.normal(0, "r", shape)
        with self.assertRaises(ValueError):
            ht.random.normal(0, -1, shape)

    def test_permutation(self):
        # Reset RNG
        ht.random.set_state(("Threefry", 0, 0))
        ht.random.seed()
        if self.device.torch_device == "cpu":
            state = torch.random.get_rng_state()
        else:
            state = torch.cuda.get_rng_state(self.device.torch_device)

        # results
        a = ht.random.permutation(10)

        b_arr = ht.arange(10, dtype=ht.float32)
        b = ht.random.permutation(ht.resplit(b_arr, 0))

        c_arr = ht.arange(16).reshape((4, 4))
        c = ht.random.permutation(c_arr)

        c0 = ht.random.permutation(ht.resplit(c_arr, 0))
        c1 = ht.random.permutation(ht.resplit(c_arr, 1))

        if self.device.torch_device == "cpu":
            torch.random.set_rng_state(state)
        else:
            torch.cuda.set_rng_state(state, self.device.torch_device)

        # torch results to compare to
        a_cmp = torch.randperm(a.shape[0], device=self.device.torch_device)
        b_cmp = b_arr.larray[torch.randperm(b.shape[0], device=self.device.torch_device)]
        c_cmp = c_arr.larray[torch.randperm(c.shape[0], device=self.device.torch_device)]
        c0_cmp = c_arr.larray[torch.randperm(c.shape[0], device=self.device.torch_device)]
        c1_cmp = c_arr.larray[torch.randperm(c.shape[0], device=self.device.torch_device)]

        # compare
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue((a.larray == a_cmp).all())
        self.assertEqual(b.dtype, ht.float32)
        self.assertTrue((ht.resplit(b).larray == b_cmp).all())
        self.assertTrue((c.larray == c_cmp).all())
        self.assertTrue((ht.resplit(c0).larray == c0_cmp).all())
        self.assertTrue((ht.resplit(c1).larray == c1_cmp).all())

        with self.assertRaises(TypeError):
            ht.random.permutation("abc")

    def test_rand(self):
        ht.random.set_state(("Threefry", 0, 0))
        ht.random.seed()
        # int64 tests

        # Resetting seed works
        seed = 12345
        ht.random.seed(seed)
        a = ht.random.rand(2, 5, 7, 3, split=0)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a.larray.dtype, torch.float32)
        b = ht.random.rand(2, 5, 7, 3, split=0)
        self.assertFalse(ht.equal(a, b))
        ht.random.seed(seed)
        c = ht.random.rand(2, 5, 7, 3, dtype=ht.float32, split=0)
        self.assertTrue(ht.equal(a, c))

        # Random numbers with overflow
        ht.random.set_state(("Threefry", seed, 0xFFFFFFFFFFFFFFF0))
        a = ht.random.rand(2, 3, 4, 5, split=0)
        ht.random.set_state(("Threefry", seed, 0x10000000000000000))
        b = ht.random.rand(2, 44, split=0)
        a = a.flatten()
        b = b.flatten()
        self.assertTrue(ht.equal(a[32:], b))

        # Check that random numbers don't repeat after first overflow
        seed = 12345
        ht.random.set_state(("Threefry", seed, 0x100000000))
        a = ht.random.rand(2, 44)
        ht.random.seed(seed)
        b = ht.random.rand(2, 44)
        self.assertFalse(ht.equal(a, b))

        # Check that we start from beginning after 128 bit overflow
        ht.random.seed(seed)
        a = ht.random.rand(2, 34, split=0)
        ht.random.set_state(("Threefry", seed, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0))
        b = ht.random.rand(2, 50, split=0)
        a = a.flatten()
        b = b.flatten()
        self.assertTrue(ht.equal(a, b[32:]))

        # different split axis with resetting seed
        ht.random.seed(seed)
        a = ht.random.rand(3, 5, 2, 9, split=3)
        ht.random.seed(seed)
        c = ht.random.rand(3, 5, 2, 9, split=3)
        self.assertTrue(ht.equal(a, c))

        # Random values are in correct order
        ht.random.seed(seed)
        a = ht.random.rand(2, 50, split=0)
        ht.random.seed(seed)
        b = ht.random.rand(100, split=None)
        a = a.flatten()
        b = ht.resplit(b, 0)
        self.assertTrue(ht.equal(a, b))

        # On different shape and split the same random values are used
        ht.random.seed(seed)
        a = ht.random.rand(3, 5, 2, 9, split=3)
        ht.random.seed(seed)
        b = ht.random.rand(30, 9, split=1)
        a = np.sort(a.numpy().flatten())
        b = np.sort(b.numpy().flatten())
        self.assertTrue(np.array_equal(a, b))

        # One large array does not have two similar values
        a = ht.random.rand(11, 15, 3, 7, split=2)
        a = a.numpy()
        _, counts = np.unique(a, return_counts=True)
        # Assert that no value appears more than once
        self.assertTrue((counts == 1).all())

        if not (torch.cuda.is_available() and torch.version.hip):
            # Two large arrays that were created after each other don't share any values
            b = ht.random.rand(14, 7, 3, 12, 18, 42, split=5, comm=ht.MPI_WORLD, dtype=ht.float64)
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

        # Too big arrays cant be created
        with self.assertRaises(RuntimeError):
            ht.random.randn(0x7FFFFFFFFFFFFFFF)
        with self.assertRaises(ValueError):
            ht.random.rand(3, 2, -2, 5, split=1)
        with self.assertRaises(ValueError):
            ht.random.randn(12, 43, dtype=ht.int32, split=0)

        # 32 Bit tests
        ht.random.seed(9876)
        shape = (13, 43, 13, 23)
        a = ht.random.rand(*shape, dtype=ht.float32, split=0)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a.larray.dtype, torch.float32)

        ht.random.seed(9876)
        b = ht.random.rand(np.prod(shape), dtype=ht.float32)
        a = a.flatten()
        b = ht.resplit(b, 0)
        self.assertTrue(ht.equal(a, b))

        a = ht.random.rand(21, 16, 17, 21, dtype=ht.float32, split=2)
        b = ht.random.rand(15, 11, 19, 31, dtype=ht.float32, split=0)
        a = a.flatten()
        b = b.flatten()
        c = ht.concatenate((a, b))

        # Values should be spread evenly across the range [0, 1)
        mean = ht.mean(c)
        # median = np.median(c)
        std = ht.std(c)
        self.assertTrue(0.49 < mean < 0.51)
        # self.assertTrue(0.49 < median < 0.51)
        self.assertTrue(std < 0.3)
        self.assertTrue(((0 <= c) & (c < 1)).all())

        ht.random.seed(11111)
        a = ht.random.rand(12, 32, 44, split=1, dtype=ht.float32)
        # Overflow reached
        ht.random.set_state(("Threefry", 11111, 0x10000000000000000))
        b = ht.random.rand(12, 32, 44, split=1, dtype=ht.float32)
        self.assertTrue(ht.equal(a, b))

        ht.random.set_state(("Threefry", 11111, 0x100000000))
        c = ht.random.rand(12, 32, 44, split=1, dtype=ht.float32)
        self.assertFalse(ht.equal(a, c))
        self.assertFalse(ht.equal(b, c))

        # To check working with large number of elements
        ht.random.randn(6667, 3523, dtype=ht.float64, split=None)
        ht.random.randn(6667, 3523, dtype=ht.float64, split=0)
        ht.random.randn(6667, 3523, dtype=ht.float64, split=1)

    def test_randint(self):
        ht.random.set_state(("Threefry", 0, 0))
        ht.random.seed()
        # Checked that the random values are in the correct range
        a = ht.random.randint(low=0, high=10, size=(10, 10), dtype=ht.int64)
        self.assertEqual(a.dtype, ht.int64)
        self.assertTrue(((0 <= a) & (a < 10)).all())

        a = ht.random.randint(low=100000, high=150000, size=(31, 25, 11), dtype=ht.int64, split=2)
        self.assertTrue(((100000 <= a) & (a < 150000)).all())

        # For the range [0, 1) only the value 0 is allowed
        a = ht.random.randint(1, size=(10,), split=0, dtype=ht.int64)
        b = ht.zeros((10,), dtype=ht.int64, split=0)
        self.assertTrue(ht.equal(a, b))

        # size parameter allows int arguments
        a = ht.random.randint(1, size=10, split=0, dtype=ht.int64)
        self.assertTrue(ht.equal(a, b))

        # size is None
        a = ht.random.randint(0, 10)
        self.assertEqual(a.shape, ())

        # Two arrays with the same seed and same number of elements have the same random values
        ht.random.seed(13579)
        shape = (15, 13, 9, 21, 65)
        a = ht.random.randint(15, 100, size=shape, split=0, dtype=ht.int64)
        a = a.flatten()

        ht.random.seed(13579)
        elements = np.prod(shape)
        b = ht.random.randint(low=15, high=100, size=(elements,), dtype=ht.int64)
        self.assertTrue(ht.equal(a, b))

        # Two arrays with the same seed and shape have identical values
        ht.random.seed(13579)
        a = ht.random.randint(10000, size=shape, split=2, dtype=ht.int64)

        ht.random.seed(13579)
        b = ht.random.randint(low=0, high=10000, size=shape, split=2, dtype=ht.int64)

        ht.random.seed(13579)
        c = ht.random.randint(low=0, high=10000, dtype=ht.int64)
        self.assertTrue(ht.equal(b[0, 0, 0, 0, 0], c))

        self.assertTrue(ht.equal(a, b))
        mean = ht.mean(a)
        # median = np.median(a)
        std = ht.std(a)

        # Mean and median should be in the center while the std is very high due to an even distribution
        self.assertTrue(4900 < mean < 5100)
        # self.assertTrue(4900 < median < 5100)
        self.assertTrue(std < 2900)

        with self.assertRaises(ValueError):
            ht.random.randint(5, 5, size=(10, 10), split=0)
        with self.assertRaises(ValueError):
            ht.random.randint(low=0, high=10, size=(3, -4))
        with self.assertRaises(ValueError):
            ht.random.randint(low=0, high=10, size=(15,), dtype=ht.float32)

        # int32 tests
        ht.random.seed(4545)
        a = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0)
        ht.random.set_state(("Threefry", 4545, 0x10000000000000000))
        b = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0)

        self.assertEqual(a.dtype, ht.int32)
        self.assertEqual(a.larray.dtype, torch.int32)
        self.assertEqual(b.dtype, ht.int32)
        self.assertTrue(ht.equal(a, b))
        self.assertTrue(((50 <= a) & (a < 1000)).all())
        self.assertTrue(((50 <= b) & (b < 1000)).all())

        c = ht.random.randint(50, 1000, size=(13, 45), dtype=ht.int32, split=0)
        self.assertFalse(ht.equal(a, c))
        self.assertFalse(ht.equal(b, c))
        self.assertTrue(((50 <= c) & (c < 1000)).all())

        ht.random.seed(0xFFFFFFF)
        a = ht.random.randint(
            10000, size=(123, 42, 13, 21), split=3, dtype=ht.int32, comm=ht.MPI_WORLD
        )
        mean = ht.mean(a)
        #        median = np.median(a)
        std = ht.std(a)

        # Mean and median should be in the center while the std is very high due to an even distribution
        self.assertTrue(4900 < mean < 5100)
        # self.assertTrue(4900 < median < 5100)
        self.assertTrue(std < 2900)

        # test aliases
        ht.random.seed(234)
        a = ht.random.randint(10, 50)
        ht.random.seed(234)
        b = ht.random.random_integer(10, 50)
        self.assertTrue(ht.equal(a, b))

    def test_randn(self):
        ht.random.set_state(("Threefry", 0, 0))
        ht.random.seed()
        # Test that the random values have the correct distribution
        ht.random.seed(54321)
        shape = (5, 13, 23, 20)
        a = ht.random.randn(*shape, split=0, dtype=ht.float64)
        self.assertEqual(a.dtype, ht.float64)
        mean = ht.mean(a)
        median = ht.median(a)
        std = ht.std(a)
        self.assertTrue(-0.02 < mean < 0.02)
        self.assertTrue(-0.02 < median < 0.02)
        self.assertTrue(0.99 < std < 1.01)

        # Compare to a second array with a different shape but same number of elements and same seed
        ht.random.seed(54321)
        elements = np.prod(shape)
        b = ht.random.randn(elements, split=0, dtype=ht.float64)
        a = a.flatten()
        self.assertTrue(ht.allclose(a, b))

        # Creating the same array two times without resetting seed results in different elements
        c = ht.random.randn(elements, split=0, dtype=ht.float64)
        self.assertEqual(c.shape, b.shape)
        self.assertFalse(ht.allclose(b, c))

        # All the created values should be different
        d = ht.concatenate((b, c))
        d.resplit_(None)
        d = d.numpy()
        _, counts = np.unique(d, return_counts=True)
        self.assertTrue((counts == 1).all())

        # Two arrays are the same for same seed and split-axis != 0
        ht.random.seed(12345)
        a = ht.random.randn(*shape, split=3, dtype=ht.float64)
        ht.random.seed(12345)
        b = ht.random.randn(*shape, split=3, dtype=ht.float64)
        self.assertTrue(ht.equal(a, b))

        # Tests with float32
        ht.random.seed(54321)
        a = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2)
        self.assertEqual(a.dtype, ht.float32)
        self.assertEqual(a.larray[0, 0, 0].dtype, torch.float32)
        mean = ht.mean(a)
        #        median = np.median(a)
        std = ht.std(a)
        self.assertTrue(-0.01 < mean < 0.01)
        # self.assertTrue(-0.01 < median < 0.01)
        self.assertTrue(0.99 < std < 1.01)

        ht.random.set_state(("Threefry", 54321, 0x10000000000000000))
        b = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2)
        self.assertTrue(ht.allclose(a, b))

        c = ht.random.randn(30, 30, 30, dtype=ht.float32, split=2)
        self.assertFalse(ht.allclose(a, c))
        self.assertFalse(ht.allclose(b, c))

    def test_randperm(self):
        ht.random.set_state(("Threefry", 0, 0))
        ht.random.seed()
        if self.device.torch_device == "cpu":
            state = torch.random.get_rng_state()
        else:
            state = torch.cuda.get_rng_state(self.device.torch_device)

        # results
        a = ht.random.randperm(10, dtype=ht.int32)
        b = ht.random.randperm(4, dtype=ht.float32, split=0)
        c = ht.random.randperm(5, split=0)
        d = ht.random.randperm(5, dtype=ht.float64)

        if self.device.torch_device == "cpu":
            torch.random.set_rng_state(state)
        else:
            torch.cuda.set_rng_state(state, self.device.torch_device)

        # torch results to compare to
        a_cmp = torch.randperm(10, dtype=torch.int32, device=self.device.torch_device)
        b_cmp = torch.randperm(4, dtype=torch.float32, device=self.device.torch_device)
        c_cmp = torch.randperm(5, dtype=torch.int64, device=self.device.torch_device)
        d_cmp = torch.randperm(5, dtype=torch.float64, device=self.device.torch_device)

        self.assertEqual(a.dtype, ht.int32)
        self.assertTrue((a.larray == a_cmp).all())
        self.assertEqual(b.dtype, ht.float32)
        self.assertTrue((ht.resplit(b).larray == b_cmp).all())
        self.assertEqual(c.dtype, ht.int64)
        self.assertTrue((ht.resplit(c).larray == c_cmp).all())
        self.assertEqual(d.dtype, ht.float64)
        self.assertTrue((d.larray == d_cmp).all())

        with self.assertRaises(TypeError):
            ht.random.randperm("abc")

    def test_standard_normal(self):
        ht.random.set_state(("Threefry", 0, 0))
        ht.random.seed()
        # empty input
        stdn = ht.random.standard_normal()
        self.assertEqual(stdn.dtype, ht.float32)
        self.assertEqual(stdn.shape, (1,))

        # simple test
        shape = (3, 4, 6)
        ht.random.seed(11235)
        stdn = ht.random.standard_normal(shape, split=2)
        ht.random.seed(11235)
        rndn = ht.random.randn(*shape, split=2)

        self.assertEqual(stdn.shape, rndn.shape)
        self.assertEqual(stdn.dtype, rndn.dtype)
        self.assertEqual(stdn.device, rndn.device)
        self.assertTrue(ht.equal(stdn, rndn))
