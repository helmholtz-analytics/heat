import numpy as np
import torch

import heat as ht
from heat.core.tests.test_suites.basic_test import TestCase


class TestFFT(TestCase):
    def test_fft(self):
        # 1D non-distributed
        x = ht.random.randn(6)
        y = ht.fft.fft(x)
        np_y = np.fft.fft(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assert_array_equal(y, np_y)

        # 1D distributed
        x = ht.random.randn(6, split=0)
        n = 8
        y = ht.fft.fft(x, n=n)
        np_y = np.fft.fft(x.numpy(), n=n)
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # n-D distributed
        x = ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last axis
        n = 5
        y = ht.fft.fft(x, n=n)
        np_y = np.fft.fft(x.numpy(), n=n)
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # FFT along distributed axis, n not None
        n = 8
        y = ht.fft.fft(x, axis=0, n=n)
        np_y = np.fft.fft(x.numpy(), axis=0, n=n)
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # complex input
        x = x + 1j * ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last axis (distributed)
        x.resplit_(axis=2)
        y = ht.fft.fft(x, n=n)
        np_y = np.fft.fft(x.numpy(), n=n)
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 2)
        self.assert_array_equal(y, np_y)

        # exceptions
        # wrong input type
        x = np.random.randn(6, 3, 3)
        with self.assertRaises(TypeError):
            ht.fft.fft(x)

    def test_fft2(self):
        # 2D FFT along non-split axes
        x = ht.random.randn(10, 6, 6, split=0)
        y = ht.fft.fft2(x)
        np_y = np.fft.fft2(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # 2D FFT along split axes
        x = ht.random.randn(10, 6, 6, split=0)
        axes = (0, 1)
        y = ht.fft.fft2(x, axes=axes)
        np_y = np.fft.fft2(x.numpy(), axes=axes)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assertTrue(ht.allclose(y, ht.array(np_y, split=y.split)))

        # exceptions
        x = ht.arange(10, split=0)
        with self.assertRaises(IndexError):
            ht.fft.fft2(x)

    def test_ifft(self):
        # 1D non-distributed
        x = ht.random.randn(6, dtype=ht.float64)
        x_fft = ht.fft.fft(x)
        y = ht.fft.ifft(x_fft)
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assert_array_equal(y, x.numpy())

    def test_rfft(self):
        pass

    def test_irfft(self):
        pass

    def test_ifft2(self):
        pass

    def test_rfft2(self):
        pass

    def test_irfft2(self):
        pass

    def test_fftn(self):
        # 1D non-distributed
        x = ht.random.randn(6)
        y = ht.fft.fftn(x)
        np_y = np.fft.fftn(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assert_array_equal(y, np_y)

        # 1D distributed
        x = ht.random.randn(6, split=0)
        y = ht.fft.fftn(x)
        np_y = np.fft.fftn(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # n-D distributed
        x = ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last 2 axes
        y = ht.fft.fftn(x, s=(6, 6))
        np_y = np.fft.fftn(x.numpy(), s=(6, 6))
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # FFT along distributed axis
        x.resplit_(axis=1)
        y = ht.fft.fftn(x, axes=(0, 1), s=(10, 8))
        np_y = np.fft.fftn(x.numpy(), axes=(0, 1), s=(10, 8))
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 1)
        self.assert_array_equal(y, np_y)

        # exceptions
        # wrong input type
        x = torch.randn(6, 3, 3)
        with self.assertRaises(TypeError):
            ht.fft.fftn(x)
        # s larger than dimensions
        x = ht.random.randn(6, 3, 3, split=0)
        with self.assertRaises(ValueError):
            ht.fft.fftn(x, s=(10, 10, 10, 10))

    def test_ifftn(self):
        pass

    def test_rfftn(self):
        pass

    def test_irfftn(self):
        pass
