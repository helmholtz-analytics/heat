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

        # n-D distributed
        x = ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last axis
        y = ht.fft.fft(x)
        np_y = np.fft.fft(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # FFT along distributed axis
        y = ht.fft.fft(x, axis=0)
        np_y = np.fft.fft(x.numpy(), axis=0)
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # complex input
        x = x + 1j * ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last axis (distributed)
        x.resplit_(axis=2)
        y = ht.fft.fft(x)
        np_y = np.fft.fft(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(y.split == 2)
        self.assert_array_equal(y, np_y)

    def test_ifft(self):
        # 1D non-distributed
        x = ht.random.randn(6)
        x_fft = ht.fft.fft(x)
        y = ht.fft.ifft(x_fft)
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assert_array_equal(y, x.numpy())

    def test_rfft(self):
        pass

    def test_irfft(self):
        pass

    def test_fft2(self):
        pass

    def test_ifft2(self):
        pass

    def test_rfft2(self):
        pass

    def test_irfft2(self):
        pass

    def test_fftn(self):
        pass

    def test_ifftn(self):
        pass

    def test_rfftn(self):
        pass

    def test_irfftn(self):
        pass
