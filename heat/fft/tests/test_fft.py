import numpy as np
import torch

import heat as ht
from heat.core.tests.test_suites.basic_test import TestCase


class TestFFT(TestCase):
    def test_fft_ifft(self):
        # 1D non-distributed
        x = ht.random.randn(6, dtype=ht.float64)
        y = ht.fft.fft(x)
        np_y = np.fft.fft(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assert_array_equal(y, np_y)
        backwards = ht.fft.ifft(y)
        self.assertTrue(ht.allclose(backwards, x))

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

        # on GPU, test only on less than 4 processes
        if x.device == ht.cpu or x.comm.size < 4:
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
        # axis out of range
        x = ht.random.randn(6, 3, 3)
        with self.assertRaises(IndexError):
            ht.fft.fft(x, axis=3)
        # n-D axes
        with self.assertRaises(TypeError):
            ht.fft.fft(x, axis=(0, 1))

    def test_fft2_ifft2(self):
        # 2D FFT along non-split axes
        x = ht.random.randn(10, 6, 6, split=0, dtype=ht.float64)
        y = ht.fft.fft2(x)
        np_y = np.fft.fft2(x.numpy())
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)
        backwards = ht.fft.ifft2(y)
        self.assertTrue(ht.allclose(backwards, x))

        # 2D FFT along split axes
        x = ht.random.randn(10, 6, 6, split=0, dtype=ht.float64)
        axes = (0, 1)
        y = ht.fft.fft2(x, axes=axes)
        np_y = np.fft.fft2(x.numpy(), axes=axes)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)
        backwards = ht.fft.ifft2(y, axes=axes)
        self.assertTrue(ht.allclose(backwards, x))

        # exceptions
        x = ht.arange(10, split=0)
        with self.assertRaises(IndexError):
            ht.fft.fft2(x)

    def test_fftn_ifftn(self):
        # 1D non-distributed
        x = ht.random.randn(6)
        y = ht.fft.fftn(x)
        np_y = np.fft.fftn(x.numpy())
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, x.shape)
        self.assert_array_equal(y, np_y)
        backwards = ht.fft.ifftn(y)
        self.assertTrue(ht.allclose(backwards, x))

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

        # on GPU, test only on less than 4 processes
        if x.device == ht.cpu or x.comm.size < 4:
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

    # def test_hfft_ihfft(self):
    #     # follows example in torch.fft.hfft docs
    #     x = ht.zeros((3, 5), split=0, dtype=ht.float64)
    #     edges = [1, 3, 7]
    #     for i, n in enumerate(edges):
    #         x[i] = ht.linspace(0, n, 5)

    #     inv_fft = ht.fft.ifft(x)
    #     # inv_fft is hermitian symmetric along the rows
    #     # we can reconstruct the original signal by transforming the first half of the rows only
    #     reconstructed_x = ht.fft.hfft(inv_fft[:3], n=5)
    #     self.assertTrue(ht.allclose(reconstructed_x, x))
    #     n = 2 * (x.shape[-1] - 1)
    #     reconstructed_x = ht.fft.hfft(inv_fft[:3])
    #     self.assertEqual(reconstructed_x.shape, (3, n))

    # def test_hfftn_ihfftn(self):
    #     # follows example in torch.fft.hfftn docs
    #     x = ht.random.randn(10, 6, 6, dtype=ht.float64)
    #     inv_fft = ht.fft.ifftn(x)
    #     reconstructed_x = ht.fft.hfftn(inv_fft, s=x.shape)
    #     self.assertTrue(ht.allclose(reconstructed_x, x))

    def test_rfft_irfft(self):
        # n-D distributed
        x = ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last axis
        y = ht.fft.fft(x)
        np_y = np.fft.fft(x.numpy())
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)
        backwards = ht.fft.irfft(y, n=x.shape[-1])
        self.assertTrue(ht.allclose(backwards, x))

        # exceptions
        # complex input
        x = x + 1j * ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        with self.assertRaises(TypeError):
            ht.fft.rfft(x)

    def test_rfftn_irfftn(self):
        # n-D distributed
        x = ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last 2 axes
        y = ht.fft.rfftn(x, axes=(1, 2))
        np_y = np.fft.rfftn(x.numpy(), axes=(1, 2))
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        # FFT along all axes
        # TODO: comment this out after merging indexing PR
        # y = ht.fft.rfftn(x)
        # backwards = ht.fft.irfftn(y, s=x.shape)
        # self.assertTrue(ht.allclose(backwards, x))

        # exceptions
        # complex input
        x = x + 1j * ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        with self.assertRaises(TypeError):
            ht.fft.rfftn(x)
