import numpy as np
import torch
import unittest

import heat as ht
from heat.core.tests.test_suites.basic_test import TestCase

torch_ihfftn = hasattr(torch.fft, "ihfftn")


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
        x = ht.random.randn(3, 6, 6, split=0, dtype=ht.float64)
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

    def test_fftfreq_rfftfreq(self):
        # non-distributed
        n = 10
        d = 0.1
        y = ht.fft.fftfreq(n, d=d)
        np_y = np.fft.fftfreq(n, d=d)
        self.assertEqual(y.shape, np_y.shape)
        self.assert_array_equal(y, np_y)

        # distributed
        y = ht.fft.fftfreq(n, d=d, split=0)
        self.assertEqual(y.shape, np_y.shape)
        self.assert_array_equal(y, np_y)

        # real
        n = 107
        d = 0.22365
        y = ht.fft.rfftfreq(n, d=d)
        np_y = np.fft.rfftfreq(n, d=d)
        self.assertEqual(y.shape, np_y.shape)
        self.assert_array_equal(y, np_y)

        # exceptions
        # wrong input type
        n = 10
        d = 0.1
        with self.assertRaises(TypeError):
            ht.fft.fftfreq(n, d=d, dtype=ht.int32)
        # unsupported n
        n = 10.7
        with self.assertRaises(ValueError):
            ht.fft.fftfreq(n, d=d)
        # unsupported d
        # torch does not support complex d
        n = 10
        d = 0.1 + 1j
        with self.assertRaises(NotImplementedError):
            ht.fft.fftfreq(n, d=d)
        d = ht.array(0.1)
        with self.assertRaises(TypeError):
            ht.fft.fftfreq(n, d=d)

    def test_fftshift_ifftshift(self):
        # non-distributed
        x = ht.fft.fftfreq(10)
        y = ht.fft.fftshift(x)
        np_y = np.fft.fftshift(x.numpy())
        self.assertEqual(y.shape, np_y.shape)
        self.assert_array_equal(y, np_y)
        backwards = ht.fft.ifftshift(y)
        self.assertTrue(ht.allclose(backwards, x))

        # distributed
        # (following fftshift example from torch.fft)
        x = ht.fft.fftfreq(5, d=1 / 5, split=0) + 0.1 * ht.fft.fftfreq(5, d=1 / 5, split=0).reshape(
            5, 1
        )
        y = ht.fft.fftshift(x, axes=(0, 1))
        np_y = np.fft.fftshift(x.numpy(), axes=(0, 1))
        self.assert_array_equal(y, np_y)

        # exceptions
        # wrong axis
        with self.assertRaises(IndexError):
            ht.fft.fftshift(x, axes=(0, 2))

    def test_hfft_ihfft(self):
        x = ht.zeros((3, 5), split=0, dtype=ht.float64)
        edges = [1, 3, 7]
        for i, n in enumerate(edges):
            x[i] = ht.linspace(0, n, 5)
        inv_fft = ht.fft.ihfft(x)

        # inv_fft is hermitian-symmetric along the rows
        reconstructed_x = ht.fft.hfft(inv_fft, n=5)
        self.assertTrue(ht.allclose(reconstructed_x, x))
        n = 2 * (x.shape[-1] - 1)
        reconstructed_x = ht.fft.hfft(inv_fft, n=n)
        self.assertEqual(reconstructed_x.shape[-1], n)

    @unittest.skipIf(not torch_ihfftn, "no torch.fft.ihfftn in torch < 1.11")
    def test_hfft2_ihfft2(self):
        x = ht.random.randn(10, 6, 6, dtype=ht.float64)
        inv_fft = ht.fft.ihfft2(x)
        reconstructed_x = ht.fft.hfft2(inv_fft, s=x.shape[-2:])
        self.assertTrue(ht.allclose(reconstructed_x, x))

    @unittest.skipIf(not torch_ihfftn, "no torch.fft.ihfftn in torch < 1.11")
    def test_hfftn_ihfftn(self):
        x = ht.random.randn(10, 6, 6, dtype=ht.float64)
        inv_fft = ht.fft.ifftn(x)
        reconstructed_x = ht.fft.hfftn(inv_fft, s=x.shape)
        self.assertTrue(ht.allclose(reconstructed_x, x))
        reconstructed_x_no_s = ht.fft.hfftn(inv_fft)
        self.assertEqual(reconstructed_x_no_s.shape[-1], 2 * (inv_fft.shape[-1] - 1))

    def test_rfft_irfft(self):
        # n-D distributed
        x = ht.random.randn(10, 8, 3, dtype=ht.float64, split=0)
        # FFT along last axis
        y = ht.fft.rfft(x)
        np_y = np.fft.rfft(x.numpy())
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)
        backwards = ht.fft.irfft(y, n=x.shape[-1])
        self.assertTrue(ht.allclose(backwards, x))
        backwards_no_n = ht.fft.irfft(y)
        self.assertEqual(backwards_no_n.shape[-1], 2 * (y.shape[-1] - 1))

        # exceptions
        # complex input
        x = x + 1j * ht.random.randn(10, 8, 3, dtype=ht.float64, split=0)
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
        backwards = ht.fft.irfftn(y, s=x.shape[-2:])
        self.assertTrue(ht.allclose(backwards, x))
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

    def test_rfft2_irfft2(self):
        # n-D distributed
        x = ht.random.randn(10, 8, 6, dtype=ht.float64, split=0)
        # FFT along last 2 axes
        y = ht.fft.rfft2(x, axes=(1, 2))
        np_y = np.fft.rfft2(x.numpy(), axes=(1, 2))
        self.assertIsInstance(y, ht.DNDarray)
        self.assertEqual(y.shape, np_y.shape)
        self.assertTrue(y.split == 0)
        self.assert_array_equal(y, np_y)

        backwards = ht.fft.irfft2(y, s=x.shape[-2:])
        self.assertTrue(ht.allclose(backwards, x))
