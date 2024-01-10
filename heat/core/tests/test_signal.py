import numpy as np
import torch
import unittest
import heat as ht
from heat import manipulations

from .test_suites.basic_test import TestCase


class TestSignal(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestSignal, cls).setUpClass()

    def test_convolve(self):
        full_odd = ht.array(
            [0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 29, 15]
        ).astype(ht.int)
        full_even = ht.array(
            [0, 1, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 42, 29, 15]
        ).astype(ht.int)

        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        signal = ht.arange(0, 16).astype(ht.int)
        full_ones = ht.ones(7, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = [1, 1, 1, 1]
        dis_kernel_odd = ht.ones(3, split=0).astype(ht.int)
        dis_kernel_even = ht.ones(4, split=0).astype(ht.int)

        with self.assertRaises(TypeError):
            signal_wrong_type = [0, 1, 2, "tre", 4, "five", 6, "ʻehiku", 8, 9, 10]
            ht.convolve(signal_wrong_type, kernel_odd, mode="full")
        with self.assertRaises(TypeError):
            filter_wrong_type = [1, 1, "pizza", "pineapple"]
            ht.convolve(dis_signal, filter_wrong_type, mode="full")
        with self.assertRaises(ValueError):
            ht.convolve(dis_signal, kernel_odd, mode="invalid")
        with self.assertRaises(ValueError):
            s = dis_signal.reshape((2, -1))
            ht.convolve(s, kernel_odd)
        with self.assertRaises(ValueError):
            k = ht.eye(3)
            ht.convolve(dis_signal, k)
        with self.assertRaises(ValueError):
            ht.convolve(dis_signal, kernel_even, mode="same")
        if self.comm.size > 1:
            with self.assertRaises(ValueError):
                ht.convolve(full_ones, kernel_even, mode="valid")
            with self.assertRaises(ValueError):
                ht.convolve(kernel_even, full_ones, mode="valid")
        if self.comm.size > 5:
            with self.assertRaises(ValueError):
                ht.convolve(dis_signal, kernel_even)

        # test modes, avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "same", "valid"]
            for i, mode in enumerate(modes):
                # odd kernel size
                conv = ht.convolve(dis_signal, kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))

                conv = ht.convolve(dis_signal, dis_kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))

                conv = ht.convolve(signal, dis_kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))

                # different data types
                conv = ht.convolve(dis_signal.astype(ht.float), kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve(dis_signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve(signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                # even kernel size
                # skip mode 'same' for even kernels
                if mode != "same":
                    conv = ht.convolve(dis_signal, kernel_even, mode=mode)
                    dis_conv = ht.convolve(dis_signal, dis_kernel_even, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)
                    dis_gathered = manipulations.resplit(dis_conv, axis=None)

                    if mode == "full":
                        self.assertTrue(ht.equal(full_even, gathered))
                        self.assertTrue(ht.equal(full_even, dis_gathered))
                    else:
                        self.assertTrue(ht.equal(full_even[3:-3], gathered))
                        self.assertTrue(ht.equal(full_even[3:-3], dis_gathered))

                # distributed large signal and kernel
                np.random.seed(12)
                np_a = np.random.randint(1000, size=4418)
                np_b = np.random.randint(1000, size=1543)
                np_conv = np.convolve(np_a, np_b, mode=mode)

                a = ht.array(np_a, split=0, dtype=ht.int32)
                b = ht.array(np_b, split=0, dtype=ht.int32)
                conv = ht.convolve(a, b, mode=mode)
                self.assert_array_equal(conv, np_conv)

        # test edge cases
        # non-distributed signal, size-1 kernel
        signal = ht.arange(0, 16).astype(ht.int)
        alt_signal = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        kernel = ht.ones(1).astype(ht.int)
        conv = ht.convolve(alt_signal, kernel)
        self.assertTrue(ht.equal(signal, conv))

        conv = ht.convolve(1, 5)
        self.assertTrue(ht.equal(ht.array([5]), conv))

    def test_fftconvolve(self):
        axes_1 = ["", None, 0, [0], -1, [-1]]
        axes_2 = [1, [1], -1, [-1]]

        # test real
        a = ht.array([1, 2, 3])
        expected = ht.array([1, 4, 10, 12, 9.0])

        for axis in axes_1:
            if axis == "":
                out = ht.fftconvolve(a, a)
            else:
                out = ht.fftconvolve(a, a, axes=axis)
            self.assertTrue(ht.allclose(out, expected))

        # test real axes
        a = ht.tile(a, [2, 1])
        expected = ht.tile(expected, [2, 1])

        for axis in axes_2:
            out = ht.fftconvolve(a, a, axes=axis)
            self.assertTrue(ht.allclose(out, expected))

    """
    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_complex(self, axes):
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_complex_axes(self, axes):
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        a = np.tile(a, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['',
                                      None,
                                      [0, 1],
                                      [1, 0],
                                      [0, -1],
                                      [-1, 0],
                                      [-2, 1],
                                      [1, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_real_same(self, axes):
        a = array([[1, 2, 3],
                   [4, 5, 6]])
        expected = array([[1, 4, 10, 12, 9],
                          [8, 26, 56, 54, 36],
                          [16, 40, 73, 60, 36]])

        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[1, 2],
                                      [2, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-2, 2],
                                      [2, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_real_same_axes(self, axes):
        a = array([[1, 2, 3],
                   [4, 5, 6]])
        expected = array([[1, 4, 10, 12, 9],
                          [8, 26, 56, 54, 36],
                          [16, 40, 73, 60, 36]])

        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['',
                                      None,
                                      [0, 1],
                                      [1, 0],
                                      [0, -1],
                                      [-1, 0],
                                      [-2, 1],
                                      [1, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_complex_same(self, axes):
        a = array([[1 + 2j, 3 + 4j, 5 + 6j],
                   [2 + 1j, 4 + 3j, 6 + 5j]])
        expected = array([
            [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
            [10j, 44j, 118j, 156j, 122j],
            [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]
            ])

        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[1, 2],
                                      [2, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-2, 2],
                                      [2, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_complex_same_axes(self, axes):
        a = array([[1 + 2j, 3 + 4j, 5 + 6j],
                   [2 + 1j, 4 + 3j, 6 + 5j]])
        expected = array([
            [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
            [10j, 44j, 118j, 156j, 122j],
            [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]
            ])

        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_real_same_mode(self, axes):
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = array([35., 41., 47.])
        expected_2 = array([9., 20., 25., 35., 41., 47., 39., 28., 2.])

        if axes == '':
            out = fftconvolve(a, b, 'same')
        else:
            out = fftconvolve(a, b, 'same', axes=axes)
        assert_array_almost_equal(out, expected_1)

        if axes == '':
            out = fftconvolve(b, a, 'same')
        else:
            out = fftconvolve(b, a, 'same', axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize('axes', [1, -1, [1], [-1]])
    def test_real_same_mode_axes(self, axes):
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = array([35., 41., 47.])
        expected_2 = array([9., 20., 25., 35., 41., 47., 39., 28., 2.])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected_1 = np.tile(expected_1, [2, 1])
        expected_2 = np.tile(expected_2, [2, 1])

        out = fftconvolve(a, b, 'same', axes=axes)
        assert_array_almost_equal(out, expected_1)

        out = fftconvolve(b, a, 'same', axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_real(self, axes):
        # See gh-5897
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1]])
    def test_valid_mode_real_axes(self, axes):
        # See gh-5897
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_complex(self, axes):
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = array([45. + 12.j, 30. + 23.j, 48 + 32.j])

        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_valid_mode_complex_axes(self, axes):
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = array([45. + 12.j, 30. + 23.j, 48 + 32.j])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    def test_valid_mode_ignore_nonaxes(self):
        # See gh-5897
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [1, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'valid', axes=1)
        assert_array_almost_equal(out, expected)

    def test_empty(self):
        # Regression test for #1745: crashes with 0-length input.
        assert_(fftconvolve([], []).size == 0)
        assert_(fftconvolve([5, 6], []).size == 0)
        assert_(fftconvolve([], [7]).size == 0)

    def test_zero_rank(self):
        a = array(4967)
        b = array(3920)
        out = fftconvolve(a, b)
        assert_equal(out, a * b)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        out = fftconvolve(a, b)
        assert_equal(out, a * b)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_random_data(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, 'full')

        if axes == '':
            out = fftconvolve(a, b, 'full')
        else:
            out = fftconvolve(a, b, 'full', axes=axes)
        assert_(np.allclose(out, expected, rtol=1e-10))

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_random_data_axes(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, 'full')

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'full', axes=axes)
        assert_(np.allclose(out, expected, rtol=1e-10))

    @pytest.mark.parametrize('axes', [[1, 4],
                                      [4, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-4, 4],
                                      [4, -4],
                                      [-4, -1],
                                      [-1, -4]])
    def test_random_data_multidim_axes(self, axes):
        a_shape, b_shape = (123, 22), (132, 11)
        np.random.seed(1234)
        a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
        b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
        expected = convolve2d(a, b, 'full')

        a = a[:, :, None, None, None]
        b = b[:, :, None, None, None]
        expected = expected[:, :, None, None, None]

        a = np.moveaxis(a.swapaxes(0, 2), 1, 4)
        b = np.moveaxis(b.swapaxes(0, 2), 1, 4)
        expected = np.moveaxis(expected.swapaxes(0, 2), 1, 4)

        # use 1 for dimension 2 in a and 3 in b to test broadcasting
        a = np.tile(a, [2, 1, 3, 1, 1])
        b = np.tile(b, [2, 1, 1, 4, 1])
        expected = np.tile(expected, [2, 1, 3, 4, 1])

        out = fftconvolve(a, b, 'full', axes=axes)
        assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'n',
        list(range(1, 100)) +
        list(range(1000, 1500)) +
        np.random.RandomState(1234).randint(1001, 10000, 5).tolist())
    def test_many_sizes(self, n):
        a = np.random.rand(n) + 1j * np.random.rand(n)
        b = np.random.rand(n) + 1j * np.random.rand(n)
        expected = np.convolve(a, b, 'full')

        out = fftconvolve(a, b, 'full')
        assert_allclose(out, expected, atol=1e-10)

        out = fftconvolve(a, b, 'full', axes=[0])
        assert_allclose(out, expected, atol=1e-10)

    def test_fft_nan(self):
        n = 1000
        rng = np.random.default_rng(43876432987)
        sig_nan = rng.standard_normal(n)

        for val in [np.nan, np.inf]:
            sig_nan[100] = val
            coeffs = signal.firwin(200, 0.2)

            with pytest.warns(RuntimeWarning, match="Use of fft convolution"):
                signal.convolve(sig_nan, coeffs, mode='same', method='fft')
    """
