import numpy as np
import torch
import unittest
import pytest
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
            ht.convolve(signal_wrong_type, kernel_odd, mode="full", stride=1)
        with self.assertRaises(TypeError):
            filter_wrong_type = [1, 1, "pizza", "pineapple"]
            ht.convolve(dis_signal, filter_wrong_type, mode="full", stride=1)
        with self.assertRaises(ValueError):
            ht.convolve(dis_signal, kernel_odd, mode="invalid", stride=1)
        if dis_signal.comm.size > 1:
            with self.assertRaises(ValueError):
                s = dis_signal.reshape((2, -1)).resplit(axis=1)
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
                if not self.is_mps:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve(dis_signal, kernel_odd, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))

                    conv = ht.convolve(dis_signal, dis_kernel_odd, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))

                    conv = ht.convolve(signal, dis_kernel_odd, mode=mode).astype(ht.float)
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
                    # int tests not on MPS
                    if not self.is_mps:
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
                    else:
                        # float tests
                        conv = ht.convolve(dis_signal.astype(ht.float), kernel_even, mode=mode)
                        dis_conv = ht.convolve(
                            dis_signal.astype(ht.float), dis_kernel_even.astype(ht.float), mode=mode
                        )
                        gathered = manipulations.resplit(conv, axis=None)
                        dis_gathered = manipulations.resplit(dis_conv, axis=None)

                        if mode == "full":
                            self.assertTrue(ht.equal(full_even.astype(ht.float), gathered))
                            self.assertTrue(ht.equal(full_even.astype(ht.float), dis_gathered))
                        else:
                            self.assertTrue(ht.equal(full_even[3:-3].astype(ht.float), gathered))
                            self.assertTrue(
                                ht.equal(full_even[3:-3].astype(ht.float), dis_gathered)
                            )

                # distributed large signal and kernel
                np.random.seed(12)
                np_a = np.random.randint(1000, size=4418)
                np_b = np.random.randint(1000, size=1543)
                np_conv = np.convolve(np_a, np_b, mode=mode)

                if self.is_mps:
                    # torch convolution only supports float on MPS
                    a = ht.array(np_a, split=0, dtype=ht.float32)
                    b = ht.array(np_b, split=0, dtype=ht.float32)
                    conv = ht.convolve(a, b, mode=mode)
                    self.assert_array_equal(conv, np_conv.astype(np.float32))
                else:
                    a = ht.array(np_a, split=0, dtype=ht.int32)
                    b = ht.array(np_b, split=0, dtype=ht.int32)
                    conv = ht.convolve(a, b, mode=mode)
                    self.assert_array_equal(conv, np_conv)

        # test edge cases
        # non-distributed signal, size-1 kernel
        if self.is_mps:
            # torch convolution only supports float on MPS
            signal = ht.arange(0, 16, dtype=ht.float32)
            alt_signal = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            kernel = ht.ones(1, dtype=ht.float32)
            conv = ht.convolve(alt_signal, kernel)
        else:
            signal = ht.arange(0, 16).astype(ht.int)
            alt_signal = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            kernel = ht.ones(1).astype(ht.int)
            conv = ht.convolve(alt_signal, kernel)
        self.assertTrue(ht.equal(signal, conv))

        if not self.is_mps:
            conv = ht.convolve(1, 5)
            self.assertTrue(ht.equal(ht.array([5]), conv))

        # test batched convolutions
        float_dtype = ht.float32 if self.is_mps else ht.float64
        # distributed along the first axis
        signal = ht.random.randn(1000, dtype=float_dtype)
        batch_signal = ht.empty((10, 1000), dtype=float_dtype, split=0)
        batch_signal.larray[:] = signal.larray
        kernel = ht.random.randn(19, dtype=float_dtype)
        batch_convolved = ht.convolve(batch_signal, kernel, mode="same")
        self.assertTrue(ht.equal(ht.convolve(signal, kernel, mode="same"), batch_convolved[0]))

        # distributed kernel
        dis_kernel = ht.array(kernel, split=0)
        batch_convolved = ht.convolve(batch_signal, dis_kernel)
        self.assertTrue(ht.equal(ht.convolve(signal, kernel), batch_convolved[0]))
        batch_kernel = ht.empty((10, 19), dtype=float_dtype, split=1)
        batch_kernel.larray[:] = dis_kernel.larray
        batch_convolved = ht.convolve(batch_signal, batch_kernel, mode="full")
        self.assertTrue(ht.equal(ht.convolve(signal, kernel, mode="full"), batch_convolved[0]))

        # n-D batch convolution
        batch_signal = ht.empty((4, 3, 3, 1000), dtype=float_dtype, split=1)
        batch_signal.larray[:, :, :] = signal.larray
        batch_convolved = ht.convolve(batch_signal, kernel, mode="valid")
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, mode="valid"), batch_convolved[1, 2, 0])
        )

        # test batch-convolve exceptions
        batch_kernel_wrong_shape = ht.random.randn(3, 19, dtype=float_dtype)
        with self.assertRaises(ValueError):
            ht.convolve(batch_signal, batch_kernel_wrong_shape)
        if kernel.comm.size > 1:
            batch_signal_wrong_split = batch_signal.resplit(-1)
            with self.assertRaises(ValueError):
                ht.convolve(batch_signal_wrong_split, kernel)

    def test_convolve_stride_errors(self):
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = [1, 1, 1, 1]

        # stride not positive integer
        with self.assertRaises(ValueError):
            ht.convolve(dis_signal, kernel_even, mode="full", stride=0)

        # stride > 1 for mode 'same'
        with self.assertRaises(ValueError):
            ht.convolve(dis_signal, kernel_odd, mode="same", stride=2)

    def test_convolve_stride_batch_convolutions(self):
        float_dtype = ht.float32 if self.is_mps else ht.float64
        signal = ht.random.randn(1000, dtype=float_dtype)
        kernel = ht.random.randn(19, dtype=float_dtype)

        # distributed input along the first axis
        stride = np.random.randint(1, high=1000, size=1)[0]
        batch_signal = ht.empty((10, 1000), dtype=float_dtype, split=0)
        batch_signal.larray[:] = signal.larray

        batch_convolved = ht.convolve(batch_signal, kernel, mode="valid", stride=stride)
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, mode="valid", stride=stride), batch_convolved[0])
        )

        # distributed kernel
        stride = np.random.randint(1, high=200, size=1)[0]
        dis_kernel = ht.array(kernel, split=0)

        batch_convolved = ht.convolve(batch_signal, dis_kernel, stride=stride)
        self.assertTrue(ht.equal(ht.convolve(signal, kernel, stride=stride), batch_convolved[0]))

        # batch kernel
        stride = np.random.randint(1, high=200, size=1)[0]
        batch_kernel = ht.empty((10, 19), dtype=float_dtype, split=1)
        batch_kernel.larray[:] = dis_kernel.larray

        batch_convolved = ht.convolve(batch_signal, kernel, mode="full", stride=stride)
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, mode="full", stride=stride), batch_convolved[0])
        )

        # n-D batch convolution
        stride = np.random.randint(1, high=200, size=1)[0]
        batch_signal = ht.empty((4, 3, 3, 1000), dtype=float_dtype, split=1)
        batch_signal.larray[:, :, :] = signal.larray

        batch_convolved = ht.convolve(batch_signal, kernel, mode="valid", stride=stride)
        self.assertTrue(
            ht.equal(
                ht.convolve(signal, kernel, mode="valid", stride=stride),
                batch_convolved[1, 2, 0],
            )
        )

    def assert_convolution_stride(self, signal, kernel, mode, stride, solution):
        conv = ht.convolve(signal, kernel, mode=mode, stride=stride)
        gathered = manipulations.resplit(conv, axis=None)
        self.assertTrue(ht.equal(solution, gathered))

    def test_convolve_stride_kernel_odd_mode_full(self):

        ht_dtype = ht.int

        mode = "full"
        stride = 2
        solution = ht.array([0, 3, 9, 15, 21, 27, 33, 39, 29]).astype(ht_dtype)

        dis_signal = ht.arange(0, 16, split=0).astype(ht_dtype)
        signal = ht.arange(0, 16).astype(ht_dtype)
        kernel = ht.ones(3).astype(ht_dtype)
        dis_kernel = ht.ones(3, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        if self.comm.size <= 3:

            if not self.is_mps:
                # torch convolution does not support int on MPS
                self.assert_convolution_stride(dis_signal, kernel, mode, stride, solution)
                self.assert_convolution_stride(signal, dis_kernel, mode, stride, solution)
                self.assert_convolution_stride(dis_signal, dis_kernel, mode, stride, solution)

            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, solution
            )

    def test_convolve_stride_kernel_odd_mode_valid(self):

        ht_dtype = ht.int

        mode = "valid"
        stride = 2
        solution = ht.array([3, 9, 15, 21, 27, 33, 39]).astype(ht_dtype)

        dis_signal = ht.arange(0, 16, split=0).astype(ht_dtype)
        signal = ht.arange(0, 16).astype(ht_dtype)
        kernel = ht.ones(3).astype(ht_dtype)
        dis_kernel = ht.ones(3, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        if self.comm.size <= 3:

            if not self.is_mps:
                # torch convolution does not support int on MPS
                self.assert_convolution_stride(dis_signal, kernel, mode, stride, solution)
                self.assert_convolution_stride(signal, dis_kernel, mode, stride, solution)
                self.assert_convolution_stride(dis_signal, dis_kernel, mode, stride, solution)

            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, solution
            )

    def test_convolve_stride_kernel_even_mode_full(self):

        ht_dtype = ht.int

        mode = "full"
        stride = 2
        solution = ht.array([0, 3, 10, 18, 26, 34, 42, 50, 42, 15]).astype(ht_dtype)

        dis_signal = ht.arange(0, 16, split=0).astype(ht_dtype)
        signal = ht.arange(0, 16).astype(ht_dtype)
        kernel = [1, 1, 1, 1]
        dis_kernel = ht.ones(4, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        if self.comm.size <= 3:

            if not self.is_mps:
                # torch convolution does not support int on MPS
                self.assert_convolution_stride(dis_signal, kernel, mode, stride, solution)
                self.assert_convolution_stride(signal, dis_kernel, mode, stride, solution)
                self.assert_convolution_stride(dis_signal, dis_kernel, mode, stride, solution)

            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, solution
            )

    def test_convolve_stride_kernel_even_mode_valid(self):

        ht_dtype = ht.int

        mode = "valid"
        stride = 2
        solution = ht.array([6, 14, 22, 30, 38, 46, 54]).astype(ht_dtype)

        dis_signal = ht.arange(0, 16, split=0).astype(ht_dtype)
        signal = ht.arange(0, 16).astype(ht_dtype)
        kernel = [1, 1, 1, 1]
        dis_kernel = ht.ones(4, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        if self.comm.size <= 3:

            if not self.is_mps:
                # torch convolution does not support int on MPS
                self.assert_convolution_stride(dis_signal, kernel, mode, stride, solution)
                self.assert_convolution_stride(signal, dis_kernel, mode, stride, solution)
                self.assert_convolution_stride(dis_signal, dis_kernel, mode, stride, solution)

            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, solution
            )
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, solution
            )

    def test_convolution_stride_large_signal_and_kernel_modes(self):
        if self.comm.size <= 3:
            # prep
            np.random.seed(12)
            np_a = np.random.randint(1000, size=4418)
            np_b = np.random.randint(1000, size=154)
            # torch convolution does not support int on MPS
            ht_dtype = ht.float32 if self.is_mps else ht.int32
            stride = np.random.randint(1, high=len(np_a), size=1)[0]
            # generate solution
            t_a = torch.asarray(np_a, dtype=torch.float32).reshape([1, 1, len(np_a)])
            t_b = torch.asarray(np_b, dtype=torch.float32).reshape([1, 1, len(np_b)])
            t_b = torch.flip(t_b, [2])
            for mode in ["full", "valid"]:
                if mode == "full":
                    solution = torch.conv1d(t_a, t_b, stride=stride, padding=len(np_b) - 1)
                else:
                    solution = torch.conv1d(t_a, t_b, stride=stride, padding=0)

                solution = torch.squeeze(solution).numpy()
                if solution.shape == ():
                    solution = solution.reshape((1,))
                    # test
                a = ht.array(np_a, split=0, dtype=ht_dtype)
                b = ht.array(np_b, split=None, dtype=ht_dtype)
                conv = ht.convolve(a, b, mode=mode, stride=stride)
                self.assert_array_equal(conv, solution)

                b = ht.array(np_b, split=0, dtype=ht_dtype)
                conv = ht.convolve(a, b, mode=mode, stride=stride)
                self.assert_array_equal(conv, solution)

    def test_convolution_stride_kernel_size_1(self):

        # prep
        ht_dtype = ht.float32 if self.is_mps else ht.int32

        # non-distributed signal
        signal = ht.arange(0, 16, dtype=ht_dtype)
        alt_signal = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        kernel = ht.ones(1, dtype=ht_dtype)
        conv = ht.convolve(alt_signal, kernel, stride=2)
        self.assertTrue(ht.equal(signal[0::2], conv))

        if not self.is_mps:
            for s in [2, 3, 4]:
                conv = ht.convolve(1, 5, stride=s)
                self.assertTrue(ht.equal(ht.array([5]), conv))
