import numpy as np
import unittest
import heat as ht
from heat import manipulations
import scipy.signal as sig
from .test_suites.basic_test import TestCase
import os

from ..signal import conv_input_check, conv_batchprocessing_check, convgenpad


class TestSignal(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestSignal, cls).setUpClass()

    def test_conv_input_check_invalid_types(self):
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = ht.ones(4).astype(ht.int)
        mode = "full"

        # Invalid input
        for conv_dim in [1,2]:
            if conv_dim == 1:
                stride = 1
            else:
                stride = tuple([1] * conv_dim)

            with self.assertRaises(TypeError):
                signal_wrong_type = [[0, 1, 2, "tre", 4, "five", 6, "ʻehiku", 8, 9, 10]]
                if conv_dim == 1:
                    conv_input_check(signal_wrong_type[0], kernel_odd, stride, mode, conv_dim)
                else:
                    conv_input_check(signal_wrong_type, kernel_odd, stride, mode, conv_dim)
            with self.assertRaises(TypeError):
                filter_wrong_type = [[1, 1, "pizza", "pineapple"]]
                if conv_dim == 1:
                    conv_input_check(dis_signal[0], filter_wrong_type, stride, mode, conv_dim)
                else:
                    conv_input_check(dis_signal, filter_wrong_type, stride, mode, conv_dim)
            with self.assertRaises(ValueError):
                if conv_dim == 2:
                    dis_signal = dis_signal.reshape((conv_dim, -1))
                    kernel_even = kernel_even.reshape((conv_dim, -1))
                conv_input_check(dis_signal, kernel_even, stride, "invalid", conv_dim)

    def test_conv_input_check_scaler(self):
        a = 1
        v = 2
        mode="full"

        for conv_dim in [1,2]:
            if conv_dim == 1:
                stride=1
            else:
                stride = tuple([1]*conv_dim)
            a_out, v_out = conv_input_check(a, v, stride, mode, conv_dim)

            if conv_dim == 1:
                target_shape = (1,)
            else:
                target_shape = (1,1)
            assert(a_out.shape == target_shape)
            assert(v_out.shape == target_shape)

    def test_conv_input_check_stride(self):
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = ht.array([1, 1, 1, 1]).astype(ht.int)

        for conv_dim in [1,2]:
            if conv_dim == 2:
                dis_signal = dis_signal.reshape((conv_dim, -1))
                kernel_even = kernel_even.reshape((conv_dim, -1))
                kernel_odd = kernel_odd.reshape((conv_dim, -1))
                stride_0 = (1,0)
                stride_2 = (1,2)
            else:
                stride_0 = 0
                stride_2 = 2

            # stride not positive integer
            with self.assertRaises(ValueError):
                conv_input_check(dis_signal, kernel_even, stride=stride_0, mode="full", convolution_dim=conv_dim)

            # stride > 1 for mode 'same'
            with self.assertRaises(ValueError):
                conv_input_check(dis_signal, kernel_odd, stride=stride_2, mode="same", convolution_dim=conv_dim)

    def test_conv_input_check_even_kernel_mode_same(self):

        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        full_ones = ht.ones(7, split=0).astype(ht.int)
        kernel_even = [1, 1, 1, 1]

        for conv_dim in [1,2]:
            if conv_dim == 1:
                stride=1
            else:
                stride = tuple([1]*conv_dim)
                kernel_even = [kernel_even]
                dis_signal = dis_signal.reshape((conv_dim, -1))

            # Even kernel for mode same
            with self.assertRaises(ValueError):
                conv_input_check(dis_signal, kernel_even, stride, "same", conv_dim)

    def test_conv_input_check_flip_a_v(self):
        mode = "full"
        signal = ht.arange(0, 16, split=0).reshape((2, 8)).astype(ht.int)
        for conv_dim in [1,2]:
            if conv_dim == 1:
                stride = 1
            else:
                stride = tuple([1]*conv_dim)

            # switch, all dimensions larger
            kernel = ht.ones((1, 3)).astype(ht.int)
            signal_out, kernel_out = conv_input_check(kernel, signal, stride, mode, conv_dim)
            self.assertTrue(ht.equal(kernel_out, kernel))
            self.assertTrue(ht.equal(signal_out, signal))

            # switch, mix equal and larger dimensions
            if conv_dim == 2:
                kernel = ht.ones((2, 3))
                signal_out, kernel_out = conv_input_check(kernel, signal, stride, mode, conv_dim)
                self.assertTrue(ht.equal(kernel_out, kernel))
                self.assertTrue(ht.equal(signal_out, signal))

            #no switch
            kernel_signal = ht.ones(16).reshape((2, 8)).astype(ht.int)
            signal_out, kernel_out = conv_input_check(kernel_signal, signal, stride, mode, conv_dim)
            self.assertTrue(ht.equal(kernel_out, signal))
            self.assertTrue(ht.equal(signal_out, kernel_signal))

    def test_conv_input_check_flip_2d_error(self):
        dis_signal = ht.arange(0, 16, split=0).reshape((2,8)).astype(ht.int)
        test_kernels = [ht.ones((1,10)), ht.ones((10,1))]
        stride = (1,1)
        mode = "full"
        conv_dim = 2

        for kernel in test_kernels:
            with self.assertRaises(ValueError):
                conv_input_check(dis_signal, kernel, stride, mode, conv_dim)

    def test_conv_batchprocessing_check_1d_errors(self):
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)

        # kernel dimensions greater than 1 but convolution dims not
        with self.assertRaises(ValueError): #batchprocessing check
            k = ht.eye(3)
            conv_batchprocessing_check(dis_signal, k, 1)

        # kernel has different dimensions than signal
        signal = ht.random.randn(1000, dtype=ht.float32)
        batch_signal = ht.empty((10, 1000), dtype=ht.float32, split=0)
        batch_signal.larray[:] = signal.larray
        batch_kernel_wrong_shape = ht.random.randn(3, 19, dtype=ht.float32)
        with self.assertRaises(ValueError):
            conv_batchprocessing_check(batch_signal, batch_kernel_wrong_shape, 1)

        # signal split dimension
        if dis_signal.comm.size > 1:
            with self.assertRaises(ValueError):
                signal_wrong_split = dis_signal.reshape((2, -1)).resplit(axis=1)
                conv_batchprocessing_check(signal_wrong_split, kernel_odd, 1)

    def test_conv_batchprocessing_check_1d_true(self):
        signal = ht.random.randn(1000, dtype=ht.float32)
        batch_signal = ht.empty((10, 1000), dtype=ht.float32, split=0)
        batch_signal.larray[:] = signal.larray
        kernel = ht.random.randn(19, dtype=ht.float32)
        assert(conv_batchprocessing_check(batch_signal, kernel, 1))

        dis_kernel = ht.array(kernel, dtype=ht.float32, split=0)
        assert(conv_batchprocessing_check(batch_signal, dis_kernel, 1))

        batch_kernel = ht.empty((10, 19), dtype=ht.float32, split=1)
        batch_kernel.larray[:] = dis_kernel.larray
        assert(conv_batchprocessing_check(batch_signal, batch_kernel, 1))

    def test_conv_batchprocessing_check_2d_error(self):
        dis_signal = ht.arange(0, 16, split=0).reshape((2,8)).astype(ht.int)
        kernel_odd = ht.ones((2,3)).astype(ht.int)

        # kernel dimensions greater than 2 but convolution dims not
        with self.assertRaises(ValueError):
            k = ht.eye(3).reshape((1,3,3))
            conv_batchprocessing_check(dis_signal, k, 2)

        # kernel has different dimensions than signal
        signal = ht.random.randn(10, 100, dtype=ht.float32)
        batch_signal = ht.empty((10, 10, 100), dtype=ht.float32, split=0)
        batch_signal.larray[:] = signal.larray
        batch_kernel_wrong_shape = ht.random.randn(3, 3, 19, dtype=ht.float32)
        with self.assertRaises(ValueError):
            conv_batchprocessing_check(batch_signal, batch_kernel_wrong_shape, 2)

        # Batch processed signal split along convolution dimension
        if dis_signal.comm.size > 1:
            with self.assertRaises(ValueError):
                signal_wrong_split = dis_signal.reshape((2, 2, 4)).resplit(axis=1)
                conv_batchprocessing_check(signal_wrong_split, kernel_odd, 2)

    def test_conv_batchprocessing_check_2d_true(self):
        # set batch processing true
        signal = ht.random.randn(1000, dtype=ht.float32)
        batch_signal = ht.empty((10, 1000), dtype=ht.float32, split=0)
        batch_signal.larray[:] = signal.larray
        kernel = ht.random.randn(19, dtype=ht.float32)
        assert(conv_batchprocessing_check(batch_signal, kernel, 2))

        dis_kernel = ht.array(kernel, split=0)
        assert(conv_batchprocessing_check(batch_signal, dis_kernel, 2))

        batch_kernel = ht.empty((10, 19), dtype=ht.float32, split=1)
        batch_kernel.larray[:] = dis_kernel.larray
        assert(conv_batchprocessing_check(batch_signal, batch_kernel, 2))

    def test_only_balanced_kernel(self):
        for conv_dim in (1,2):
            if conv_dim == 1:
                signal = ht.array([0,1,3,4,5,6,7,8,9,10], split=0).astype(ht.float32)
                dis_kernel = ht.array([1, 1], split=0).astype(ht.float32)
                test_target = ht.convolve
            else:
                signal = ht.array([[0, 1, 3, 4, 5],[6, 7, 8, 9, 10]], split=0).astype(ht.float32)
                dis_kernel = ht.array([[1, 1],[1, 1]], split=0).astype(ht.float32)
                test_target = ht.convolve2d

            if self.comm.size > 1:
                target_map = dis_kernel.lshape_map
                target_map[0,0] = 2
                target_map[1:,0] = 0
                dis_kernel.redistribute_(dis_kernel.lshape_map, target_map)
                with self.assertRaises(ValueError):
                    test_target(signal, dis_kernel)

    def test_convolve_local_chunks_error(self):
        full_ones = ht.ones(7, split=0).astype(ht.int)
        kernel_even = [1, 1, 1, 1]
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)

        if self.comm.size > 1:
            with self.assertRaises(ValueError):
                ht.convolve(full_ones, kernel_even, mode="valid")
            with self.assertRaises(ValueError):
                ht.convolve(kernel_even, full_ones, mode="valid")
        if self.comm.size > 5:
            with self.assertRaises(ValueError):
                ht.convolve(dis_signal, kernel_even)

    def test_convolve2d_local_chunks_error(self):
        assert(False)

    def assert_convolution_stride(self, signal, kernel, mode, stride, solution):
        conv = ht.convolve(signal, kernel, mode=mode, stride=stride)
        gathered = manipulations.resplit(conv, axis=None)
        self.assertTrue(ht.equal(solution, gathered))

    def test_convolve_batch_convolutions(self):
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

    def test_convolve_stride_batch_convolutions(self):
        float_dtype = ht.float32 if self.is_mps else ht.float64
        signal = ht.random.randn(1000, dtype=float_dtype)
        kernel = ht.random.randn(19, dtype=float_dtype)

        # distributed input along the first axis
        stride = 123
        batch_signal = ht.empty((10, 1000), dtype=float_dtype, split=0)
        batch_signal.larray[:] = signal.larray

        batch_convolved = ht.convolve(batch_signal, kernel, mode="valid", stride=stride)
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, mode="valid", stride=stride), batch_convolved[0])
        )

        # distributed kernel
        stride = 142
        dis_kernel = ht.array(kernel, split=0)

        batch_convolved = ht.convolve(batch_signal, dis_kernel, stride=stride)
        self.assertTrue(ht.equal(ht.convolve(signal, kernel, stride=stride), batch_convolved[0]))

        # batch kernel
        stride = 41
        batch_kernel = ht.empty((10, 19), dtype=float_dtype, split=1)
        batch_kernel.larray[:] = dis_kernel.larray

        batch_convolved = ht.convolve(batch_signal, kernel, mode="full", stride=stride)
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, mode="full", stride=stride), batch_convolved[0])
        )

        # n-D batch convolution
        stride = 55
        batch_signal = ht.empty((4, 3, 3, 1000), dtype=float_dtype, split=1)
        batch_signal.larray[:, :, :] = signal.larray

        batch_convolved = ht.convolve(batch_signal, kernel, mode="valid", stride=stride)
        self.assertTrue(
            ht.equal(
                ht.convolve(signal, kernel, mode="valid", stride=stride),
                batch_convolved[1, 2, 0],
            )
        )

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
            np_type = np.float32 if self.is_mps else np.int32
            stride = np.random.randint(1, high=len(np_a), size=1)[0]

            for mode in ["full", "valid"]:
                # solution
                np_conv = np.convolve(np_a, np_b, mode=mode)
                solution = np_conv[::stride].astype(np_type)

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
    def test_convolve2d(self):
        test_device = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
        print("DEBUGGING: test_device", test_device, test_device == "gpu")
        if test_device == "gpu":
            # CUDA does not support int 2D convolution
            # if tests are on GPU, set dtype to float
            dtype = ht.float32
        else:
            dtype = ht.int
        dis_signal = ht.arange(256, split=0).reshape((16, 16)).astype(dtype)
        signal = ht.arange(256).reshape((16, 16)).astype(dtype)

        kernel_odd = ht.arange(9).reshape((3, 3)).astype(dtype)
        kernel_even = ht.arange(16).reshape((4, 4)).astype(dtype)

        np_sig = np.arange(256).reshape((16, 16))
        np_k_odd = np.arange(9).reshape((3, 3))
        np_k_even = np.arange(16).reshape((4, 4))

        full_odd = ht.array(sig.convolve2d(np_sig, np_k_odd))
        full_even = ht.array(sig.convolve2d(np_sig, np_k_even))

        dis_kernel_odd = ht.arange(9, split=0).reshape((3, 3)).astype(dtype)
        dis_kernel_even = ht.arange(16, split=0).reshape((4, 4)).astype(dtype)

        with self.assertRaises(TypeError):
            signal_wrong_type = [[0, 1, 2, "tre", 4]] * 5
            ht.convolve2d(signal_wrong_type, kernel_odd)
        with self.assertRaises(TypeError):
            filter_wrong_type = [[1, "pizza", "pineapple"]] * 3
            ht.convolve2d(dis_signal, filter_wrong_type, mode="full")
        with self.assertRaises(ValueError):
            ht.convolve2d(dis_signal, kernel_odd, mode="invalid")
        with self.assertRaises(ValueError):
            s = dis_signal.reshape((2, 2, -1))
            ht.convolve2d(s, kernel_odd)
        with self.assertRaises(ValueError):
            k = ht.arange(3)
            ht.convolve2d(dis_signal, k)
        with self.assertRaises(ValueError):
            ht.convolve2d(dis_signal, kernel_even, mode="same")
        if self.comm.size > 2:
            with self.assertRaises(ValueError):
                ht.convolve2d(dis_signal, signal, mode="valid")

        # test modes, avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "same", "valid"]
            for i, mode in enumerate(modes):
                # odd kernel size
                print(
                    "DEBUGGING: dis_signal.dtype, kernel_odd.dtype",
                    dis_signal.dtype,
                    kernel_odd.dtype,
                )
                conv = ht.convolve2d(dis_signal, kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered)
                )

                conv = ht.convolve2d(dis_signal, dis_kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered)
                )

                conv = ht.convolve2d(signal, dis_kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered)
                )

                # different data types
                conv = ht.convolve2d(dis_signal.astype(ht.float64), kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float64), gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float64), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float64), gathered))

                conv = ht.convolve2d(signal.astype(ht.float64), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float64), gathered))

                # even kernel size
                # skip mode 'same' for even kernels
                if mode != "same":
                    conv = ht.convolve2d(dis_signal, kernel_even, mode=mode)
                    dis_conv = ht.convolve2d(dis_signal, dis_kernel_even, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)
                    dis_gathered = manipulations.resplit(dis_conv, axis=None)

                    if mode == "full":
                        self.assertTrue(ht.equal(full_even, gathered))
                        self.assertTrue(ht.equal(full_even, dis_gathered))
                    else:
                        self.assertTrue(ht.equal(full_even[3:-3, 3:-3], gathered))
                        self.assertTrue(ht.equal(full_even[3:-3, 3:-3], dis_gathered))

                # distributed large signal and kernel
                np.random.seed(12)
                np_a = np.random.randint(1000, size=(140, 250))
                np_b = np.random.randint(1000, size=(39, 17))
                sc_conv = sig.convolve2d(np_a, np_b, mode=mode)

                a = ht.array(np_a, split=0, dtype=dtype)
                b = ht.array(np_b, split=0, dtype=dtype)
                conv = ht.convolve2d(a, b, mode=mode)
                self.assert_array_equal(conv, sc_conv)

                a = ht.array(np_a, split=1, dtype=dtype)
                b = ht.array(np_b, split=1, dtype=dtype)
                conv = ht.convolve2d(a, b, mode=mode)
                self.assert_array_equal(conv, sc_conv)

        # test edge cases
        # non-distributed signal, size-1 kernel
        signal = ht.arange(0, 16).reshape(4, 4).astype(dtype)
        alt_signal = ht.arange(16).reshape(4, 4).astype(dtype)
        # please review and then remove: here we require (for whathever reason): ht.float32 instead of ht.int
        kernel = ht.ones(1).reshape((1, 1)).astype(ht.float32)
        conv = ht.convolve2d(alt_signal, kernel)
        self.assertTrue(ht.equal(signal, conv))
