import numpy as np
import torch
import heat as ht
from heat import manipulations
import scipy.signal as sig
from .test_suites.basic_test import TestCase
import os

from ..signal import conv_input_check, conv_batchprocessing_check

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
        for conv_dim in [1, 2]:
            if conv_dim == 1:
                stride = 1
            else:
                stride = tuple([1] * conv_dim)

            with self.assertRaises(TypeError):
                signal_wrong_type = [[0, 1, 2, "tre", 4, "five", 6, "ʻehiku", 8, 9, 10]]
                if conv_dim == 1:
                    conv_input_check(
                        signal_wrong_type[0], kernel_odd, stride, mode, conv_dim
                    )
                else:
                    conv_input_check(
                        signal_wrong_type, kernel_odd, stride, mode, conv_dim
                    )
            with self.assertRaises(TypeError):
                filter_wrong_type = [[1, 1, "pizza", "pineapple"]]
                if conv_dim == 1:
                    conv_input_check(
                        dis_signal[0], filter_wrong_type, stride, mode, conv_dim
                    )
                else:
                    conv_input_check(
                        dis_signal, filter_wrong_type, stride, mode, conv_dim
                    )
            with self.assertRaises(ValueError):
                if conv_dim == 2:
                    dis_signal = dis_signal.reshape((conv_dim, -1)).astype(ht.float)
                    kernel_even = kernel_even.reshape((conv_dim, -1)).astype(ht.float)
                conv_input_check(dis_signal, kernel_even, stride, "invalid", conv_dim)

    def test_conv_input_check_detailed_dtype(self):
        signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel = ht.ones(3)

        mode = "full"
        def get_signal_kernel(dtype, conv_dim):
            if conv_dim == 2:
                signal = ht.arange(0, 16, split=0).astype(dtype).reshape(4,4)
                kernel = ht.array([[1,1,1]]).astype(dtype)
            else:
                signal = ht.arange(0, 16, split=0).astype(dtype)
                kernel = ht.ones(3).astype(dtype)
            return signal, kernel

        for conv_dim in [1, 2]:
            if conv_dim == 1:
                stride = 1
            elif conv_dim == 2:
                stride = (1,1)

            # check for any type except float and integer
            for dtype in [ht.bool, ht.complex64, ht.uint8]:
                signal, kernel = get_signal_kernel(dtype, conv_dim)

                with self.assertRaises(TypeError):
                    conv_input_check(signal, kernel, stride, mode, conv_dim)

            # integer not supported for mps and gpu conv2d, int only possible for conv1d on gpu
            for dtype in [ht.int8, ht.int16, ht.int32, ht.int64]:
                signal, kernel = get_signal_kernel(dtype, conv_dim)

                if self.is_mps:
                    with self.assertRaises(TypeError):
                        conv_input_check(signal, kernel, stride, mode, conv_dim)
                elif "gpu" in ht.get_device().device_type and conv_dim > 1:
                    with self.assertRaises(TypeError):
                        conv_input_check(signal, kernel, stride, mode, conv_dim)

            # float should always pass
            for dtype in [ht.float16, ht.float32, ht.float64]:
                signal, kernel = get_signal_kernel(dtype, conv_dim)

                try:
                    conv_input_check(signal, kernel, stride, mode, conv_dim)
                except TypeError:
                    assert False

    def test_conv_input_check_scalar(self):
        a = float(1)
        v = float(2)
        mode = "full"

        for conv_dim in [1, 2]:
            if conv_dim == 1:
                stride = 1
            else:
                stride = tuple([1] * conv_dim)
            a_out, v_out = conv_input_check(a, v, stride, mode, conv_dim)

            if conv_dim == 1:
                target_shape = (1,)
            else:
                target_shape = (1, 1)
            assert a_out.shape == target_shape
            assert v_out.shape == target_shape

    def test_conv_input_check_stride(self):
        dis_signal = ht.arange(0, 16, split=0).astype(ht.float)
        kernel_odd = ht.ones(3).astype(ht.float)
        kernel_even = ht.array([1, 1, 1, 1]).astype(ht.float)

        for conv_dim in [1, 2]:
            if conv_dim == 2:
                dis_signal = dis_signal.reshape((conv_dim, -1))
                kernel_even = kernel_even.reshape((conv_dim, -1))
                kernel_odd = ht.ones((3,3)).astype(ht.float)
                stride_0 = (1, 0)
                stride_2 = (1, 2)
            else:
                stride_0 = 0
                stride_2 = 2

            # stride not positive integer
            with self.assertRaises(ValueError):
                conv_input_check(
                    dis_signal,
                    kernel_even,
                    stride=stride_0,
                    mode="full",
                    convolution_dim=conv_dim,
                )

            # stride > 1 for mode 'same'
            with self.assertRaises(ValueError):
                conv_input_check(
                    dis_signal,
                    kernel_odd,
                    stride=stride_2,
                    mode="same",
                    convolution_dim=conv_dim,
                )

    def test_conv_input_check_even_kernel_mode_same(self):
        dis_signal = ht.arange(0, 16, split=0).astype(ht.float)
        kernel_even = [1, 1, 1, 1]

        for conv_dim in [1, 2]:
            if conv_dim == 1:
                stride = 1
            else:
                stride = tuple([1] * conv_dim)
                kernel_even = [kernel_even]
                dis_signal = dis_signal.reshape((conv_dim, -1))

            # Even kernel for mode same
            with self.assertRaises(ValueError):
                conv_input_check(dis_signal, kernel_even, stride, "same", conv_dim)

    def test_conv_input_check_flip_a_v(self):
        mode = "full"
        signal = ht.arange(0, 16, split=0).reshape((2, 8)).astype(ht.float)
        for conv_dim in [1, 2]:
            if conv_dim == 1:
                stride = 1
            else:
                stride = tuple([1] * conv_dim)

            # switch, all dimensions larger
            kernel = ht.ones((1, 3)).astype(ht.float)
            signal_out, kernel_out = conv_input_check(
                kernel, signal, stride, mode, conv_dim
            )
            self.assertTrue(ht.equal(kernel_out, kernel))
            self.assertTrue(ht.equal(signal_out, signal))

            # switch, mix equal and larger dimensions
            if conv_dim == 2:
                kernel = ht.ones((2, 3))
                signal_out, kernel_out = conv_input_check(
                    kernel, signal, stride, mode, conv_dim
                )
                self.assertTrue(ht.equal(kernel_out, kernel))
                self.assertTrue(ht.equal(signal_out, signal))

            # no switch
            kernel_signal = ht.ones(16).reshape((2, 8)).astype(ht.float)
            signal_out, kernel_out = conv_input_check(
                kernel_signal, signal, stride, mode, conv_dim
            )
            self.assertTrue(ht.equal(kernel_out, signal))
            self.assertTrue(ht.equal(signal_out, kernel_signal))

    def test_conv_input_check_flip_2d_error(self):
        dis_signal = ht.arange(0, 16, split=0).reshape((2, 8)).astype(ht.float)
        test_kernels = [ht.ones((1, 10)), ht.ones((10, 1))]
        stride = (1, 1)
        mode = "full"
        conv_dim = 2

        for kernel in test_kernels:
            with self.assertRaises(ValueError):
                conv_input_check(dis_signal, kernel, stride, mode, conv_dim)

    def test_conv_batchprocessing_check_1d_errors(self):
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)

        # kernel dimensions greater than 1 but convolution dims not
        with self.assertRaises(ValueError):  # batchprocessing check
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
        self.assertTrue(conv_batchprocessing_check(batch_signal, kernel, 1))

        dis_kernel = ht.array(kernel, dtype=ht.float32, split=0)
        self.assertTrue(conv_batchprocessing_check(batch_signal, dis_kernel, 1))

        batch_kernel = ht.empty((10, 19), dtype=ht.float32, split=1)
        batch_kernel.larray[:] = dis_kernel.larray
        self.assertTrue(conv_batchprocessing_check(batch_signal, batch_kernel, 1))

    def test_conv_batchprocessing_check_2d_error(self):
        dis_signal = ht.arange(0, 16, split=0).reshape((2, 8)).astype(ht.int)
        kernel_odd = ht.ones((2, 3)).astype(ht.int)

        # kernel dimensions greater than 2 but convolution dims not
        with self.assertRaises(ValueError):
            k = ht.eye(3).reshape((1, 3, 3))
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
            for split_axis in (1, 2):
                with self.assertRaises(ValueError):
                    signal_wrong_split = dis_signal.reshape((2, 2, 4)).resplit(
                        axis=split_axis
                    )
                    conv_batchprocessing_check(signal_wrong_split, kernel_odd, 2)

    def test_conv_batchprocessing_check_2d_true(self):
        # set batch processing true
        signal = ht.random.randn(10, 100, dtype=ht.float32)
        batch_signal = ht.empty((10, 10, 100), dtype=ht.float32, split=0)
        batch_signal.larray[:] = signal.larray
        kernel = ht.random.randn(3, 19, dtype=ht.float32)
        self.assertTrue(conv_batchprocessing_check(batch_signal, kernel, 2))

        dis_kernel = ht.random.randn(3, 19, dtype=ht.float32, split=0)
        self.assertTrue(conv_batchprocessing_check(batch_signal, dis_kernel, 2))

        batch_kernel = ht.empty((10, 3, 19), dtype=ht.float32, split=1)
        batch_kernel.larray[:] = dis_kernel.larray
        self.assertTrue(conv_batchprocessing_check(batch_signal, batch_kernel, 2))

    def test_only_balanced_kernel(self):
        for conv_dim in (1, 2):
            if conv_dim == 1:
                signal = ht.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10], split=0).astype(
                    ht.float32
                )
                dis_kernel = ht.array([1, 1], split=0).astype(ht.float32)
                test_target = ht.convolve
            else:
                signal = ht.array([[0, 1, 3, 4, 5], [6, 7, 8, 9, 10]], split=0).astype(
                    ht.float32
                )
                dis_kernel = ht.array([[1, 1], [1, 1]], split=0).astype(ht.float32)
                test_target = ht.convolve2d

            if self.comm.size > 1:
                target_map = dis_kernel.lshape_map
                target_map[0, 0] = 2
                target_map[1:, 0] = 0
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
        full_ones = ht.ones((7,3), split=1).astype(ht.int)
        kernel_even = [[1, 1], [1, 1]]
        kernel_odd = [[1,1,1], [1,1,1]]
        if self.comm.size > 4:
            with self.assertRaises(ValueError):
                ht.convolve2d(full_ones, kernel_even, mode="valid")
            with self.assertRaises(ValueError):
                ht.convolve2d(kernel_even, full_ones, mode="valid")

            full_ones.resplit_(0)
            with self.assertRaises(ValueError):
                ht.convolve2d(full_ones, kernel_odd)

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

        # kernel without batch dimensions
        kernel = ht.random.randn(19, dtype=float_dtype)
        batch_convolved = ht.convolve(batch_signal, kernel, mode="same")
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, mode="same"), batch_convolved[0])
        )

        # distributed kernel including gathering to all ranks
        dis_kernel = ht.array(kernel, split=0)
        batch_convolved = ht.convolve(batch_signal, dis_kernel)
        self.assertTrue(ht.equal(ht.convolve(signal, kernel), batch_convolved[0]))

        # batch kernel including resplit to signal axis
        batch_kernel = ht.empty((10, 19), dtype=float_dtype, split=1)
        batch_kernel.larray[:] = dis_kernel.larray
        batch_convolved = ht.convolve(batch_signal, batch_kernel, mode="full")
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, mode="full"), batch_convolved[0])
        )

        # n-D batch convolution
        batch_signal = ht.empty((4, 3, 3, 1000), dtype=float_dtype, split=1)
        batch_signal.larray[:, :, :] = signal.larray
        batch_convolved = ht.convolve(batch_signal, kernel, mode="valid")
        self.assertTrue(
            ht.equal(
                ht.convolve(signal, kernel, mode="valid"), batch_convolved[1, 2, 0]
            )
        )

    def test_convolve_stride_batch_convolutions(self):
        float_dtype = ht.float32 if self.is_mps else ht.float64

        # distributed input along the first axis
        signal = ht.random.randn(1000, dtype=float_dtype)
        batch_signal = ht.empty((10, 1000), dtype=float_dtype, split=0)
        batch_signal.larray[:] = signal.larray

        # kernel without batch dimensions
        stride = 123
        kernel = ht.random.randn(19, dtype=float_dtype)
        batch_convolved = ht.convolve(batch_signal, kernel, mode="valid", stride=stride)
        self.assertTrue(
            ht.equal(
                ht.convolve(signal, kernel, mode="valid", stride=stride),
                batch_convolved[0],
            )
        )

        # distributed kernel including gathering to all ranks
        stride = 142
        dis_kernel = ht.array(kernel, split=0)
        batch_convolved = ht.convolve(batch_signal, dis_kernel, stride=stride)
        self.assertTrue(
            ht.equal(ht.convolve(signal, kernel, stride=stride), batch_convolved[0])
        )

        # batch kernel including resplit to signal axis
        stride = 41
        batch_kernel = ht.empty((10, 19), dtype=float_dtype, split=1)
        batch_kernel.larray[:] = dis_kernel.larray

        batch_convolved = ht.convolve(batch_signal, batch_kernel, mode="full", stride=stride)
        self.assertTrue(
            ht.equal(
                ht.convolve(signal, kernel, mode="full", stride=stride),
                batch_convolved[0],
            )
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

    def test_convolve2d_batch_convolutions(self):
        float_dtype = ht.float32 if self.is_mps else ht.float64

        if self.comm.size < 5:
            # distributed input along the first axis
            signal = ht.random.randn(10, 100, dtype=float_dtype)
            batch_signal = ht.empty((10, 10, 100), dtype=float_dtype, split=0)
            batch_signal.larray[:] = signal.larray

            # kernel without batch dimensions
            kernel = ht.random.randn(5, 19, dtype=float_dtype)
            batch_convolved = ht.convolve2d(batch_signal, kernel, mode="valid")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="valid"),
                    batch_convolved[0],
                )
            )

            # distributed kernel including gathering to all ranks
            dis_kernel = ht.array(kernel, split=0)
            batch_convolved = ht.convolve2d(batch_signal, dis_kernel)
            self.assertTrue(
                ht.equal(ht.convolve2d(signal, kernel),
                         batch_convolved[5])
            )

            # batch kernel including resplit to signal axis
            batch_kernel = ht.empty((10, 5, 19), dtype=float_dtype, split=1)
            batch_kernel.larray[:] = dis_kernel.larray
            batch_convolved = ht.convolve2d(batch_signal, batch_kernel, mode="same")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="same"),
                    batch_convolved[-1],
                )
            )

            # n-D batch convolution
            batch_signal = ht.empty((4, 5, 3, 10, 100), dtype=float_dtype, split=1)
            batch_signal.larray[:, :, :] = signal.larray
            batch_convolved = ht.convolve2d(batch_signal, kernel, mode="valid")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="valid"),
                    batch_convolved[1, 2, 0]
                )
            )

    def test_convolve2d_stride_batch_convolutions(self):
        float_dtype = ht.float32 if self.is_mps else ht.float64

        if self.comm.size < 5:
            # distributed input along the first axis
            signal = ht.random.randn(10, 100, dtype=float_dtype)
            batch_signal = ht.empty((10, 10, 100), dtype=float_dtype, split=0)
            batch_signal.larray[:] = signal.larray

            # kernel without batch dimensions
            stride = (3,15)
            kernel = ht.random.randn(5, 19, dtype=float_dtype)
            batch_convolved = ht.convolve2d(batch_signal, kernel, mode="valid", stride=stride)
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="valid", stride=stride),
                    batch_convolved[0],
                )
            )

            # distributed kernel including gathering to all ranks
            stride = (2,4)
            dis_kernel = ht.array(kernel, split=0)
            batch_convolved = ht.convolve2d(batch_signal, dis_kernel, stride=stride)
            self.assertTrue(
                ht.equal(ht.convolve2d(signal, kernel, stride=stride),
                         batch_convolved[5])
            )

            # batch kernel including resplit to signal axis
            stride = (3,4)
            batch_kernel = ht.empty((10, 5, 19), dtype=float_dtype, split=1)
            batch_kernel.larray[:] = dis_kernel.larray
            batch_convolved = ht.convolve2d(batch_signal, batch_kernel, mode="full",
                                          stride=stride)
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="full", stride=stride),
                    batch_convolved[-1],
                )
            )

            # n-D batch convolution
            stride = (4,3)
            batch_signal = ht.empty((4, 5, 3, 10, 100), dtype=float_dtype, split=1)
            batch_signal.larray[:, :, :] = signal.larray
            batch_convolved = ht.convolve2d(batch_signal, kernel, mode="valid", stride=stride)
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="valid", stride=stride),
                    batch_convolved[1, 2, 0]
                )
            )

    def test_convolve_kernel_odd_modes(self):
        ht_dtype = ht.int

        np_sig = np.arange(0,16)
        np_k_odd = np.arange(3)
        full_odd = ht.array(sig.convolve(np_sig, np_k_odd)).astype(ht_dtype)

        signal = ht.array(np_sig).astype(ht_dtype)
        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)

        kernel_odd = ht.array(np_k_odd).astype(ht.int)
        dis_kernel_odd = ht.array(np_k_odd, split=0).astype(ht.int)

        # avoid kernel larger than signal chunk
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

    def test_convolve2d_kernel_odd_modes(self):
        ht_dtype = ht.int

        np_sig = np.arange(256).reshape((16, 16))
        np_k_odd = np.arange(9).reshape((3, 3))
        full_odd = ht.array(sig.convolve2d(np_sig, np_k_odd)).astype(ht_dtype)

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_odd = ht.array(np_k_odd).astype(ht_dtype)
        dis_kernel_odd = ht.array(np_k_odd, split=0).astype(ht_dtype)

        #avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "same", "valid"]
            for i, mode in enumerate(modes):
                # odd kernel size
                if ht.get_device() == ht.cpu:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve2d(dis_signal, kernel_odd, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered))

                    conv = ht.convolve2d(signal, dis_kernel_odd, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered))


                    conv = ht.convolve2d(dis_signal, dis_kernel_odd, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)

                    self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i, i : len(full_odd) - i], gathered))

                # different data types
                conv = ht.convolve2d(dis_signal.astype(ht.float), kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve2d(signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

    def test_convolve_stride_kernel_odd_mode_full(self):
        ht_dtype = ht.int

        mode = "full"
        stride = 2

        np_sig = np.arange(0, 16)
        np_k_odd = np.arange(3)
        full_odd = ht.array(sig.convolve(np_sig, np_k_odd)).astype(ht_dtype)

        signal = ht.array(np_sig).astype(ht_dtype)
        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)

        kernel = ht.array(np_k_odd).astype(ht.int)
        dis_kernel = ht.array(np_k_odd, split = 0).astype(ht.int)

        # avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            if not self.is_mps:
                # torch convolution does not support int on MPS
                self.assert_convolution_stride(
                    dis_signal, kernel, mode, stride, full_odd[::stride])
                self.assert_convolution_stride(
                    signal, dis_kernel, mode, stride, full_odd[::stride])
                self.assert_convolution_stride(
                    dis_signal, dis_kernel, mode, stride, full_odd[::stride])

            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, full_odd[::stride])
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, full_odd[::stride])
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, full_odd[::stride])

    def test_convolve_stride_kernel_odd_mode_valid(self):
        ht_dtype = ht.int

        mode = "valid"
        stride = 2

        np_sig = np.arange(0, 16)
        np_k_odd = np.arange(3)
        valid_odd = ht.array(sig.convolve(np_sig, np_k_odd, mode="valid")).astype(ht_dtype)

        signal = ht.array(np_sig).astype(ht_dtype)
        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)

        kernel = ht.array(np_k_odd).astype(ht.int)
        dis_kernel = ht.array(np_k_odd, split=0).astype(ht.int)

        # avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            if not self.is_mps:
                # torch convolution does not support int on MPS
                self.assert_convolution_stride(
                    dis_signal, kernel, mode, stride, valid_odd[::stride])

                self.assert_convolution_stride(
                    signal, dis_kernel, mode, stride, valid_odd[::stride])

                self.assert_convolution_stride(
                    dis_signal, dis_kernel, mode, stride, valid_odd[::stride])

            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, valid_odd[::stride])
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, valid_odd[::stride])
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, valid_odd[::stride])

    def test_convolve2d_stride_kernel_odd_mode_full(self):
        ht_dtype = ht.int

        np_sig = np.arange(256).reshape((16, 16))
        np_k_odd = np.arange(9).reshape((3, 3))
        full_odd = ht.array(sig.convolve2d(np_sig, np_k_odd)).astype(ht_dtype)

        mode = "full"
        strides = [(1,2), (2,1), (2,2)]

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_odd = ht.array(np_k_odd).astype(ht_dtype)
        dis_kernel_odd = ht.array(np_k_odd, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        for stride in strides:
            if self.comm.size <= 3:
                if ht.get_device() == ht.cpu:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve2d(dis_signal, kernel_odd, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(signal, dis_kernel_odd, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                        ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(dis_signal, dis_kernel_odd, stride=stride)
                    self.assertTrue(
                        ht.equal(full_odd[::stride[0], ::stride[1]], gathered))



                # different data types of input and kernel
                conv = ht.convolve2d(dis_signal.astype(ht.float), kernel_odd, stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[::stride[0], ::stride[1]].astype(ht.float), gathered))

                conv = ht.convolve2d(signal.astype(ht.float), dis_kernel_odd, stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float), dis_kernel_odd, stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

    def test_convolve2d_stride_kernel_odd_mode_valid(self):
        ht_dtype = ht.int

        mode = "valid"
        np_sig = np.arange(256).reshape((16, 16))
        np_k_odd = np.ones(9).reshape((3, 3))
        full_odd = ht.array(sig.convolve2d(np_sig, np_k_odd, mode=mode)).astype(ht_dtype)

        strides = [(1, 2), (2, 1), (2, 2)]

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_odd = ht.array(np_k_odd).astype(ht_dtype)
        dis_kernel_odd = ht.array(np_k_odd, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        for stride in strides:
            if self.comm.size <= 3:
                if ht.get_device() == ht.cpu:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve2d(dis_signal, kernel_odd, mode=mode, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                        ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(signal, dis_kernel_odd, mode=mode, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                        ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(dis_signal, dis_kernel_odd, mode=mode,
                                         stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                        ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                # different data types of input and kernel
                conv = ht.convolve2d(dis_signal.astype(ht.float), kernel_odd, mode=mode,
                                     stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(
                        full_odd[::stride[0], ::stride[1]].astype(ht.float),
                        gathered))

                conv = ht.convolve2d(signal.astype(ht.float), dis_kernel_odd, mode=mode,
                                     stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float),
                                     dis_kernel_odd, mode=mode, stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

    def test_convolve_kernel_even_modes(self):
        full_even = ht.array(
            [0, 1, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 42, 29, 15]
        ).astype(ht.int)

        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_even = [1, 1, 1, 1]
        dis_kernel_even = ht.ones(4, split=0).astype(ht.int)

        # test modes, avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "valid"]
            for mode in modes:
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
                    conv = ht.convolve(
                        dis_signal.astype(ht.float), kernel_even, mode=mode
                    )
                    dis_conv = ht.convolve(
                        dis_signal.astype(ht.float),
                        dis_kernel_even.astype(ht.float),
                        mode=mode,
                    )
                    gathered = manipulations.resplit(conv, axis=None)
                    dis_gathered = manipulations.resplit(dis_conv, axis=None)

                    if mode == "full":
                        self.assertTrue(ht.equal(full_even.astype(ht.float), gathered))
                        self.assertTrue(
                            ht.equal(full_even.astype(ht.float), dis_gathered)
                        )
                    else:
                        self.assertTrue(
                            ht.equal(full_even[3:-3].astype(ht.float), gathered)
                        )
                        self.assertTrue(
                            ht.equal(full_even[3:-3].astype(ht.float), dis_gathered)
                        )

    def test_convolve2d_kernel_even_modes(self):
        ht_dtype = ht.int if self.is_mps else ht.float
        np_sig = np.arange(256).reshape((16, 16))
        np_k_even = np.ones(16).reshape((4, 4))
        full_even = ht.array(sig.convolve2d(np_sig, np_k_even))

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_even = ht.array(np_k_even).astype(ht_dtype)
        dis_kernel_even = ht.array(np_k_even, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "valid"]
            for mode in modes:

                conv = ht.convolve2d(dis_signal, kernel_even, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                if mode == "full":
                    self.assertTrue(ht.equal(full_even, gathered))
                else:
                    self.assertTrue(ht.equal(full_even[3:-3, 3:-3], gathered))

                conv = ht.convolve2d(dis_signal, dis_kernel_even, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                if mode == "full":
                    self.assertTrue(ht.equal(full_even, gathered))
                else:
                    self.assertTrue(ht.equal(full_even[3:-3, 3:-3], gathered))

                conv = ht.convolve2d(signal, dis_kernel_even, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                if mode == "full":
                    self.assertTrue(ht.equal(full_even, gathered))
                else:
                    self.assertTrue(ht.equal(full_even[3:-3, 3:-3], gathered))

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
                self.assert_convolution_stride(
                    dis_signal, kernel, mode, stride, solution
                )
                self.assert_convolution_stride(
                    signal, dis_kernel, mode, stride, solution
                )
                self.assert_convolution_stride(
                    dis_signal, dis_kernel, mode, stride, solution
                )

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
                self.assert_convolution_stride(
                    dis_signal, kernel, mode, stride, solution
                )
                self.assert_convolution_stride(
                    signal, dis_kernel, mode, stride, solution
                )
                self.assert_convolution_stride(
                    dis_signal, dis_kernel, mode, stride, solution
                )

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

    def test_convolve2d_stride_kernel_even_mode_full(self):
        ht_dtype = ht.int

        np_sig = np.arange(256).reshape((16, 16))
        np_k_even = np.arange(4).reshape((2,2))
        full_even = ht.array(sig.convolve2d(np_sig, np_k_even)).astype(ht_dtype)

        mode = "full"
        strides = [(1, 2), (2, 1), (2, 2)]


        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_even = ht.array(np_k_even).astype(ht_dtype)
        dis_kernel_even = ht.array(np_k_even, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        for stride in strides:
            if self.comm.size <= 3:
                if ht.get_device() == ht.cpu:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve2d(dis_signal, kernel_even, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                        ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(signal, dis_kernel_even, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None) # this resplit fails,
                    self.assertTrue(
                        ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(dis_signal, dis_kernel_even,
                                         stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                        ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                # different data types of input and kernel
                conv = ht.convolve2d(dis_signal.astype(ht.float), kernel_even,
                                     stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(
                        full_even[::stride[0], ::stride[1]].astype(ht.float),
                        gathered))

                conv = ht.convolve2d(signal.astype(ht.float), dis_kernel_even,
                                     stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float),
                                     dis_kernel_even, stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_even[::stride[0], ::stride[1]], gathered))

    def test_convolve2d_stride_kernel_even_mode_valid(self):
        ht_dtype = ht.int

        mode = "valid"
        np_sig = np.arange(256).reshape((16, 16))
        np_k_even = np.ones(16).reshape((4,4))
        full_even = ht.array(sig.convolve2d(np_sig, np_k_even, mode=mode)).astype(
            ht_dtype)

        strides = [(1, 2), (2, 1), (2, 2)]

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_even = ht.array(np_k_even).astype(ht_dtype)
        dis_kernel_even = ht.array(np_k_even, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        for stride in strides:
            if self.comm.size <= 3:
                if ht.get_device() == ht.cpu:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve2d(dis_signal, kernel_even, mode=mode,
                                         stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                        ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(signal, dis_kernel_even, mode=mode,
                                         stride=stride)

                    gathered = manipulations.resplit(conv, axis=None)

                    self.assertTrue(
                        ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(dis_signal, dis_kernel_even, mode=mode,
                                       stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(
                         ht.equal(full_even[::stride[0], ::stride[1]], gathered))



                # different data types of input and kernel
                conv = ht.convolve2d(dis_signal.astype(ht.float), kernel_even,
                                     mode=mode,
                                     stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(
                        full_even[::stride[0], ::stride[1]].astype(ht.float),
                        gathered))

                conv = ht.convolve2d(signal.astype(ht.float), dis_kernel_even,
                                    mode=mode,
                                    stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float),
                                     dis_kernel_even, mode=mode, stride=stride)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(
                    ht.equal(full_even[::stride[0], ::stride[1]], gathered))

    def test_convolve_large_signal_and_kernel_modes(self):
        if self.comm.size <= 4:
            # prep
            np.random.seed(12)
            np_a = np.random.randint(1000, size=4418)
            np_b = np.random.randint(1000, size=913)
            # torch convolution does not support int on MPS
            ht_dtype = ht.float32 if self.is_mps else ht.int32
            np_dtype = np.float32 if self.is_mps else np.int32
            random_stride = np.random.randint(1, high=len(np_a), size=1)[0]

            for mode in ["full", "same", "valid"]:
                strides = [1, random_stride] if mode != "same" else [1]
                for stride in strides:
                    # solution
                    np_conv = np.convolve(np_a, np_b, mode=mode)
                    solution = np_conv[::stride].astype(np_dtype)

                    # test
                    a = ht.array(np_a, split=0, dtype=ht_dtype)
                    b = ht.array(np_b, split=None, dtype=ht_dtype)
                    conv = ht.convolve(a, b, mode=mode, stride=stride)
                    self.assert_array_equal(conv, solution)

                    b = ht.array(np_b, split=0, dtype=ht_dtype)
                    conv = ht.convolve(a, b, mode=mode, stride=stride)
                    self.assert_array_equal(conv, solution)

    def test_convolve2d_large_signal_and_kernel_modes(self):
        if self.comm.size <= 4:
            np.random.seed(12)
            ht_dtype = ht.int32 if ht.get_device() == ht.cpu else ht.float32
            np_dtype = np.int32 if ht.get_device() == ht.cpu else np.float32

            np_a = np.random.randint(0,100, size=(734, 680)).astype(np_dtype)
            np_b = np.random.randint(0,10, size=(39, 17)).astype(np_dtype)
            #np_b = np.arange(585).reshape((39,15))

            # np_b = np.zeros((39,17))
            #np_b = np.ones((39,17))
            random_stride = tuple(np.random.randint(1, high=20, size=2))
            for mode in ["full", "same", "valid"]:
                strides = [(1,1), random_stride] if mode != "same" else [(1,1)]
                for stride in strides:
                    sc_conv = sig.convolve2d(np_a, np_b, mode=mode)
                    solution = sc_conv[::stride[0], ::stride[1]]

                    a = ht.array(np_a, split=0, dtype=ht_dtype)
                    b = ht.array(np_b, split=None, dtype=ht_dtype)
                    conv = ht.convolve2d(a, b, mode=mode, stride=stride)
                    self.assert_array_equal(conv, solution)

                    b = ht.array(np_b, split=0, dtype=ht_dtype)
                    conv = ht.convolve2d(a, b, mode=mode, stride=stride)
                    self.assert_array_equal(conv, solution)

                    a = ht.array(np_a, split=1, dtype=ht_dtype)
                    b = ht.array(np_b, split=None, dtype=ht_dtype)
                    conv = ht.convolve2d(a, b, mode=mode, stride=stride)
                    self.assert_array_equal(conv, solution)

                    b = ht.array(np_b, split=1, dtype=ht_dtype)
                    conv = ht.convolve2d(a, b, mode=mode, stride=stride)
                    self.assert_array_equal(conv, solution)

    def test_convolve_kernel_size_1(self):
        # prep
        ht_dtype = ht.float32 if self.is_mps else ht.int32

        # non-distributed signal
        signal = ht.arange(0, 16, dtype=ht_dtype)
        alt_signal = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        kernel = ht.ones(1, dtype=ht_dtype)
        for stride in range(1, 4):
            conv = ht.convolve(alt_signal, kernel, stride=stride)
            self.assertTrue(ht.equal(signal[0::stride], conv))

            if not self.is_mps:
                conv = ht.convolve(1, 5, stride=stride)
                self.assertTrue(ht.equal(ht.array([5]), conv))

    def test_convolve2d_kernel_size_1(self):
        # prep
        ht_dtype = ht.int32 if ht.get_device() == ht.cpu else ht.float32

        signal = ht.arange(0, 16, dtype=ht_dtype).reshape(4,4)
        alt_signal = [[0,1,2,3], [4,5,6,7], [8,9,10,11],[12,13,14,15]]
        kernel = ht.ones(1,dtype=ht_dtype).reshape(1,1)

        for stride in range(1,4):
            conv = ht.convolve2d(alt_signal, kernel, stride=(stride,stride))
            self.assertTrue(ht.equal(signal[::stride, ::stride], conv))

            if not self.is_mps:
                conv = ht.convolve2d(float(1),float(5),stride=(stride,stride))
                self.assertTrue(ht.equal(ht.array([[5]]), conv))
