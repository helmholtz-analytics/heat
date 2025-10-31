import numpy as np
import torch
import heat as ht
from heat import manipulations
import scipy.signal as sig
from .test_suites.basic_test import TestCase
import os

from ..signal import conv_input_check, conv_batchprocessing_check, conv_pad


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
                    dis_signal = dis_signal.reshape((conv_dim, -1))
                    kernel_even = kernel_even.reshape((conv_dim, -1))
                conv_input_check(dis_signal, kernel_even, stride, "invalid", conv_dim)

    def test_conv_input_check_scaler(self):
        a = 1
        v = 2
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
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = ht.array([1, 1, 1, 1]).astype(ht.int)

        for conv_dim in [1, 2]:
            if conv_dim == 2:
                dis_signal = dis_signal.reshape((conv_dim, -1))
                kernel_even = kernel_even.reshape((conv_dim, -1))
                kernel_odd = ht.ones((3,3)).astype(ht.int)
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
        dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
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
        signal = ht.arange(0, 16, split=0).reshape((2, 8)).astype(ht.int)
        for conv_dim in [1, 2]:
            if conv_dim == 1:
                stride = 1
            else:
                stride = tuple([1] * conv_dim)

            # switch, all dimensions larger
            kernel = ht.ones((1, 3)).astype(ht.int)
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
            kernel_signal = ht.ones(16).reshape((2, 8)).astype(ht.int)
            signal_out, kernel_out = conv_input_check(
                kernel_signal, signal, stride, mode, conv_dim
            )
            self.assertTrue(ht.equal(kernel_out, signal))
            self.assertTrue(ht.equal(signal_out, kernel_signal))

    def test_conv_input_check_flip_2d_error(self):
        dis_signal = ht.arange(0, 16, split=0).reshape((2, 8)).astype(ht.int)
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

    def test_conv_pad_boundary_error(self):

        signal1d = ht.arange(16, dtype=ht.float32).reshape(1, 1, 16)
        signal2d = ht.zeros((1, 1, 10, 16), dtype=ht.float32)
        signal2d[:] = ht.arange(16, dtype=ht.float32)
        padding = [2, 2, 1, 1]
        boundary = "bla"
        fillvalue = 100

        for i, signal in enumerate([signal1d, signal2d]):
            with self.assertRaises(ValueError):
                local_signal = signal.larray
                conv_pad(signal, i + 1, local_signal, padding, boundary, fillvalue)

    def test_conv_pad_circular_error(self):
        dis_signal1d = ht.ones(20, dtype=ht.float32, split=0)
        dis_signal2d = ht.ones((20, 30), dtype=ht.float32, split=0)
        padding = [2,2,4,4]
        boundary = "circular"
        if self.comm.size > 1:
            local_signal = dis_signal1d.larray
            local_signal = local_signal.reshape(1,1,local_signal.shape[0])
            with self.assertRaises(ValueError):
                conv_pad(dis_signal1d, 1, local_signal, padding[0:2], boundary, 0)

            local_signal = dis_signal2d.larray
            local_signal = local_signal.reshape(1,1, local_signal.shape[0], local_signal.shape[1])
            with self.assertRaises(ValueError):
                conv_pad(dis_signal2d, 2, local_signal, padding, boundary, 0)

            dis_signal2d.resplit_(1)
            local_signal = dis_signal2d.larray
            local_signal = local_signal.reshape(1, 1, local_signal.shape[0], local_signal.shape[1])
            with self.assertRaises(ValueError):
                conv_pad(dis_signal2d, 2, local_signal, padding, boundary, 0)

    def test_conv_pad_reflect_error(self):
        dis_signal1d = ht.ones((1,1,20), dtype=ht.float32, split=2)
        dis_signal2d = ht.ones((1,1,20, 30), dtype=ht.float32, split=2)
        padding = [10,10,15,15]
        boundary = "reflect"

        if self.comm.size > 1:
            local_signal = dis_signal1d.larray
            with self.assertRaises(ValueError):
                conv_pad(dis_signal1d, 1, local_signal, padding[0:2], boundary, 0)

            local_signal = dis_signal2d.larray
            with self.assertRaises(ValueError):
                conv_pad(dis_signal2d, 2, local_signal, padding, boundary, 0)

            dis_signal2d.resplit_(3)
            local_signal = dis_signal2d.larray
            with self.assertRaises(ValueError):
                conv_pad(dis_signal2d, 2, local_signal, padding, boundary, 0)

    def test_conv_pad_circular(self):
        signal1d = ht.arange(16, dtype=ht.float32).reshape(1,1,16)
        signal2d = ht.zeros((1,1,10,16), dtype=ht.float32)
        signal2d[:] = ht.arange(16, dtype=ht.float32)
        padding = [2,2,1,1]
        boundary = "circular"

        local_signal = signal1d.larray
        padded_signal = ht.array(conv_pad(signal1d, 1, local_signal, padding[0:2], boundary, 0))
        self.assertTrue(ht.equal(padded_signal.squeeze()[:2], ht.array([14,15])))
        self.assertTrue(ht.equal(padded_signal.squeeze()[-2:], ht.array([0,1])))

        local_signal = signal2d.larray
        padded_signal = ht.array(conv_pad(signal2d, 2, local_signal, padding, boundary, 0))
        self.assertTrue(ht.equal(padded_signal.squeeze()[:2,:1], signal2d.squeeze()[-2:,-1:]))
        self.assertTrue(ht.equal(padded_signal.squeeze()[-2:,-1:], signal2d.squeeze()[:2,:1]))

        # split axis not along convolution dimension

        signal1d.resplit_(1)
        signal2d.resplit_(1)

        local_signal = signal1d.larray
        padded_signal = conv_pad(signal1d, 1, local_signal, padding[0:2], boundary, 0)

        if self.comm.rank == 0:
            self.assertTrue(
                torch.equal(padded_signal.squeeze()[:2], torch.tensor([14, 15])))
            self.assertTrue(
                torch.equal(padded_signal.squeeze()[-2:], torch.tensor([0, 1])))


        local_signal = signal2d.larray
        padded_signal = conv_pad(signal2d, 2, local_signal, padding, boundary, 0)

        if self.comm.rank == 0:
            self.assertTrue(torch.equal(padded_signal.squeeze()[:2,:1], local_signal.squeeze()[-2:,-1:]))
            self.assertTrue(torch.equal(padded_signal.squeeze()[-2:,-1:], local_signal.squeeze()[:2,:1]))

        # split axis along convolution dimension but comm size == 1
        if self.comm.size == 1:
            signal1d.resplit_(2)
            signal2d.resplit_(2)

            local_signal = signal1d.larray
            padded_signal = ht.array(
                conv_pad(signal1d, 1, local_signal, padding[0:2],
                         boundary, 0))
            self.assertTrue(
                ht.equal(padded_signal.squeeze()[:2], ht.array([14, 15])))
            self.assertTrue(
                ht.equal(padded_signal.squeeze()[-2:], ht.array([0, 1])))

            local_signal = signal2d.larray
            padded_signal = ht.array(
                conv_pad(signal2d, 2, local_signal, padding, boundary,
                         0))
            self.assertTrue(ht.equal(padded_signal.squeeze()[:2, :1],
                                     signal2d.squeeze()[-2:, -1:]))
            self.assertTrue(ht.equal(padded_signal.squeeze()[-2:, -1:],
                                     signal2d.squeeze()[:2, :1]))

    def test_conv_pad_reflect(self):
        signal1d = ht.arange(16, dtype=ht.float32).reshape(1, 1, 16)
        signal2d = ht.zeros((1, 1, 16, 16), dtype=ht.float32)
        signal2d[:] = ht.arange(16, dtype=ht.float32)
        padding = [2, 2, 1, 1]
        boundary = "reflect"

        local_signal = signal1d.larray
        padded_signal = ht.array(
            conv_pad(signal1d, 1, local_signal, padding[0:2], boundary, 0))
        self.assertTrue(ht.equal(padded_signal.squeeze()[:2], ht.array([2,1])))
        self.assertTrue(ht.equal(padded_signal.squeeze()[-2:], ht.array([14,13])))

        local_signal = signal2d.larray
        padded_signal = ht.array(
            conv_pad(signal2d, 2, local_signal, padding, boundary, 0))
        self.assertTrue(
            ht.equal(padded_signal.squeeze()[:2, :1],
                     manipulations.flip(signal2d.squeeze()[1:3, 1:2], [0, 1])))
        self.assertTrue(
            ht.equal(padded_signal.squeeze()[-2:, -1:],
                     manipulations.flip(signal2d.squeeze()[-3:-1, -2:-1], [0, 1])))

        # split axis not along convolution dimension
        signal1d.resplit_(1)
        signal2d.resplit_(1)

        local_signal = signal1d.larray

        if self.comm.rank == 0:
            padded_signal = conv_pad(signal1d, 1, local_signal, padding[0:2],
                                     boundary, 0)

            self.assertTrue(torch.equal(padded_signal.squeeze()[:2], torch.tensor([2, 1])))
            self.assertTrue(
                torch.equal(padded_signal.squeeze()[-2:], torch.tensor([14, 13])))

        local_signal = signal2d.larray

        if self.comm.rank == 0:
            padded_signal = conv_pad(signal2d, 2, local_signal, padding,
                                     boundary, 0)

            self.assertTrue(
                torch.equal(padded_signal.squeeze()[:2, :1],
                         torch.flip(local_signal.squeeze()[1:3, 1:2], [0, 1])))
            self.assertTrue(
                torch.equal(padded_signal.squeeze()[-2:, -1:],
                         torch.flip(local_signal.squeeze()[-3:-1, -2:-1], [0, 1])))

        # split axis along convolution dimension
        signal1d.resplit_(2)
        signal2d.resplit_(2)

        local_signal = signal1d.larray
        padded_signal = ht.array(
            conv_pad(signal1d, 1, local_signal, padding[0:2], boundary,
                     0)).squeeze()
        if self.comm.size == 1:
            self.assertTrue(
                ht.equal(padded_signal[:2], ht.array([2, 1])))
            self.assertTrue(
                ht.equal(padded_signal[-2:], ht.array([14, 13])))
        else:
            if self.comm.rank == 0:
                self.assertTrue(
                    ht.equal(padded_signal[:2], ht.array([2, 1])))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1*2)
            elif self.comm.rank == self.comm.size -1:
                self.assertTrue(
                    ht.equal(padded_signal.squeeze()[-2:], ht.array([14, 13])))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
            else:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0])

        local_signal = signal2d.larray
        padded_signal = conv_pad(signal2d, 2, local_signal, padding, boundary, 0).squeeze()
        if self.comm.size == 1:
            self.assertTrue(
                ht.equal(ht.array(padded_signal[:2, :1]),
                         manipulations.flip(signal2d.squeeze()[1:3, 1:2],[0, 1])))
            self.assertTrue(
                ht.equal(padded_signal[-2:, -1:],
                         manipulations.flip(signal2d.squeeze()[-3:-1, -2:-1], [0, 1])))
        else:
            if self.comm.rank == 0:
                self.assertTrue(
                    torch.equal(padded_signal[:2, :1], torch.flip(local_signal.squeeze()[1:3, 1:2], [0, 1])))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
                self.assertTrue(padded_signal.shape[1] == local_signal.squeeze().shape[1] + 2 * 1)
            elif self.comm.rank == self.comm.size - 1:
                self.assertTrue(torch.equal(padded_signal[-2:, -1:],
                                torch.flip(local_signal.squeeze()[-3:-1, -2:-1], [0, 1])))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
                self.assertTrue(padded_signal.shape[1] == local_signal.squeeze().shape[1] + 2 * 1)
            else:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0])
                self.assertTrue(padded_signal.shape[1] == local_signal.squeeze().shape[1] + 2 * 1)

    def test_conv_pad_replicate(self):
        signal1d = ht.arange(16, dtype=ht.float32).reshape(1, 1, 16)
        signal2d = ht.zeros((1, 1, 16, 16), dtype=ht.float32)
        signal2d[:] = ht.arange(16, dtype=ht.float32)
        padding = [2, 2, 1, 1]
        boundary = "replicate"

        local_signal = signal1d.larray
        padded_signal = ht.array(
            conv_pad(signal1d, 1, local_signal, padding[0:2], boundary, 0))
        self.assertTrue(
            ht.equal(padded_signal.squeeze()[:4], ht.array([0,0,0,1])))
        self.assertTrue(
            ht.equal(padded_signal.squeeze()[-4:], ht.array([14, 15, 15,15])))

        local_signal = signal2d.larray
        padded_signal = ht.array(conv_pad(signal2d, 2, local_signal, padding, boundary, 0))
        self.assertTrue(ht.equal(padded_signal.squeeze()[0, 1:-1], ht.arange(16, dtype=ht.float32)))
        self.assertTrue(ht.equal(padded_signal.squeeze()[-1, :4], ht.array([0,0,1,2])))
        self.assertTrue(ht.equal(padded_signal.squeeze()[-1, -4:], ht.array([13,14,15,15])))

        # split axis not along convolution dimension
        signal1d.resplit_(0)
        signal2d.resplit_(0)

        local_signal = signal1d.larray
        padded_signal = conv_pad(signal1d, 1, local_signal, padding[0:2], boundary, 0).squeeze()

        if self.comm.rank == 0:
            self.assertTrue(torch.equal(padded_signal[:4], torch.tensor([0, 0, 0, 1 ])))
            self.assertTrue(torch.equal(padded_signal[-4:], torch.tensor([14, 15, 15, 15])))

        local_signal = signal2d.larray
        padded_signal = conv_pad(signal2d, 2, local_signal, padding, boundary,0)

        if self.comm.rank == 0:
            self.assertTrue(torch.equal(padded_signal.squeeze()[0, 1:-1], torch.arange(16, dtype=torch.float32)))
            self.assertTrue(torch.equal(padded_signal.squeeze()[-1, :4], torch.tensor([0,0,1,2])))
            self.assertTrue(torch.equal(padded_signal.squeeze()[-1, -4:], torch.tensor([13, 14, 15, 15])))


        # split axis along convolution dimension but comm size == 1
        signal1d.resplit_(2)
        signal2d.resplit_(2)

        local_signal = signal1d.larray
        padded_signal = conv_pad(signal1d, 1, local_signal, padding[0:2], boundary,
                     0).squeeze()
        if self.comm.size == 1:
            self.assertTrue(torch.equal(padded_signal[:4], torch.tensor([0, 0, 0, 1 ])))
            self.assertTrue(torch.equal(padded_signal[-4:], torch.tensor([14, 15, 15, 15])))
        else:
            if self.comm.rank == 0:
                self.assertTrue(torch.equal(padded_signal[:4], torch.tensor([0, 0, 0, 1 ])))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
            elif self.comm.rank == self.comm.size - 1:
                self.assertTrue(torch.equal(padded_signal[-4:], torch.tensor([14, 15, 15, 15])))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
            else:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0])

        local_signal = signal2d.larray
        padded_signal = conv_pad(signal2d, 2, local_signal, padding, boundary, 0).squeeze()
        if self.comm.size == 1:
            self.assertTrue(torch.equal(padded_signal[0, 1:-1], torch.arange(16, dtype=torch.float32)))
            self.assertTrue(torch.equal(padded_signal[-1, :4], torch.tensor([0, 0, 1, 2])))
            self.assertTrue(torch.equal(padded_signal[-1, -4:], torch.tensor([13, 14, 15, 15])))
        else:
            if self.comm.rank == 0:
                self.assertTrue(torch.equal(padded_signal[1, 1:-1], torch.arange(16, dtype=torch.float32)))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
                self.assertTrue(padded_signal.shape[1] == local_signal.squeeze().shape[1] + 2 * 1)
            elif self.comm.rank == self.comm.size - 1:
                self.assertTrue(torch.equal(padded_signal[-2, 1:-1], torch.arange(16, dtype=torch.float32)))
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
                self.assertTrue(padded_signal.shape[1] == local_signal.squeeze().shape[1] + 2 * 1)
            else:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0])
                self.assertTrue(padded_signal.shape[1] == local_signal.squeeze().shape[1] + 2 * 1)

    def test_conv_pad_constant(self):
        signal1d = ht.arange(16, dtype=ht.float32).reshape(1, 1, 16)
        signal2d = ht.zeros((1, 1, 16, 16), dtype=ht.float32)
        signal2d[:] = ht.arange(16, dtype=ht.float32)
        padding = [2, 2, 1, 1]
        boundary = "constant"
        fillvalue = 100

        local_signal = signal1d.larray
        padded_signal = ht.array(
            conv_pad(signal1d, 1, local_signal, padding[0:2], boundary, fillvalue)).squeeze()
        self.assertTrue(padded_signal.shape[0] == 20)
        self.assertTrue(ht.all(padded_signal[:2] == fillvalue))
        self.assertTrue(ht.equal(padded_signal[-3:], ht.array([15, fillvalue, fillvalue])))

        local_signal = signal2d.larray
        padded_signal = ht.array(
            conv_pad(signal2d, 2, local_signal, padding, boundary, fillvalue)).squeeze()
        self.assertTrue(padded_signal.shape[0] == 20 and padded_signal.shape[1] == 18)
        self.assertTrue(ht.all(padded_signal[0:2,:] == fillvalue))
        self.assertTrue(ht.all(padded_signal[:,-1] == fillvalue))
        self.assertTrue(ht.equal(padded_signal[-3, -2:], ht.array([15, fillvalue])))

        # split axis not along convolution dimension
        signal1d.resplit_(1)
        signal2d.resplit_(1)

        local_signal = signal1d.larray
        padded_signal = conv_pad(signal1d, 1, local_signal, padding[0:2], boundary, fillvalue).squeeze()
        if self.comm.size == 1:
            self.assertTrue(padded_signal.shape[0] == 20)
            self.assertTrue((padded_signal[:2] == fillvalue).all())
            self.assertTrue(torch.equal(padded_signal[-3:], torch.tensor([15, fillvalue, fillvalue])))



        local_signal = signal2d.larray
        padded_signal = conv_pad(signal2d, 2, local_signal, padding, boundary, fillvalue).squeeze()
        if self.comm.size == 1:
            self.assertTrue(padded_signal.shape[0] == 20 and padded_signal.shape[1] == 18)
            self.assertTrue((padded_signal.squeeze()[0:2,:] == fillvalue).all())
            self.assertTrue((padded_signal.squeeze()[:,-1] == fillvalue).all())
            self.assertTrue(torch.equal(padded_signal.squeeze()[-3, -2:], torch.tensor([15, fillvalue])))

        # split axis along convolution dimension but comm size == 1
        signal1d.resplit_(2)
        signal2d.resplit_(2)

        local_signal = signal1d.larray
        padded_signal = conv_pad(signal1d, 1, local_signal, padding[0:2], boundary,
                     fillvalue).squeeze()
        if self.comm.size == 1:
            self.assertTrue(padded_signal.shape[0] == 20)
            self.assertTrue((padded_signal[:2] == fillvalue).all())
            self.assertTrue(torch.equal(padded_signal[-3:], torch.tensor([15, fillvalue, fillvalue])))
        else:
            if self.comm.rank == 0:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 2)
                self.assertTrue((padded_signal[:2] == fillvalue).all())
            elif self.comm.rank == self.comm.size - 1:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 2)
                self.assertTrue(torch.equal(padded_signal[-3:], torch.tensor([15, fillvalue, fillvalue])))
            else:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0])


        local_signal = signal2d.larray
        padded_signal = conv_pad(signal2d, 2, local_signal, padding, boundary,fillvalue).squeeze()
        if self.comm.size == 1:
            self.assertTrue(padded_signal.shape[0] == 20 and padded_signal.shape[1] == 18)
            self.assertTrue((padded_signal.squeeze()[0:2, :] == fillvalue).all())
            self.assertTrue((padded_signal.squeeze()[:, -1] == fillvalue).all())
            self.assertTrue(torch.equal(padded_signal.squeeze()[-3, -2:], torch.tensor([15, fillvalue])))
        else:
            if self.comm.rank == 0:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1*2)
                self.assertTrue(padded_signal.shape[1] == 18)
                self.assertTrue((padded_signal.squeeze()[0:2, :] == fillvalue).all())
            elif self.comm.rank == self.comm.size - 1:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0] + 1 * 2)
                self.assertTrue(padded_signal.shape[1] == 18)
                self.assertTrue((padded_signal.squeeze()[-2:0, :] == fillvalue).all())
            else:
                self.assertTrue(padded_signal.shape[0] == local_signal.squeeze().shape[0])
                self.assertTrue(padded_signal.shape[1] == 18)

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
        dis_signal = ht.arange(0, 9, split=0).astype(ht.int).reshape(3,3)
        kernel_odd = [[1], [1]]
        if self.comm.size > 1:
            with self.assertRaises(ValueError):
                ht.convolve2d(full_ones, kernel_even, mode="valid")
            with self.assertRaises(ValueError):
                ht.convolve2d(kernel_even, full_ones, mode="valid")

        if self.comm.size > 3:
            with self.assertRaises(ValueError):
                ht.convolve(dis_signal, kernel_odd)

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

        # avoid error for boundary == "replicate"
        if self.comm.size < 5:
            # distributed input along the first axis
            signal = ht.random.randn(10, 100, dtype=float_dtype)
            batch_signal = ht.empty((10, 10, 100), dtype=float_dtype, split=0)
            batch_signal.larray[:] = signal.larray

            # kernel without batch dimensions
            kernel = ht.random.randn(5, 19, dtype=float_dtype)
            batch_convolved = ht.convolve2d(batch_signal, kernel, mode="valid", boundary="circular")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="valid", boundary="circular"),
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
            batch_convolved = ht.convolve2d(batch_signal, batch_kernel, mode="same",boundary="reflect")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="same", boundary="reflect"),
                    batch_convolved[-1],
                )
            )

            # n-D batch convolution
            batch_signal = ht.empty((4, 5, 3, 10, 100), dtype=float_dtype, split=1)
            batch_signal.larray[:, :, :] = signal.larray
            batch_convolved = ht.convolve2d(batch_signal, kernel, mode="valid", boundary="replicate")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="valid", boundary="replicate"),
                    batch_convolved[1, 2, 0]
                )
            )

    def test_convolve2d_stride_batch_convolutions(self):
        float_dtype = ht.float32 if self.is_mps else ht.float64

        # avoid error for boundary == "replicate"
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
            batch_convolved = ht.convolve2d(batch_signal, dis_kernel, stride=stride, boundary="circular")
            self.assertTrue(
                ht.equal(ht.convolve2d(signal, kernel, stride=stride, boundary="circular"),
                         batch_convolved[5])
            )

            # batch kernel including resplit to signal axis
            stride = (3,4)
            batch_kernel = ht.empty((10, 5, 19), dtype=float_dtype, split=1)
            batch_kernel.larray[:] = dis_kernel.larray
            batch_convolved = ht.convolve2d(batch_signal, batch_kernel, mode="full",
                                          stride=stride, boundary="reflect")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="full", stride=stride, boundary="reflect"),
                    batch_convolved[-1],
                )
            )

            # n-D batch convolution
            stride = (4,3)
            batch_signal = ht.empty((4, 5, 3, 10, 100), dtype=float_dtype, split=1)
            batch_signal.larray[:, :, :] = signal.larray
            batch_convolved = ht.convolve2d(batch_signal, kernel, mode="valid", stride=stride, boundary="replicate")
            self.assertTrue(
                ht.equal(
                    ht.convolve2d(signal, kernel, mode="valid", stride=stride, boundary="replicate"),
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
        full_odd = ht.array(sig.convolve2d(np_sig, np_k_odd))

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_odd = ht.array(np_k_odd).astype(ht_dtype)
        dis_kernel_odd = ht.array(np_k_odd, split=0).astype(ht_dtype)

        #avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "same", "valid"]
            for i, mode in enumerate(modes):
                # odd kernel size
                print(mode)
                if not self.is_mps:
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
                    dis_signal, kernel, mode, stride, full_odd[::stride]
                )
                print("Test distributed kernel")
                self.assert_convolution_stride(
                    signal, dis_kernel, mode, stride, full_odd[::stride]
                )
                print("Test distributed kernel and signal")
                self.assert_convolution_stride(
                    dis_signal, dis_kernel, mode, stride, full_odd[::stride]
                )
            print("Test floats")
            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, full_odd[::stride]
            )
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, full_odd[::stride]
            )
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, full_odd[::stride]
            )

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
                    dis_signal, kernel, mode, stride, valid_odd[::stride]
                )
                print("Test distributed kernel only")
                self.assert_convolution_stride(
                    signal, dis_kernel, mode, stride, valid_odd[::stride]
                )

                self.assert_convolution_stride(
                    dis_signal, dis_kernel, mode, stride, valid_odd[::stride]
                )

            # different data types of input and kernel
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), kernel, mode, stride, valid_odd[::stride]
            )
            self.assert_convolution_stride(
                signal.astype(ht.float), dis_kernel, mode, stride, valid_odd[::stride]
            )
            self.assert_convolution_stride(
                dis_signal.astype(ht.float), dis_kernel, mode, stride, valid_odd[::stride]
            )

    def test_convolve2d_stride_kernel_odd_mode_full(self):
        ht_dtype = ht.int

        np_sig = np.arange(256).reshape((16, 16))
        np_sig = np_sig[:4,:]
        np_k_odd = np.arange(9).reshape((3, 3))
        full_odd = ht.array(sig.convolve2d(np_sig, np_k_odd)).astype(ht_dtype)

        mode = "full"
        strides = [(1,1), (1,2), (2,1), (2,2)]

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        np_k_odd_fl = np.array([[3,4,5],[6,7,8],[0,1,2]])
        kernel_odd = ht.array(np_k_odd).astype(ht_dtype)
        dis_kernel_odd = ht.array(np_k_odd, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        for stride in strides:
            print(stride)
            if self.comm.size <= 3:
                if not self.is_mps:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve2d(dis_signal, kernel_odd, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    self.assertTrue(ht.equal(full_odd[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(signal, dis_kernel_odd, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    print()
                    print(gathered)
                    print(full_odd[::stride[0], ::stride[1]])
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
                if not self.is_mps:
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
        np_k_even = np.ones(4).reshape((2, 2))
        full_even = ht.array(sig.convolve2d(np_sig, np_k_even)).astype(ht_dtype)

        mode = "full"
        strides = [(1, 2), (2, 1), (2, 2)]
        strides = [(2,1)]

        dis_signal = ht.array(np_sig, split=0).astype(ht_dtype)
        signal = ht.array(np_sig).astype(ht_dtype)

        kernel_even = ht.array(np_k_even).astype(ht_dtype)
        dis_kernel_even = ht.array(np_k_even, split=0).astype(ht_dtype)

        # avoid kernel larger than signal chunk
        for stride in strides:
            print(stride)
            if self.comm.size <= 3:
                if not self.is_mps:
                    # torch convolution does not support int on MPS
                    conv = ht.convolve2d(dis_signal, kernel_even, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
                    print(gathered)
                    print(full_even[::stride[0], ::stride[1]])
                    self.assertTrue(
                        ht.equal(full_even[::stride[0], ::stride[1]], gathered))

                    conv = ht.convolve2d(signal, dis_kernel_even, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)
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
        np_k_even = np.ones(4).reshape((2,2))
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
                if not self.is_mps:
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
        if self.comm.size <= 3:
            # prep
            np.random.seed(12)
            np_a = np.random.randint(1000, size=4418)
            np_b = np.random.randint(1000, size=1543)
            # torch convolution does not support int on MPS
            ht_dtype = ht.float32 if self.is_mps else ht.int32
            np_type = np.float32 if self.is_mps else np.int32
            random_stride = np.random.randint(1, high=len(np_a), size=1)[0]

            for mode in ["full", "same", "valid"]:
                strides = [1, random_stride] if mode != "same" else [1]
                for stride in strides:
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

    def test_convolve2d_large_signal_and_kernel_modes(self):
        if self.comm.size <= 3:
            np.random.seed(12)
            ht_dtype = ht.float32 if self.is_mps else ht.int32
            np_type = np.float32 if self.is_mps else np.int32

            np_a = np.random.randint(0,1000, size=(140, 250))
            np_b = np.random.randint(0,3, size=(39, 17))
            #np_b = np.ones((39,17))
            random_stride = tuple(np.random.randint(1, high=100, size=2))
            for mode in ["full", "same", "valid"]:
                strides = [(1,1), random_stride] if mode != "same" else [(1,1)]
                for stride in strides:
                    print("Stride", stride, "Mode", mode)
                    sc_conv = sig.convolve2d(np_a, np_b, mode=mode).astype(np_type)
                    solution = sc_conv[::stride[0], ::stride[1]]

                    a = ht.array(np_a, split=0, dtype=ht_dtype)
                    b = ht.array(np_b, split=None, dtype=ht_dtype)
                    conv = ht.convolve2d(a, b, mode=mode, stride=stride)
                    self.assert_array_equal(conv, solution)

                    b = ht.array(np_b, split=0, dtype=ht_dtype)
                    conv = ht.convolve2d(a, b, mode=mode, stride=stride)
                    gathered = manipulations.resplit(conv, axis=None)

                    print(gathered[0, :10])
                    print(solution[0, :10])
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
        ht_dtype = ht.float32 if self.is_mps else ht.int32

        signal = ht.arange(0, 16, dtype=ht_dtype).reshape(4,4)
        alt_signal = [[0,1,2,3], [4,5,6,7], [8,9,10,11],[12,13,14,15]]
        kernel = ht.ones(1,dtype=ht_dtype).reshape(1,1)

        for stride in range(1,4):
            conv = ht.convolve2d(alt_signal, kernel, stride=(stride,stride))
            self.assertTrue(ht.equal(signal[::stride, ::stride], conv))

            if not self.is_mps:
                conv = ht.convolve2d(1,5,stride=(stride,stride))
                self.assertTrue(ht.equal(ht.array([[5]]), conv))
