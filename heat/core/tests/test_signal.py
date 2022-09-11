import numpy as np
import torch
import unittest
import heat as ht
from heat import manipulations
import scipy.signal as sig
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

        signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = [1, 1, 1, 1]

        with self.assertRaises(TypeError):
            signal_wrong_type = [0, 1, 2, "tre", 4, "five", 6, "ʻehiku", 8, 9, 10]
            ht.convolve(signal_wrong_type, kernel_odd, mode="full")
        with self.assertRaises(TypeError):
            filter_wrong_type = [1, 1, "pizza", "pineapple"]
            ht.convolve(signal, filter_wrong_type, mode="full")
        with self.assertRaises(ValueError):
            ht.convolve(signal, kernel_odd, mode="invalid")
        with self.assertRaises(ValueError):
            s = signal.reshape((2, -1))
            ht.convolve(s, kernel_odd)
        with self.assertRaises(ValueError):
            k = ht.eye(3)
            ht.convolve(signal, k)
        with self.assertRaises(ValueError):
            ht.convolve(kernel_even, full_even)
        with self.assertRaises(ValueError):
            ht.convolve(signal, kernel_even, mode="same")
        if self.comm.size > 1:
            with self.assertRaises(TypeError):
                k = ht.ones(4, split=0).astype(ht.int)
                ht.convolve(signal, k)
        if self.comm.size >= 5:
            with self.assertRaises(ValueError):
                ht.convolve(signal, kernel_even, mode="valid")

        # test modes, avoid kernel larger than signal chunk
        if self.comm.size <= 3:
            modes = ["full", "same", "valid"]
            for i, mode in enumerate(modes):
                # odd kernel size
                conv = ht.convolve(signal, kernel_odd, mode=mode)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))
                # different data types
                conv = ht.convolve(signal.astype(ht.float), kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                # even kernel size
                # skip mode 'same' for even kernels
                if mode != "same":
                    conv = ht.convolve(signal, kernel_even, mode=mode)
                    gathered = manipulations.resplit(conv, axis=None)

                    if mode == "full":
                        self.assertTrue(ht.equal(full_even, gathered))
                    else:
                        self.assertTrue(ht.equal(full_even[3:-3], gathered))

        # test edge cases
        # non-distributed signal, size-1 kernel
        signal = ht.arange(0, 16).astype(ht.int)
        alt_signal = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        kernel = ht.ones(1).astype(ht.int)
        conv = ht.convolve(alt_signal, kernel)
        self.assertTrue(ht.equal(signal, conv))

    def test_convolve2d(self):

        dis_signal = ht.arange(256, split=0).reshape((16, 16)).astype(ht.int)
        signal = ht.arange(256).reshape((16, 16)).astype(ht.int)

        kernel_odd = ht.arange(9).reshape((3, 3)).astype(ht.int)
        kernel_even = ht.arange(16).reshape((4, 4)).astype(ht.int)

        np_sig = np.arange(256).reshape((16, 16))
        np_k_odd = np.arange(9).reshape((3, 3))
        np_k_even = np.arange(16).reshape((4, 4))

        full_odd = ht.array(sig.convolve2d(np_sig, np_k_odd))
        full_even = ht.array(sig.convolve2d(np_sig, np_k_even))

        dis_kernel_odd = ht.arange(9, split=0).reshape((3, 3)).astype(ht.int)
        dis_kernel_even = ht.arange(16, split=0).reshape((4, 4)).astype(ht.int)

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
                conv = ht.convolve2d(dis_signal.astype(ht.float), kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve2d(dis_signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

                conv = ht.convolve2d(signal.astype(ht.float), dis_kernel_odd)
                gathered = manipulations.resplit(conv, axis=None)
                self.assertTrue(ht.equal(full_odd.astype(ht.float), gathered))

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

                a = ht.array(np_a, split=0, dtype=ht.int32)
                b = ht.array(np_b, split=0, dtype=ht.int32)
                conv = ht.convolve2d(a, b, mode=mode)
                self.assert_array_equal(conv, sc_conv)

                a = ht.array(np_a, split=1, dtype=ht.int32)
                b = ht.array(np_b, split=1, dtype=ht.int32)
                conv = ht.convolve2d(a, b, mode=mode)
                self.assert_array_equal(conv, sc_conv)

        # test edge cases
        # non-distributed signal, size-1 kernel
        signal = ht.arange(0, 16).reshape(4, 4).astype(ht.int)
        alt_signal = ht.arange(16).reshape(4, 4).astype(ht.int)
        kernel = ht.ones(1).reshape((1, 1)).astype(ht.int)
        conv = ht.convolve2d(alt_signal, kernel)
        self.assertTrue(ht.equal(signal, conv))
