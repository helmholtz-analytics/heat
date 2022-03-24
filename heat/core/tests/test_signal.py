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

        signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernel_odd = ht.ones(3).astype(ht.int)
        kernel_even = ht.ones(4).astype(ht.int)

        with self.assertRaises(TypeError):
            ht.convolve(signal, [0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.convolve([0, 1, 2, 3], kernel_odd)
        with self.assertRaises(ValueError):
            s = signal.reshape((2, -1))
            ht.convolve(s, kernel_odd)
        with self.assertRaises(ValueError):
            k = ht.eye(3)
            ht.convolve(signal, k)
        with self.assertRaises(ValueError):
            ht.convolve(kernel_even, full_even)
        with self.assertRaises(TypeError):
            k = ht.ones(3).astype(ht.float)
            ht.convolve(signal, k)
        with self.assertRaises(ValueError):
            ht.convolve(signal, kernel_even, mode="same")
        with self.assertRaises(TypeError):
            k = ht.ones(4, split=0).astype(ht.int)
            ht.convolve(signal, k)

        # test modes
        modes = ["full", "same", "valid"]
        for i, mode in enumerate(modes):
            # odd kernel size
            conv = ht.convolve(signal, kernel_odd, mode=mode)
            conv.balance_()
            gathered = manipulations.resplit(conv, axis=None)
            self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))

            # even kernel size
            # skip mode 'same' for even kernels
            if mode != "same":
                conv = ht.convolve(signal, kernel_even, mode=mode)
                conv.balance_()
                gathered = manipulations.resplit(conv, axis=None)

                if mode == "full":
                    print(conv)
                    print(gathered)
                    self.assertTrue(ht.equal(full_even, gathered))
                else:
                    self.assertTrue(ht.equal(full_even[3:-3], gathered))

        # test different data type
        conv = ht.convolve(signal.astype(ht.float), kernel_odd.astype(ht.float))
        conv.balance_()
        gathered = manipulations.resplit(conv, axis=None)
        self.assertTrue(ht.equal(full_odd, gathered))
