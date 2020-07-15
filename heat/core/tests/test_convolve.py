import numpy as np
import torch
import unittest
import heat as ht
from heat import manipulations

from .test_suites.basic_test import TestCase


class TestConvolve(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestConvolve, cls).setUpClass()

    def test_convolve1D(self):
        full_odd = ht.array(
            [0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 29, 15]
        ).astype(ht.int)
        full_even = ht.array(
            [0, 1, 3, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 42, 29, 15]
        ).astype(ht.int)

        signal = ht.arange(0, 16, split=0).astype(ht.int)
        kernal_odd = ht.ones(3).astype(ht.int)
        kernal_even = ht.ones(4).astype(ht.int)

        with self.assertRaises(TypeError):
            ht.convolve1D(signal, [0, 1, 2, 3])
        with self.assertRaises(TypeError):
            ht.convolve1D([0, 1, 2, 3], kernal_odd)
        with self.assertRaises(ValueError):
            s = signal.reshape((2, -1))
            ht.convolve1D(s, kernal_odd)
        with self.assertRaises(ValueError):
            k = ht.eye(3)
            ht.convolve1D(signal, k)
        with self.assertRaises(ValueError):
            ht.convolve1D(kernal_even, full_even)
        with self.assertRaises(TypeError):
            k = ht.ones(3).astype(ht.float)
            ht.convolve1D(signal, k)
        with self.assertRaises(ValueError):
            ht.convolve1D(signal, kernal_even, mode="same")
        with self.assertRaises(TypeError):
            k = ht.ones(4, split=0).astype(ht.int)
            ht.convolve1D(signal, k)

        # test modes
        modes = ["full", "same", "valid"]
        for i, mode in enumerate(modes):
            # odd kernal size
            conv = ht.convolve1D(signal, kernal_odd, mode=mode)
            conv.balance_()
            gathered = manipulations.resplit(conv, axis=None)
            self.assertTrue(ht.equal(full_odd[i : len(full_odd) - i], gathered))

            # even kernal size
            # skip mode 'same' for even kernals
            if mode != "same":
                conv = ht.convolve1D(signal, kernal_even, mode=mode)
                conv.balance_()
                gathered = manipulations.resplit(conv, axis=None)

                if mode == "full":
                    print(conv)
                    print(gathered)
                    self.assertTrue(ht.equal(full_even, gathered))
                else:
                    self.assertTrue(ht.equal(full_even[3:-3], gathered))

        # test different data type
        conv = ht.convolve1D(signal.astype(ht.float), kernal_odd.astype(ht.float))
        conv.balance_()
        gathered = manipulations.resplit(conv, axis=None)
        self.assertTrue(ht.equal(full_odd, gathered))
