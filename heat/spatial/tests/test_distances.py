import unittest
import os

import torch

import heat as ht
import numpy as np

if os.environ.get("DEVICE") == "gpu" and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    torch.cuda.set_device(device)


class TestDistances(unittest.TestCase):
    def test_cdist(self):
        split = None
        X = ht.ones((4, 4), dtype=ht.float32, split=split, device=ht_device)
        Y = ht.zeros((4, 4), dtype=ht.float32, split=None, device=ht_device)

        res = ht.ones((4, 4), dtype=ht.float32, split=split) * 2

        d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        self.assertTrue(ht.equal(d, res))
        self.assertEqual(d.split, split)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=True)
        self.assertTrue(ht.equal(d, res))
        self.assertEqual(d.split, split)

        split = 0
        X = ht.ones((4, 4), dtype=ht.float32, split=split)
        res = ht.ones((4, 4), dtype=ht.float32, split=split) * 2

        d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        self.assertTrue(ht.equal(d, res))
        self.assertEqual(d.split, split)

        d = ht.spatial.cdist(X, Y, quadratic_expansion=True)
        self.assertTrue(ht.equal(d, res))
        self.assertEqual(d.split, split)

        Y = ht.zeros((4, 4), dtype=ht.float32, split=split)

        with self.assertRaises(NotImplementedError):
            d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        with self.assertRaises(NotImplementedError):
            d = ht.spatial.cdist(X, Y, quadratic_expansion=True)

        X = ht.ones((4, 4), dtype=ht.float32, split=1)
        Y = ht.zeros((4, 4), dtype=ht.float32, split=None)

        with self.assertRaises(NotImplementedError):
            d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
        with self.assertRaises(NotImplementedError):
            d = ht.spatial.cdist(X, Y, quadratic_expansion=True)

        X = ht.ones((4, 4), dtype=ht.float64, split=0)
        Y = ht.zeros((4, 4), dtype=ht.float32, split=None)
        with self.assertRaises(NotImplementedError):
            d = ht.spatial.cdist(X, Y, quadratic_expansion=False)
