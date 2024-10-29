import heat as ht
import unittest
import torch
import numpy as np

from ...tests.test_suites.basic_test import TestCase
from ..pd import _estimate_largest_singularvalue


class TestAuxiliaries(TestCase):
    def test_estimate_largest_singularvalue(self):
        # split = 0, float32
        x = ht.random.randn(100, 100, split=0, dtype=ht.float32)
        est = _estimate_largest_singularvalue(x)
        self.assertIsInstance(est, ht.DNDarray)
        self.assertTrue(est >= 0)
        self.assertTrue(est.dtype, ht.float32)
        self.assertTrue(est.item() >= np.linalg.svd(x.numpy(), compute_uv=False).max())

        # split = 1, float64
        x = ht.random.randn(100, 100, split=1, dtype=ht.float64)
        est = _estimate_largest_singularvalue(x, algorithm="fro")
        self.assertEqual(est.shape, ())
        self.assertEqual(est.device, x.device)
        self.assertTrue(est.dtype, ht.float64)
        self.assertTrue(est.item() >= np.linalg.svd(x.numpy(), compute_uv=False).max())

        # catch wrong inputs
        with self.assertRaises(NotImplementedError):
            est = _estimate_largest_singularvalue(x, algorithm="invalid")
        with self.assertRaises(TypeError):
            est = _estimate_largest_singularvalue(x, algorithm=1)

    def test_condest(self):
        # split = 0, float32 (actually split = 1, but due to transposition this yields split = 0 interally)
        x = ht.random.randn(25, 25 * ht.MPI_WORLD.size, split=1, dtype=ht.float32)
        est = ht.linalg.condest(x)
        self.assertIsInstance(est, ht.DNDarray)
        self.assertTrue(est >= 0)
        self.assertTrue(est.dtype, ht.float32)
        xnp = x.numpy()
        xnpsvals = np.linalg.svd(xnp, compute_uv=False)
        self.assertTrue(est.item() >= xnpsvals.max() / xnpsvals.min())

        # split = 1, float64
        # x = ht.random.randn(25 * ht.MPI_WORLD.size + 2, 25 * ht.MPI_WORLD.size + 1, split=1, dtype=ht.float64)
        # est = ht.linalg.condest(x, algorithm='randomized')
        # self.assertEqual(est.shape, ())
        # self.assertEqual(est.device, x.device)
        # self.assertTrue(est.dtype, ht.float64)
        # self.assertTrue(est.item() >= np.linalg.svd(x.numpy(),compute_uv=False).max())

        # catch wrong inputs
        with self.assertRaises(NotImplementedError):
            est = ht.linalg.condest(x, algorithm="invalid")
        with self.assertRaises(TypeError):
            est = ht.linalg.condest(x, algorithm=3.14)
        with self.assertRaises(ValueError):
            est = ht.linalg.condest(x, p=3)
