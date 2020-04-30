import torch
import os
import unittest
import heat as ht
import numpy as np

from heat.core.tests.test_suites.basic_test import TestCase


class TestSolver(TestCase):
    def test_cg(self):
        size = ht.communication.MPI_WORLD.size * 3
        b = ht.arange(1, size + 1, dtype=ht.float32, split=0, device=self.ht_device)
        A = ht.manipulations.diag(b)
        x0 = ht.random.rand(size, dtype=b.dtype, split=b.split, device=self.ht_device)

        x = ht.ones(b.shape, dtype=b.dtype, split=b.split, device=self.ht_device)

        res = ht.linalg.cg(A, b, x0)
        self.assertTrue(ht.allclose(x, res, atol=1e-3))

        b_np = np.arange(1, size + 1)
        with self.assertRaises(TypeError):
            ht.linalg.cg(A, b_np, x0)

        with self.assertRaises(RuntimeError):
            ht.linalg.cg(A, A, x0)
        with self.assertRaises(RuntimeError):
            ht.linalg.cg(b, b, x0)
        with self.assertRaises(RuntimeError):
            ht.linalg.cg(A, b, A)
