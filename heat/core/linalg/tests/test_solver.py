import torch
import os
import unittest
import heat as ht
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestSolver(TestCase):
    def test_cg(self):
        size = ht.communication.MPI_WORLD.size * 3
        b = ht.arange(1, size + 1, dtype=ht.float32, split=0)
        A = ht.manipulations.diag(b)
        x0 = ht.random.rand(size, dtype=b.dtype, split=b.split)

        x = ht.ones(b.shape, dtype=b.dtype, split=b.split)

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

    def test_lanczos(self):
        # define positive definite matrix (n,n), split = 0
        n = 100
        A = ht.random.randn(n, n, dtype=ht.float64, split=0)
        B = A @ A.T
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n)
        # V must be unitary
        V_inv = ht.linalg.inv(V)
        self.assertTrue(ht.allclose(V_inv, V.T))
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V_inv
        self.assertTrue(ht.allclose(lanczos_B, B))

        # define positive definite matrix (n,n), split = 1
        A = ht.random.randn(n, n, dtype=ht.float64, split=1)
        B = A @ A.T
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n)
        # V must be unitary
        V_inv = ht.linalg.inv(V)
        self.assertTrue(ht.allclose(V_inv, V.T))
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V_inv
        self.assertTrue(ht.allclose(lanczos_B, B))
