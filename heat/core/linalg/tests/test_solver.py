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
        self.assertTrue(V.dtype is B.dtype)
        self.assertTrue(T.dtype is B.dtype)
        # V must be unitary
        V_inv = ht.linalg.inv(V)
        self.assertTrue(ht.allclose(V_inv, V.T))
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V_inv
        self.assertTrue(ht.allclose(lanczos_B, B))

        # float32
        A = ht.random.randn(n, n, dtype=ht.float32, split=0)
        B = A @ A.T
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n)
        self.assertTrue(V.dtype is B.dtype)
        self.assertTrue(T.dtype is B.dtype)
        # V must be unitary
        V_inv = ht.linalg.inv(V)
        if int(torch.__version__.split(".")[1]) == 13:
            tolerance = 1e-3
        else:
            tolerance = 1e-4
        self.assertTrue(ht.allclose(V_inv, V.T, atol=tolerance))
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V_inv
        self.assertTrue(ht.allclose(lanczos_B, B, atol=tolerance))

        # complex128
        A = (
            ht.random.randn(n, n, dtype=ht.float64, split=0)
            + ht.random.randn(n, n, dtype=ht.float64, split=0) * 1j
        )
        A_conj = ht.conj(A)
        B = A @ A_conj.T
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n)
        # V must be unitary
        V_inv = ht.linalg.inv(V)
        # V T V* must be = B, V conjugate transpose = V inverse
        lanczos_B = V @ T @ V_inv

        with self.assertRaises(TypeError):
            V, T = ht.lanczos(B, m="3")
        with self.assertRaises(TypeError):
            A = ht.random.randint(0, 5, (10, 10))
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(TypeError):
            A = torch.randn(10, 10)
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(TypeError):
            A = ht.random.randn(10, 12)
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(RuntimeError):
            A = ht.random.randn(10, 12, 12)
            V, T = ht.lanczos(A, m=3)
        with self.assertRaises(NotImplementedError):
            A = ht.random.randn(10, 10, split=1)
            V, T = ht.lanczos(A, m=3)
