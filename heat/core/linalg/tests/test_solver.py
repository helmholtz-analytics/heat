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

        # complex128, output buffers
        A = (
            ht.random.rand(n, n, dtype=ht.float64, split=0)
            + ht.random.rand(n, n, dtype=ht.float64, split=0) * 1j
        )
        A_conj = ht.conj(A)
        B = A @ A_conj.T
        m = n
        V_out = ht.zeros((n, m), dtype=B.dtype, split=B.split, device=B.device, comm=B.comm)
        T_out = ht.zeros((m, m), dtype=ht.float64, device=B.device, comm=B.comm)
        # Lanczos decomposition with iterations m = n
        ht.lanczos(B, m=m, V_out=V_out, T_out=T_out)
        # V must be unitary
        V_inv = ht.linalg.inv(V_out)
        self.assertTrue(ht.allclose(V_inv, ht.conj(V_out).T))
        # V T V* must be = B, V conjugate transpose = V inverse
        lanczos_B = V_out @ T_out @ V_inv
        self.assertTrue(ht.allclose(lanczos_B, B))

        # single precision tolerance
        if (
            int(torch.__version__.split(".")[0]) == 1
            and int(torch.__version__.split(".")[1]) >= 13
            or int(torch.__version__.split(".")[0]) > 1
        ):
            tolerance = 1e-3
        else:
            tolerance = 1e-4

        # float32, pre_defined v0, split mismatch
        A = ht.random.randn(n, n, dtype=ht.float32, split=0)
        B = A @ A.T
        v0 = ht.random.randn(n, device=A.device, split=None)
        v0 = v0 / ht.norm(v0)
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n, v0=v0)
        self.assertTrue(V.dtype is B.dtype)
        self.assertTrue(T.dtype is B.dtype)
        # V must be unitary
        V_inv = ht.linalg.inv(V)
        self.assertTrue(ht.allclose(V_inv, V.T, atol=tolerance))
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V_inv
        print("DEBUGGING: residuals: ", lanczos_B - B)
        self.assertTrue(ht.allclose(lanczos_B, B, atol=tolerance))

        # complex64
        A = (
            ht.random.randn(n, n, dtype=ht.float32, split=0)
            + ht.random.randn(n, n, dtype=ht.float32, split=0) * 1j
        )
        A_conj = ht.conj(A)
        B = A @ A_conj.T
        # Lanczos decomposition with iterations m = n
        V, T = ht.lanczos(B, m=n)
        # V must be unitary
        # V T V* must be = B, V conjugate transpose = V inverse
        V_conj = ht.conj(V)
        lanczos_B = V @ T @ V_conj.T
        self.assertTrue(ht.allclose(lanczos_B, B, atol=tolerance))

        # non-distributed
        A = ht.random.randn(n, n, dtype=ht.float64, split=None)
        B = A @ A.T
        # Lanczos decomposition with iterations m = n
        m = n
        V_out = ht.zeros((n, m), dtype=B.dtype, split=B.split, device=B.device, comm=B.comm)
        T_out = ht.zeros((m, m), dtype=ht.float64, device=B.device, comm=B.comm)
        ht.lanczos(B, m=m, V_out=V_out, T_out=T_out)
        self.assertTrue(V_out.dtype is B.dtype)
        self.assertTrue(T_out.dtype is B.real.dtype)
        # V must be unitary
        V_inv = ht.linalg.inv(V_out)
        self.assertTrue(ht.allclose(V_inv, V_out.T))
        # without output buffers
        V, T = ht.lanczos(B, m=m)
        # V T V.T must be = B, V transposed = V inverse
        lanczos_B = V @ T @ V.T
        self.assertTrue(ht.allclose(lanczos_B, B))

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
