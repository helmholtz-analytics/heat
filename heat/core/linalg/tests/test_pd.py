import heat as ht
import unittest
import torch
import numpy as np

from ...tests.test_suites.basic_test import TestCase


class TestZoloPD(TestCase):
    def _check_pd(self, A, U, H, dtypetol):
        # check whether output has right type, shape and dtype
        self.assertTrue(isinstance(U, ht.DNDarray))
        self.assertEqual(U.shape, A.shape)
        self.assertEqual(U.dtype, A.dtype)
        self.assertTrue(isinstance(H, ht.DNDarray))
        self.assertEqual(H.shape, (A.shape[1], A.shape[1]))
        self.assertEqual(H.dtype, A.dtype)

        # check whether output is correct
        A_np = A.numpy()
        U_np = U.numpy()
        H_np = H.numpy()
        # U orthogonal
        self.assertTrue(
            np.allclose(U_np.T @ U_np, np.eye(U_np.shape[1]), atol=dtypetol, rtol=dtypetol)
        )
        # H symmetric
        self.assertTrue(np.allclose(H_np.T, H_np, atol=dtypetol, rtol=dtypetol))
        # H positive definite, i.e., eigenvalues > 0
        self.assertTrue((np.linalg.eigvalsh(H_np) > 0).all())
        # A = U H
        self.assertTrue(np.allclose(A_np, U_np @ H_np, atol=dtypetol, rtol=dtypetol))

    def test_catch_wrong_inputs(self):
        # if A is not a DNDarray
        with self.assertRaises(TypeError):
            ht.pd("I am clearly not a DNDarray. Do you mind?")
        # test wrong input dimension
        with self.assertRaises(ValueError):
            ht.pd(ht.zeros((10, 10, 10), dtype=ht.float32))
        # test wrong input shape
        with self.assertRaises(ValueError):
            ht.pd(ht.random.rand(10, 11, dtype=ht.float32))
        # test wrong input dtype
        with self.assertRaises(TypeError):
            ht.pd(ht.ones((10, 10), dtype=ht.int32))
        # wrong input for r
        with self.assertRaises(ValueError):
            ht.pd(ht.ones((11, 10)), r=1.0)
        # wrong input for tol
        with self.assertRaises(TypeError):
            ht.pd(ht.ones((11, 10)), r=2, condition_estimate=1)

    def test_pd_split0(self):
        # split=0, float32, no condition estimate provided, silent mode
        ht.random.seed(18112024)
        for r in range(1, 9):
            A = ht.random.randn(100, 10 * r, split=0, dtype=ht.float32)
            U, H = ht.pd(A, r=r)
            dtypetol = 1e-4

            self._check_pd(A, U, H, dtypetol)

        # cases not covered so far
        A = ht.random.randn(100, 100, split=0, dtype=ht.float64)
        U, H = ht.pd(A, condition_estimate=1.0e16, silent=False)
        dtypetol = 1e-8

        self._check_pd(A, U, H, dtypetol)

        # case without calculating H
        A = ht.random.randn(100, 10, split=0, dtype=ht.float32)
        U = ht.pd(A, calcH=False)
        U_np = U.numpy()
        self.assertTrue(np.allclose(U_np.T @ U_np, np.eye(U_np.shape[1]), atol=1e-4, rtol=1e-4))
        H_np = U_np.T @ A.numpy()
        self.assertTrue(np.allclose(H_np.T, H_np, atol=1e-4, rtol=1e-4))
        self.assertTrue((np.linalg.eigvalsh(H_np) > 0).all())

    def test_pd_split1(self):
        # split=1, float64, condition estimate provided, non-silent mode
        ht.random.seed(623)
        for r in range(1, 9):
            A = ht.random.randn(100, 99, split=1, dtype=ht.float64)
            U, H = ht.pd(A, r=r, silent=False, condition_estimate=1.0e16)
            dtypetol = 1e-8

            self._check_pd(A, U, H, dtypetol)

        # cases not covered so far
        A = ht.random.randn(100, 99, split=1, dtype=ht.float32)
        U, H = ht.pd(A)
        dtypetol = 1e-4
        self._check_pd(A, U, H, dtypetol)

        # case without calculating H
        A = ht.random.randn(100, 100, split=1, dtype=ht.float64)
        U = ht.pd(A, calcH=False)
        U_np = U.numpy()
        self.assertTrue(np.allclose(U_np.T @ U_np, np.eye(U_np.shape[1]), atol=1e-8, rtol=1e-8))
        H_np = U_np.T @ A.numpy()
        self.assertTrue(np.allclose(H_np.T, H_np, atol=1e-8, rtol=1e-8))
        self.assertTrue((np.linalg.eigvalsh(H_np) > 0).all())
