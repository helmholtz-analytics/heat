import heat as ht
import unittest
import numpy as np

from heat.testing.basic_test import TestCase
from heat.core.linalg.eigh import _subspaceiteration


class TestEigh(TestCase):
    def _check_eigh_result(self, X, Lambda, H):
        dtypetol = 1e-3 if X.dtype in (ht.float32, ht.complex64) else 1e-5
        self.assertEqual(Lambda.shape, (X.shape[0],))
        self.assertEqual(H.shape, X.shape)
        self.assertEqual(H.split, X.split)
        self.assertEqual(Lambda.split, 0)
        self.assertEqual(H.dtype, X.dtype)
        self.assertEqual(Lambda.dtype, X.real.dtype)
        X_rec = H @ ht.diag(Lambda) @ H.conj().T
        self.assertTrue(ht.norm(X - X_rec) / ht.norm(X) < dtypetol)
        HtH = H.conj().T @ H
        eye_size_H = ht.eye(HtH.shape[0], split=HtH.split, dtype=X.dtype)
        self.assertTrue(ht.norm(HtH - eye_size_H) / ht.norm(eye_size_H) < dtypetol)

    def test_subspaceiteration_complex(self):
        rng = np.random.default_rng(42)
        random_state = ht.random.get_state()
        try:
            for generator in ("Batchparallel", "Threefry"):
                ht.random.set_state((generator, 42, 0))
                for dtype, tol in ((ht.complex64, 1e-4), (ht.complex128, 1e-8)):
                    for n, k in ((17, 6), (4, 2)):
                        data = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
                        basis, _ = np.linalg.qr(data)
                        spectrum = np.linspace(-2.0, 3.0, n)
                        a_np = (basis * spectrum) @ basis.conj().T
                        c_np = basis[:, :k] @ basis[:, :k].conj().T
                        for split in (None, 0, 1):
                            A = ht.array(a_np, dtype=dtype, split=split, device=self.device)
                            C = ht.array(c_np, dtype=dtype, split=split, device=self.device)
                            for options in ({}, {"maxit": 2, "tol": 0.0, "silent": False}):
                                with self.subTest(
                                    generator=generator,
                                    dtype=dtype,
                                    n=n,
                                    split=split,
                                    options=options,
                                ):
                                    Q, rank = _subspaceiteration(A, C, depth=1, **options)
                                    self.assertEqual(rank, k)
                                    self.assertEqual(Q.shape, (n, n))
                                    self.assertEqual(Q.dtype, dtype)
                                    self.assertEqual(Q.split, split if A.comm.size > 1 else None)
                                    self.assertEqual(Q.device, A.device)
                                    self.assertEqual(Q.comm, A.comm)
                                    q_np = Q.numpy()
                                    np.testing.assert_allclose(
                                        q_np.conj().T @ q_np, np.eye(n), atol=tol, rtol=tol
                                    )
                                    np.testing.assert_allclose(
                                        q_np[:, :k] @ q_np[:, :k].conj().T,
                                        c_np,
                                        atol=tol,
                                        rtol=tol,
                                    )
                                    residual = q_np[:, k:].conj().T @ a_np @ q_np[:, :k]
                                    self.assertLess(
                                        np.linalg.norm(residual) / np.linalg.norm(a_np), tol
                                    )
        finally:
            ht.random.set_state(random_state)

    def test_eigh_complex(self):
        rng = np.random.default_rng(42)
        for dtype, tol in ((ht.complex64, 1e-3), (ht.complex128, 1e-5)):
            for n in (24, 25):
                data = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
                data += data.conj().T.copy()
                expected = np.linalg.eigvalsh(data)[::-1]
                for split in (None, 0, 1):
                    with self.subTest(dtype=dtype, n=n, split=split):
                        A = ht.array(data, dtype=dtype, split=split, device=self.device)
                        eigenvalues, eigenvectors = ht.linalg.eigh(A)
                        self._check_eigh_result(A, eigenvalues, eigenvectors)
                        np.testing.assert_allclose(
                            eigenvalues.numpy(), expected, atol=tol, rtol=tol
                        )
                        self.assertEqual(eigenvectors.device, A.device)
                        self.assertEqual(eigenvalues.device, A.device)
                        self.assertEqual(eigenvectors.comm, A.comm)
                        self.assertEqual(eigenvalues.comm, A.comm)

    def test_eigh(self):
        # test with default values
        splits = [None, 0, 1]
        dtypes = [ht.float32, ht.float64]
        for split in splits:
            for dtype in dtypes:
                with self.subTest(split=split, dtype=dtype):
                    X = ht.random.randn(100, 100, split=split, dtype=dtype)
                    X = X + X.T.resplit_(X.split)
                    Lambda, H = ht.linalg.eigh(X)
                    self._check_eigh_result(X, Lambda, H)

        # test edge case discussed in #2171
        if ht.comm.size == 2:  # with some communicator sizes this fails, see Issue #2093
            X = ht.ones((32, 32), split=0, dtype=ht.float64)
            X = X + X.T.resplit_(X.split)
            Lambda, H = ht.linalg.eigh(X)
            self._check_eigh_result(X, Lambda, H)

    def test_eigh_options(self):
        # test non-default options
        X = ht.random.randn(101, 101, split=0, dtype=ht.float32)
        X = X @ X.T
        Lambda, H = ht.linalg.eigh(X, r_max_zolopd=1, silent=False)
        self._check_eigh_result(X, Lambda, H)

    def test_eigh_catch_wrong_inputs(self):
        with self.assertRaises(TypeError):
            ht.linalg.eigh("not an array")
        with self.assertRaises(TypeError):
            ht.linalg.eigh(ht.eye(4, dtype=ht.int32))
        with self.assertRaises(ValueError):
            ht.linalg.eigh(ht.zeros((2, 2, 2)))
        with self.assertRaises(TypeError):
            _subspaceiteration(ht.eye(4, dtype=ht.int32), ht.eye(4))
        # non-square DNDarray as input
        X = ht.random.rand(100, 101, split=0, dtype=ht.float32)
        with self.assertRaises(ValueError):
            ht.linalg.eigh(X)

        # r_max_zolopd not of right type
        X = ht.random.rand(100, 100, split=0, dtype=ht.float32)
        with self.assertRaises(ValueError):
            ht.linalg.eigh(X, r_max_zolopd=2.2)
